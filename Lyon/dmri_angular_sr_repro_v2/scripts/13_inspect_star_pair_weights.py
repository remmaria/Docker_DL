#!/usr/bin/env python3
"""
Etapa 13 (diagnostico): inspeciona os pesos de fusao (`pi`, ver
model/rrin3d_star.py:PairWeightHead3D/RRIN3DStar.forward com
return_pairs=True) de uma RRIN3DStar treinada, num UNICO voxel especifico,
para TODOS os alvos held-out do feixe de trincas.

Motivacao: a hipotese aventada (ver addendum, discussao do glifo "torto" de
star610 no voxel de cruzamento row1/col3 da figura de
scripts/12_visualize_fod_glyphs.py) e que, em voxels de cruzamento dificeis,
a rede pode concentrar quase todo o peso de fusao (`pi` proximo de 1.0) num
UNICO par candidato cujo warp esta mal-alinhado geometricamente pra aquela
direcao-alvo, herdando a distorcao daquele par em vez de diluir o erro entre
varios pares (o que a media uniforme do naive_ensemble_blend faria, ao custo
de borrar tudo -- ver secao 20.4 do addendum). Este script permite checar
isso diretamente: para cada direcao-alvo reconstruida pela rede, imprime o
peso de fusao de cada par candidato, a predicao de sinal de cada par
NAQUELE voxel, e duas metricas resumo:
  - concentracao: peso maximo entre os pares validos (proximo de 1/M_valido
    = uniforme; proximo de 1.0 = toda a confianca num so par)
  - discordancia: desvio padrao da predicao de sinal entre os pares
    validos NAQUELE voxel (proximo de 0 = pares concordam, nao importa
    muito qual pese mais; alto = pares discordam MUITO, e a escolha de
    peso importa de verdade)
Quando concentracao alta E discordancia alta coincidem no mesmo alvo, e
evidencia direta a favor da hipotese (rede "aposta tudo" num par que
discorda dos outros).

NAO reconstroi o volume inteiro (seria o papel de
scripts/05f_reconstruct_rrin_star.py) -- so roda UM forward num patch
pequeno centrado no voxel pedido, o suficiente pra popular os campos
conv3d ao redor dele. Por isso o resultado de `pred` aqui pode diferir
ligeiramente do valor final salvo por 05f nesse voxel, se 05f tiver
combinado por media de patches sobrepostos (--stride < --patch-size) que
cobrem o voxel a partir de origens diferentes -- efeito de borda de
convolucao, tipicamente pequeno. Isso e' aceitavel para este diagnostico
(queremos ENTENDER o comportamento de fusao, nao reproduzir o pipeline de
producao voxel a voxel).

A funcao `centered_patch` abaixo duplica (por convencao do projeto: sem
cross-import entre scripts numerados) a mesma logica de
scripts/12_visualize_fod_glyphs.py:centered_patch -- centra um patch cubico
num voxel, com clipping nas bordas do volume.

Uso:
    python scripts/13_inspect_star_pair_weights.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --checkpoint work_dir/rrin_star_checkpoints/shell1000_n16_star3/best.pt \
        --shell-b 1000 --n-level 16 \
        --subject sub-01 \
        --voxel "59,58,26" \
        --out work_dir/diagnostics/pair_weights_sub-01_vox59-58-26.csv

Requer PyTorch (+ GPU opcional, CPU funciona pra um voxel so). Nao
executado neste ambiente -- verificado so por py_compile e por um teste
sintetico standalone de `centered_patch` (mesma funcao ja testada em
12_visualize_fod_glyphs.py).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, load_dwi, split_shells
from utils.masking import load_or_build_mask
from utils.dataset import _resolve_shell_key
from model.rrin3d_star import build_star_model


def centered_patch(local_center, patch_size, shape):
    """Centra um patch cubico `patch_size^3` no voxel `local_center` (x,y,z
    em coordenadas LOCAIS do volume `shape`), com clipping nas bordas --
    MESMA logica/espirito de scripts/12_visualize_fod_glyphs.py:
    centered_patch (duplicada aqui por convencao do projeto: sem
    cross-import entre scripts numerados), so que em 3D (os 3 eixos, nao 2
    -- aqui nao ha slice_axis porque o forward do modelo precisa do volume
    3D inteiro do patch, nao de uma fatia 2D). Retorna a origem (x,y,z) do
    patch, ou None se `shape` for menor que `patch_size` em algum eixo ou
    se o voxel central cair fora do volume."""
    if any(shape[d] < patch_size for d in range(3)):
        return None
    if any(local_center[d] < 0 or local_center[d] >= shape[d] for d in range(3)):
        return None
    origin = []
    for d in range(3):
        lo = local_center[d] - patch_size // 2
        lo = max(0, min(lo, shape[d] - patch_size))
        origin.append(lo)
    return tuple(origin)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="mesma pasta/convencao de scripts/05f_reconstruct_rrin_star.py")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--subject", required=True,
                     help="tag do sujeito (subject ou subject_session), UM so por rodada")
    ap.add_argument("--voxel", required=True,
                     help="'X,Y,Z' em coordenadas GLOBAIS do volume (mesmo sistema impresso "
                          "por scripts/12_visualize_fod_glyphs.py)")
    ap.add_argument("--patch-size", type=int, default=10,
                     help="tamanho do patch cubico rodado pelo modelo ao redor do voxel "
                          "(default 10, mesmo default de 05f_reconstruct_rrin_star.py -- "
                          "nao precisa bater com o --patch-size do script 12, que e so pra "
                          "exibicao 2D do glifo, nao pro forward do modelo)")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--top", type=int, default=None,
                     help="se dado, imprime so os --top alvos com maior concentracao de peso "
                          "(peso_max), pra focar nos casos mais extremos; default: todos")
    ap.add_argument("--out", default=None,
                     help="se dado, tambem salva a tabela completa (todos os alvos x todos os "
                          "pares) em CSV nesse caminho")
    args = ap.parse_args()

    parts = [p.strip() for p in args.voxel.split(",")]
    if len(parts) != 3:
        sys.exit(f"--voxel precisa ser 'X,Y,Z' (3 valores), recebi {args.voxel!r}.")
    try:
        voxel_global = tuple(int(p) for p in parts)
    except ValueError:
        sys.exit(f"--voxel precisa ser 3 inteiros separados por virgula, recebi {args.voxel!r}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    use_quality_cond = ckpt_args.get("use_quality_cond", False)
    norm_type = ckpt_args.get("norm_type", "instance")
    ensemble_m = ckpt_args.get("ensemble_m")
    if ensemble_m is None:
        sys.exit(f"checkpoint {args.checkpoint} nao tem 'ensemble_m' em args -- confira se e "
                  f"mesmo um checkpoint da etapa 4e (RRIN3DStar), nao de par unico.")
    model = build_star_model(base_ch=ckpt_args.get("base_ch", 16),
                              max_disp=ckpt_args.get("max_disp", 0.5),
                              use_quality_cond=use_quality_cond,
                              norm_type=norm_type).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint carregado (epoca {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss')}, "
          f"ensemble_m={ensemble_m}, norm_type={norm_type})", flush=True)

    entries = [e for e in load_manifest(args.manifest)]

    def _tag_of(e):
        return e.subject if not e.session else f"{e.subject}_{e.session}"

    matches = [e for e in entries if _tag_of(e) == args.subject]
    if not matches:
        sys.exit(f"--subject {args.subject!r} nao encontrado no manifesto.")
    e = matches[0]

    key = f"{args.shell_b}__{args.n_level}"
    trip_path = Path(args.triplets_dir) / f"{args.subject}_rrin_triplets.npz"
    if not trip_path.exists():
        sys.exit(f"{trip_path} nao existe.")
    trip = np.load(trip_path)
    if f"{key}__ens_pair_a" not in trip.files:
        sys.exit(f"sem campos '{key}__ens_pair_a' em {trip_path.name} -- rode "
                  f"scripts/02b_build_rrin_triplets.py --ensemble-m {ensemble_m} de novo "
                  f"para este sujeito.")

    target_idx = trip[f"{key}__target"]
    ens_pair_a = trip[f"{key}__ens_pair_a"][:, :ensemble_m]
    ens_pair_b = trip[f"{key}__ens_pair_b"][:, :ensemble_m]
    ens_t_frac = trip[f"{key}__ens_t_frac"][:, :ensemble_m]
    ens_valid = trip[f"{key}__ens_valid"][:, :ensemble_m]
    ens_residual_deg = trip[f"{key}__ens_residual_deg"][:, :ensemble_m]
    ens_gap_deg = trip[f"{key}__ens_gap_deg"][:, :ensemble_m]
    ens_pair_a_safe = np.where(ens_pair_a < 0, 0, ens_pair_a)
    ens_pair_b_safe = np.where(ens_pair_b < 0, 0, ens_pair_b)

    bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
    data, affine, header = load_dwi(e.dwi_path)
    shells = split_shells(bvals, tol=args.shell_tol)
    b0_mean = data[..., shells[0]].mean(axis=-1)
    mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)
    shell_key = _resolve_shell_key(shells, args.shell_b, args.shell_tol)
    shell_idxs = np.asarray(shells[shell_key], dtype=int)
    mask_bool = mask.astype(bool)
    shell_vals = data[..., shell_idxs][mask_bool]
    xmax = float(np.percentile(shell_vals, 99)) if shell_vals.size else 1.0
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    signal = data / xmax
    shape3d = data.shape[:3]

    if not (0 <= voxel_global[0] < shape3d[0] and 0 <= voxel_global[1] < shape3d[1]
            and 0 <= voxel_global[2] < shape3d[2]):
        sys.exit(f"--voxel {voxel_global} fora dos limites do volume {shape3d}.")
    if not mask_bool[voxel_global]:
        print(f"[aviso] voxel {voxel_global} esta FORA da mascara de cerebro -- os resultados "
              f"abaixo nao tem significado anatomico (sinal provavelmente ~0/ruido).",
              flush=True)

    origin = centered_patch(voxel_global, args.patch_size, shape3d)
    if origin is None:
        sys.exit(f"--patch-size {args.patch_size} nao cabe no volume {shape3d}, ou voxel fora "
                  f"dos limites -- diminua --patch-size.")
    local_vox = tuple(voxel_global[d] - origin[d] for d in range(3))
    ps = args.patch_size
    sl = (slice(origin[0], origin[0] + ps), slice(origin[1], origin[1] + ps),
          slice(origin[2], origin[2] + ps))
    print(f"Voxel global {voxel_global} -> patch {ps}^3 na origem {origin} "
          f"(voxel local dentro do patch: {local_vox})", flush=True)

    bvecs_t = torch.from_numpy(bvecs.astype(np.float32))
    n_target = target_idx.shape[0]
    bvec_a_all = bvecs_t[ens_pair_a_safe].to(device)
    bvec_b_all = bvecs_t[ens_pair_b_safe].to(device)
    bvec_t_single = bvecs_t[target_idx].to(device)
    bvec_t_all = bvec_t_single.unsqueeze(1).expand(-1, ensemble_m, -1).contiguous()
    t_frac_all = torch.from_numpy(ens_t_frac.astype(np.float32)).to(device)
    ensemble_mask_all = torch.from_numpy(ens_valid.astype(bool)).to(device)
    quality_all = None
    if use_quality_cond:
        quality_np = np.stack([ens_residual_deg / 90.0, ens_gap_deg / 90.0],
                               axis=-1).astype(np.float32)
        quality_all = torch.from_numpy(quality_np).to(device)

    vol_a_patch = signal[sl][..., ens_pair_a_safe]  # (ps,ps,ps,n_target,M)
    vol_b_patch = signal[sl][..., ens_pair_b_safe]
    vol_a_t = torch.from_numpy(
        np.moveaxis(vol_a_patch, (3, 4), (0, 1))[:, :, None].astype(np.float32)).to(device)
    vol_b_t = torch.from_numpy(
        np.moveaxis(vol_b_patch, (3, 4), (0, 1))[:, :, None].astype(np.float32)).to(device)

    with torch.no_grad():
        out, extra = model(vol_a_t, vol_b_t, bvec_a_all, bvec_b_all, bvec_t_all, t_frac_all,
                            ensemble_mask_all, quality=quality_all, return_pairs=True)
    # extra["pred"], extra["pi"]: (n_target, M, 1, ps, ps, ps) -- so nos importa o voxel local
    lx, ly, lz = local_vox
    pred_vox = extra["pred"][:, :, 0, lx, ly, lz].cpu().numpy()   # (n_target, M)
    pi_vox = extra["pi"][:, :, 0, lx, ly, lz].cpu().numpy()       # (n_target, M)
    out_vox = out[:, 0, lx, ly, lz].cpu().numpy()                 # (n_target,)

    rows = []
    for i in range(n_target):
        valid_m = ens_valid[i].astype(bool)
        n_valid = int(valid_m.sum())
        pred_valid = pred_vox[i][valid_m]
        pi_valid = pi_vox[i][valid_m]
        peso_max = float(pi_valid.max()) if n_valid else float("nan")
        discordancia = float(pred_valid.std()) if n_valid > 1 else 0.0
        argmax_m = int(np.argmax(pi_vox[i] * valid_m))
        rows.append({
            "target_idx": int(target_idx[i]),
            "target_bvec": tuple(round(float(v), 3) for v in bvecs[target_idx[i]]),
            "n_pares_validos": n_valid,
            "peso_max": peso_max,
            "discordancia_std": discordancia,
            "par_dominante": argmax_m,
            "saida_final_vox": float(out_vox[i]),
            "pares": [
                {
                    "m": m,
                    "valido": bool(ens_valid[i, m]),
                    "pair_a": int(ens_pair_a[i, m]), "pair_b": int(ens_pair_b[i, m]),
                    "t_frac": float(ens_t_frac[i, m]),
                    "residual_deg": float(ens_residual_deg[i, m]),
                    "gap_deg": float(ens_gap_deg[i, m]),
                    "peso_pi": float(pi_vox[i, m]),
                    "pred_sinal": float(pred_vox[i, m]),
                }
                for m in range(ensemble_m)
            ],
        })

    rows_sorted = sorted(rows, key=lambda r: r["peso_max"], reverse=True)
    display_rows = rows_sorted[: args.top] if args.top else rows_sorted

    print(f"\n{len(rows)} alvos held-out neste voxel. Top "
          f"{'todos' if not args.top else args.top} por concentracao de peso (peso_max):\n",
          flush=True)
    print(f"{'target_idx':>10} {'n_valid':>8} {'peso_max':>9} {'disc_std':>9} "
          f"{'par_dom':>8} {'saida_vox':>10}")
    for r in display_rows:
        print(f"{r['target_idx']:>10} {r['n_pares_validos']:>8} {r['peso_max']:>9.3f} "
              f"{r['discordancia_std']:>9.4f} {r['par_dominante']:>8} {r['saida_final_vox']:>10.4f}")
        for p in r["pares"]:
            marca = " <== dominante" if p["m"] == r["par_dominante"] and p["valido"] else ""
            print(f"    par m={p['m']}: valido={p['valido']} pair_a={p['pair_a']} "
                  f"pair_b={p['pair_b']} t={p['t_frac']:.2f} residual_deg={p['residual_deg']:.1f} "
                  f"gap_deg={p['gap_deg']:.1f} peso={p['peso_pi']:.3f} "
                  f"pred_sinal={p['pred_sinal']:.4f}{marca}")

    n_alerta = sum(1 for r in rows if r["peso_max"] > 0.8 and r["discordancia_std"] > 0.05)
    print(f"\nResumo: {n_alerta}/{len(rows)} alvo(s) com peso concentrado (peso_max>0.8) E "
          f"discordancia alta entre pares (std>0.05) -- evidencia direta de fusao "
          f"'tudo-ou-nada' num par que discorda dos outros nesse voxel, se >0.", flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target_idx", "target_bvec_x", "target_bvec_y", "target_bvec_z",
                        "m", "valido", "pair_a", "pair_b", "t_frac", "residual_deg", "gap_deg",
                        "peso_pi", "pred_sinal", "peso_max_alvo", "discordancia_std_alvo",
                        "par_dominante_alvo", "saida_final_vox_alvo"])
            for r in rows:
                bx, by, bz = r["target_bvec"]
                for p in r["pares"]:
                    w.writerow([r["target_idx"], bx, by, bz, p["m"], p["valido"], p["pair_a"],
                                p["pair_b"], p["t_frac"], p["residual_deg"], p["gap_deg"],
                                p["peso_pi"], p["pred_sinal"], r["peso_max"],
                                r["discordancia_std"], r["par_dominante"], r["saida_final_vox"]])
        print(f"\nTabela completa salva em {out_path}", flush=True)


if __name__ == "__main__":
    main()