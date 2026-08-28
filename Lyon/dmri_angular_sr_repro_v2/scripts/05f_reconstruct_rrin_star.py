#!/usr/bin/env python3
"""
Etapa 5f: aplica uma RRIN3DStar treinada (checkpoint da etapa 4e, ver
scripts/04e_train_rrin_star.py e protocolo secao 14.5 item 1) aos sujeitos
de um split (tipicamente "test"), reconstruindo as direcoes-alvo do esquema
de trincas volume inteiro (blocos/patches com overlap, costurados por media
-- mesmo esquema de scripts/05b_reconstruct_rrin.py).

Diferenca central em relacao a scripts/05b_reconstruct_rrin.py: em vez de UM
par de entrada por alvo, le o feixe de "ensemble_m" pares (`{key}__ens_*`,
ver scripts/02b_build_rrin_triplets.py --ensemble-m) e passa TODOS eles pro
modelo de uma vez -- RRIN3DStar funde as predicoes internamente (ver
model/rrin3d_star.py). O M usado aqui e lido do proprio checkpoint
(`ckpt["args"]["ensemble_m"]"`), igual a `use_quality_cond`/`norm_type` ja
sao lidos do checkpoint em 05b_reconstruct_rrin.py -- nao precisa ser
passado de novo na linha de comando.

Reconstroi TODOS os alvos presentes no esquema de trincas para este
(shell,n_level), MESMO os marcados `valid=False` no par-UNICO -- mesma
justificativa de 05b_reconstruct_rrin.py (comparacao justa com
baseline_sh/RCAE/RRIN3D/AMT3D na MESMA cobertura de direcoes held-out, e
estratificacao posterior por 06_evaluate_reconstruction.py).

Uso:
    python scripts/05f_reconstruct_rrin_star.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --checkpoint work_dir/rrin_star_checkpoints/shell1000_n16_star3/best.pt \
        --shell-b 1000 --n-level 16 \
        --out-dir work_dir/rrin_star_recon \
        --split test --patch-size 10 --stride 8

Requer PyTorch + GPU (ou CPU, mais lento). Nao executado neste ambiente.
"""
import argparse
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


def sliding_window_origins(shape, patch_size, stride):
    origins = []
    for dim_size in shape:
        pos = list(range(0, max(1, dim_size - patch_size + 1), stride))
        if not pos or pos[-1] + patch_size < dim_size:
            pos.append(max(0, dim_size - patch_size))
        origins.append(sorted(set(pos)))
    ox, oy, oz = origins
    return [(x, y, z) for x in ox for y in oy for z in oz]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="pasta com os <tag>_rrin_triplets.npz -- PRECISA ter os campos "
                          "'{key}__ens_*' (etapa 2b rodada com --ensemble-m>=M do checkpoint)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--subjects", default=None,
                     help="mesma convencao de --subjects em 05_reconstruct_rcae.py")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import nibabel as nib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    use_quality_cond = ckpt_args.get("use_quality_cond", False)
    norm_type = ckpt_args.get("norm_type", "instance")
    ensemble_m = ckpt_args.get("ensemble_m")
    if ensemble_m is None:
        raise ValueError(
            f"checkpoint {args.checkpoint} nao tem 'ensemble_m' em args -- confira se e "
            f"mesmo um checkpoint da etapa 4e (scripts/04e_train_rrin_star.py), nao de "
            f"scripts/04b_train_rrin.py (RRIN3D/RRIN3DLayered, par unico).")
    model = build_star_model(base_ch=ckpt_args.get("base_ch", 16),
                              max_disp=ckpt_args.get("max_disp", 0.5),
                              use_quality_cond=use_quality_cond,
                              norm_type=norm_type).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint carregado (epoca {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss')}, "
          f"ensemble_m={ensemble_m}, norm_type={norm_type})")

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]

    def _tag_of(e):
        return e.subject if not e.session else f"{e.subject}_{e.session}"

    if args.subjects:
        wanted = {t.strip() for t in args.subjects.split(",") if t.strip()}
        entries = [e for e in entries if _tag_of(e) in wanted]
        found = {_tag_of(e) for e in entries}
        missing = wanted - found
        if missing:
            print(f"[aviso] --subjects pediu {sorted(missing)}, mas nao encontrei no split "
                  f"{args.split!r} do manifesto.", flush=True)
        if not entries:
            sys.exit(f"Nenhum dos sujeitos pedidos em --subjects foi encontrado no split "
                      f"{args.split!r} -- nada a fazer.")
    if args.limit is not None:
        entries = entries[: args.limit]

    print(f"Reconstruindo {len(entries)} sujeito(s) (RRIN3DStar, M={ensemble_m}): "
          f"{[_tag_of(e) for e in entries]}", flush=True)

    triplets_dir = Path(args.triplets_dir)
    out_dir = Path(args.out_dir)
    key = f"{args.shell_b}__{args.n_level}"

    for e in entries:
        tag = _tag_of(e)
        trip_path = triplets_dir / f"{tag}_rrin_triplets.npz"
        if not trip_path.exists():
            print(f"[aviso] {tag}: sem {trip_path.name}, pulando")
            continue
        trip = np.load(trip_path)
        if f"{key}__target" not in trip.files:
            print(f"[aviso] {tag}: sem trincas para shell={args.shell_b} n={args.n_level}")
            continue
        if f"{key}__ens_pair_a" not in trip.files:
            print(f"[aviso] {tag}: sem campos '{key}__ens_pair_a' (rode "
                  f"scripts/02b_build_rrin_triplets.py --ensemble-m {ensemble_m} de novo "
                  f"para este sujeito), pulando")
            continue
        target_idx = trip[f"{key}__target"]
        valid = trip[f"{key}__valid"]
        residual_deg = trip[f"{key}__residual_deg"]
        gap_deg = trip[f"{key}__gap_deg"]
        ens_pair_a = trip[f"{key}__ens_pair_a"][:, :ensemble_m]   # (n_target, M), -1 = padding
        ens_pair_b = trip[f"{key}__ens_pair_b"][:, :ensemble_m]
        ens_t_frac = trip[f"{key}__ens_t_frac"][:, :ensemble_m]
        ens_valid = trip[f"{key}__ens_valid"][:, :ensemble_m]     # (n_target, M) bool
        ens_residual_deg = trip[f"{key}__ens_residual_deg"][:, :ensemble_m]
        ens_gap_deg = trip[f"{key}__ens_gap_deg"][:, :ensemble_m]
        # indices de padding (-1) nao podem indexar o array de sinal -- troca
        # por 0 (qualquer indice valido serve, a posicao e mascarada mesmo)
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
        n_target = target_idx.shape[0]
        pred_accum = np.zeros(shape3d + (n_target,), dtype=np.float32)
        weight_accum = np.zeros(shape3d, dtype=np.float32)

        bvecs_t = torch.from_numpy(bvecs.astype(np.float32))
        # tensores CONSTANTES ao longo dos patches, calculados uma vez fora
        # do loop espacial -- shape (n_target, M, ...), M = ensemble_m.
        bvec_a_all = bvecs_t[ens_pair_a_safe].to(device)         # (n_target, M, 3)
        bvec_b_all = bvecs_t[ens_pair_b_safe].to(device)
        bvec_t_single = bvecs_t[target_idx].to(device)           # (n_target, 3)
        bvec_t_all = bvec_t_single.unsqueeze(1).expand(-1, ensemble_m, -1).contiguous()
        t_frac_all = torch.from_numpy(ens_t_frac.astype(np.float32)).to(device)  # (n_target, M)
        ensemble_mask_all = torch.from_numpy(ens_valid.astype(bool)).to(device)  # (n_target, M)
        quality_all = None
        if use_quality_cond:
            quality_np = np.stack([ens_residual_deg / 90.0, ens_gap_deg / 90.0],
                                   axis=-1).astype(np.float32)   # (n_target, M, 2)
            quality_all = torch.from_numpy(quality_np).to(device)

        ps = args.patch_size
        origins = sliding_window_origins(shape3d, ps, args.stride)
        with torch.no_grad():
            for (ox, oy, oz) in origins:
                sl = (slice(ox, ox + ps), slice(oy, oy + ps), slice(oz, oz + ps))
                if not mask[sl].any():
                    continue
                # (ps,ps,ps,n_target,M) -- indexa o patch pelos indices de par
                # de TODO o feixe de uma vez (numpy fancy indexing broadcast)
                vol_a_patch = signal[sl][..., ens_pair_a_safe]  # (ps,ps,ps,n_target,M)
                vol_b_patch = signal[sl][..., ens_pair_b_safe]
                # -> (n_target, M, 1, ps, ps, ps)
                vol_a_t = torch.from_numpy(
                    np.moveaxis(vol_a_patch, (3, 4), (0, 1))[:, :, None].astype(np.float32)
                ).to(device)
                vol_b_t = torch.from_numpy(
                    np.moveaxis(vol_b_patch, (3, 4), (0, 1))[:, :, None].astype(np.float32)
                ).to(device)

                pred = model(vol_a_t, vol_b_t, bvec_a_all, bvec_b_all, bvec_t_all, t_frac_all,
                             ensemble_mask_all, quality=quality_all)
                # pred: (n_target, 1, ps, ps, ps) -> (ps,ps,ps,n_target)
                pred_np = pred[:, 0].permute(1, 2, 3, 0).cpu().numpy()

                pred_accum[sl] += pred_np
                weight_accum[sl] += 1.0

        weight_safe = np.where(weight_accum > 0, weight_accum, 1.0)
        pred_signal = pred_accum / weight_safe[..., None]
        pred_dwi = pred_signal * xmax
        pred_dwi[~mask] = 0.0

        shell_out = out_dir / tag / f"shell{int(args.shell_b)}"
        sub_out = shell_out / f"n{args.n_level}"
        sub_out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred_dwi.astype(np.float32), affine), sub_out / "recon_target.nii.gz")
        np.save(sub_out / "target_idx.npy", target_idx)
        np.save(sub_out / "rrin_valid.npy", valid)
        np.save(sub_out / "rrin_residual_deg.npy", residual_deg)
        np.save(sub_out / "rrin_gap_deg.npy", gap_deg)
        # coluna extra propria do ensemble -- quantos pares REAIS o feixe
        # tinha por alvo (util pra checar se M=ensemble_m era mesmo
        # atingido na maioria dos alvos, ou se o dataset frequentemente cai
        # no fallback de 1 posicao so, ver find_star_ensemble_batch).
        np.save(sub_out / "rrin_star_n_pares.npy", ens_valid.sum(axis=1))
        mask_path = shell_out / "mask.npy"
        if not mask_path.exists():
            np.save(mask_path, mask)
        n_invalid = int((~valid).sum())
        print(f"{tag}: reconstrucao RRIN3DStar (M={ensemble_m}) salva em {sub_out} "
              f"({n_invalid}/{n_target} alvos com par-unico invalido; media de "
              f"{ens_valid.sum(axis=1).mean():.2f}/{ensemble_m} pares reais por alvo)")


if __name__ == "__main__":
    main()