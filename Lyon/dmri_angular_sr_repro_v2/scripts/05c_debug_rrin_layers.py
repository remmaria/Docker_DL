#!/usr/bin/env python3
"""
Etapa 5c (diagnostico -- NAO faz parte do pipeline principal de avaliacao,
ver scripts/05b_reconstruct_rrin.py e scripts/06_evaluate_reconstruction.py
para isso): inspeciona os pesos de camada (pi^(k)) de um checkpoint
RRIN3DLayered (K>=2, ver model/rrin3d.py e protocolo secao 13, "Toward a
layered-flow extension for crossing fibers") pra checar se a rede aprendeu
a usar mais de uma camada em algum lugar -- e, se sim, onde isso acontece
no volume.

IMPORTANTE sobre identidade de camada (leia antes de interpretar os mapas):
o indice k de uma camada NAO tem significado consistente entre alvos
(triplets) diferentes -- e uma mistura sem ordem imposta, nao ha nada que
force "camada 1" a sempre representar a mesma populacao de fibra de um
alvo pro outro (permutacao livre). Por isso este script trata os pesos
pi^(k) de duas formas diferentes:

  1. Para UM alvo especifico (--targets), os mapas pi^(k) SAO consistentes
     no volume inteiro -- a mesma triplet (a,b,t) e os mesmos pesos da rede
     sao aplicados a cada patch da janela deslizante, entao a "camada 1"
     desse alvo significa a mesma coisa em todo o cerebro. Da pra
     visualizar as K camadas desse alvo lado a lado (arquivo
     `pi_target<idx>.nii.gz`, 4D com K volumes) e procurar estrutura
     espacial reconhecivel (ex.: uma camada dominante numa regiao coerente
     tipo corpo caloso, outra assumindo mais peso em outra regiao tipo
     centrum semiovale).

  2. Para AGREGAR atraves de VARIOS alvos (e portanto varias triplets
     diferentes, cada uma com sua propria "ordem" arbitraria de camadas),
     so usamos uma quantidade INVARIANTE A PERMUTACAO: o numero EFETIVO de
     camadas usadas por voxel, definido como a perplexidade da distribuicao
     pi (exponencial da entropia de Shannon):

         n_eff(voxel) = exp( -sum_k pi_k(voxel) * log(pi_k(voxel)) )

     que vai de 1 (rede colapsou pra uma camada so naquele voxel -- pi
     concentrado, sem mistura) ate K (pi uniforme entre as K camadas --
     mistura maxima). Essa metrica NAO depende de qual k especifico "ganhou"
     em cada voxel, entao pode ser comparada e MEDIA com seguranca entre
     alvos diferentes (`efflayers_target<idx>.nii.gz` por alvo,
     `efflayers_mean_alltargets.nii.gz` agregado) -- e e a resposta mais
     direta pra "esta aprendendo alguma coisa tipo crossing em algum
     lugar, ou colapsou tudo pra K=1 efetivo em todo canto?".

Uso:
    python scripts/05c_debug_rrin_layers.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --checkpoint work_dir/rrin_checkpoints/shell1000_n10_k3/best.pt \
        --shell-b 1000 --n-level 10 \
        --subject <tag> \
        --out-dir work_dir/rrin_layer_debug \
        [--targets 0,10,20]            # default: ate 6 alvos espacados
        [--all-targets]                # sobrescreve --targets, roda TODOS
        [--gfa-path <alguma>.nii.gz]   # opcional, ver secao "correlacao com GFA"

Correlacao com GFA (opcional, --gfa-path): se voce ja tem um mapa de GFA
(ou qualquer outro indicador grosseiro de complexidade de fibra) calculado
da aquisicao COMPLETA do sujeito (NAO do n_level subamostrado -- mesmo
cuidado do protocolo secao 13 sobre nao circularizar a evidencia), o script
imprime a correlacao de Pearson dentro da mascara entre esse mapa e
`efflayers_mean_alltargets`. Uma correlacao positiva e clara (regioes de
GFA baixo/complexo com n_eff mais alto) e uma boa evidencia de que a rede
esta de fato alocando camadas de acordo com a complexidade de fibra local,
sem que isso tenha sido ensinado explicitamente (nenhuma supervisao de
GFA/CSD foi usada no treino -- ver protocolo, plano recomendado e2 comecar
sem essa supervisao).

Requer PyTorch + GPU (ou CPU, mais lento) + nibabel. Nao executado neste
ambiente de desenvolvimento -- revisado manualmente, mesmo padrao dos
outros scripts da etapa 5.
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
from model.rrin3d import build_rrin_model, RRIN3DLayered


def sliding_window_origins(shape, patch_size, stride):
    """Identico a scripts/05b_reconstruct_rrin.py -- duplicado de proposito
    (convencao do repo: scripts numerados sao autocontidos, ver tambem
    scripts/05_reconstruct_rcae.py)."""
    origins = []
    for dim_size in shape:
        pos = list(range(0, max(1, dim_size - patch_size + 1), stride))
        if not pos or pos[-1] + patch_size < dim_size:
            pos.append(max(0, dim_size - patch_size))
        origins.append(sorted(set(pos)))
    ox, oy, oz = origins
    return [(x, y, z) for x in ox for y in oy for z in oz]


def effective_num_layers(pi):
    """pi: (..., K) pesos de camada (somam 1 no ultimo eixo). Retorna
    exp(entropia de Shannon) no ultimo eixo -- ver docstring do modulo.
    Trata pi_k=0 com a convencao usual 0*log(0)=0 (nan_to_num)."""
    log_pi = np.log(np.clip(pi, 1e-12, 1.0))
    entropy = -np.sum(np.where(pi > 0, pi * log_pi, 0.0), axis=-1)
    return np.exp(entropy)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--subject", required=True,
                     help="tag do sujeito (mesma convencao de --subjects em 05b, "
                          "mas aqui e SO UM sujeito -- este e um script de diagnostico "
                          "qualitativo, nao de avaliacao em lote)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--targets", default=None,
                     help="indices de alvo (coluna target_idx do esquema de trincas) pra "
                          "salvar os mapas pi^(k) individuais, separados por virgula (ex. "
                          "'0,10,20'). Default: ate 6 alvos espacados uniformemente. Note que "
                          "sao indices DENTRO do array de alvos held-out (0..n_target-1), nao "
                          "indices de direcao bruta do esquema de aquisicao.")
    ap.add_argument("--all-targets", action="store_true",
                     help="processa TODOS os alvos (sobrescreve --targets) -- grava um "
                          "pi_target<idx>.nii.gz por alvo, pode gerar bastante arquivo "
                          "(n_target * K volumes 3D no total). Use com moderacao.")
    ap.add_argument("--gfa-path", default=None,
                     help="opcional: caminho de um mapa de GFA (ou outro indicador de "
                          "complexidade de fibra) calculado da aquisicao COMPLETA do sujeito, "
                          "no mesmo espaco/affine do DWI. Se passado, imprime a correlacao de "
                          "Pearson (dentro da mascara) com efflayers_mean_alltargets.")
    args = ap.parse_args()

    import nibabel as nib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    use_quality_cond = ckpt_args.get("use_quality_cond", False)
    num_layers = ckpt_args.get("num_layers", 1)
    if num_layers < 2:
        sys.exit(f"Este checkpoint tem num_layers={num_layers} (K=1, arquitetura RRIN3D "
                  f"original) -- este script de diagnostico so faz sentido para checkpoints "
                  f"RRIN3DLayered (K>=2, treinados com --num-layers/NUM_LAYERS>=2). Nao ha "
                  f"pesos de camada pi^(k) pra inspecionar num modelo K=1.")
    norm_type = ckpt_args.get("norm_type", "instance")
    model = build_rrin_model(num_layers=num_layers,
                              base_ch=ckpt_args.get("base_ch", 16),
                              max_disp=ckpt_args.get("max_disp", 0.5),
                              use_quality_cond=use_quality_cond,
                              norm_type=norm_type).to(device)
    assert isinstance(model, RRIN3DLayered)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint carregado (epoca {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss')}, "
          f"num_layers={num_layers}, norm_type={norm_type})")

    entries = load_manifest(args.manifest)

    def _tag_of(e):
        return e.subject if not e.session else f"{e.subject}_{e.session}"

    matches = [e for e in entries if _tag_of(e) == args.subject]
    if not matches:
        sys.exit(f"Sujeito {args.subject!r} nao encontrado no manifesto {args.manifest}")
    e = matches[0]
    tag = args.subject

    triplets_dir = Path(args.triplets_dir)
    trip_path = triplets_dir / f"{tag}_rrin_triplets.npz"
    if not trip_path.exists():
        sys.exit(f"{tag}: sem {trip_path}")
    trip = np.load(trip_path)
    key = f"{args.shell_b}__{args.n_level}"
    if f"{key}__target" not in trip.files:
        sys.exit(f"{tag}: sem trincas para shell={args.shell_b} n={args.n_level}")
    target_idx = trip[f"{key}__target"]
    pair_a = trip[f"{key}__pair_a"]
    pair_b = trip[f"{key}__pair_b"]
    t_frac = trip[f"{key}__t_frac"]
    valid = trip[f"{key}__valid"]
    residual_deg = trip[f"{key}__residual_deg"]
    gap_deg = trip[f"{key}__gap_deg"]
    n_target = target_idx.shape[0]

    if args.all_targets:
        chosen_targets = list(range(n_target))
    elif args.targets:
        chosen_targets = [int(x) for x in args.targets.split(",") if x.strip()]
        bad = [t for t in chosen_targets if t < 0 or t >= n_target]
        if bad:
            sys.exit(f"--targets tem indice(s) fora de [0,{n_target-1}]: {bad}")
    else:
        n_pick = min(6, n_target)
        chosen_targets = sorted(set(np.linspace(0, n_target - 1, n_pick).round().astype(int)))
    print(f"{tag}: {n_target} alvos no total, salvando mapas pi^(k) individuais para "
          f"{len(chosen_targets)} alvo(s): {chosen_targets}")

    bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
    data, affine, header = load_dwi(e.dwi_path)
    shells = split_shells(bvals, tol=args.shell_tol)
    b0_mean = data[..., shells[0]].mean(axis=-1)
    mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)
    mask_bool = mask.astype(bool)

    shell_key = _resolve_shell_key(shells, args.shell_b, args.shell_tol)
    shell_idxs = np.asarray(shells[shell_key], dtype=int)
    shell_vals = data[..., shell_idxs][mask_bool]
    xmax = float(np.percentile(shell_vals, 99)) if shell_vals.size else 1.0
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    signal = data / xmax

    shape3d = data.shape[:3]
    bvecs_t = torch.from_numpy(bvecs.astype(np.float32))
    bvec_a_all = bvecs_t[pair_a].to(device)
    bvec_b_all = bvecs_t[pair_b].to(device)
    bvec_t_all = bvecs_t[target_idx].to(device)
    t_frac_all = torch.from_numpy(t_frac.astype(np.float32)).to(device)
    quality_all = None
    if use_quality_cond:
        quality_np = np.stack([residual_deg / 90.0, gap_deg / 90.0], axis=1).astype(np.float32)
        quality_all = torch.from_numpy(quality_np).to(device)

    ps = args.patch_size
    origins = sliding_window_origins(shape3d, ps, args.stride)

    # pi_accum: (X,Y,Z,n_target,K) -- so acumulamos pi (nao precisamos do
    # volume reconstruido em si aqui, isso ja e feito por 05b). Para
    # sujeitos grandes isso pode pesar em memoria (n_target*K volumes
    # float32 do tamanho do cerebro); K e n_target sao tipicamente pequenos
    # o bastante (K<=4, n_target<=54) pra caber sem problema.
    K = num_layers
    pi_accum = np.zeros(shape3d + (n_target, K), dtype=np.float32)
    weight_accum = np.zeros(shape3d, dtype=np.float32)

    with torch.no_grad():
        for (ox, oy, oz) in origins:
            sl = (slice(ox, ox + ps), slice(oy, oy + ps), slice(oz, oz + ps))
            if not mask[sl].any():
                continue
            vol_a_patch = signal[sl][..., pair_a]
            vol_b_patch = signal[sl][..., pair_b]
            vol_a_t = torch.from_numpy(np.moveaxis(vol_a_patch, -1, 0)[:, None]
                                        .astype(np.float32)).to(device)
            vol_b_t = torch.from_numpy(np.moveaxis(vol_b_patch, -1, 0)[:, None]
                                        .astype(np.float32)).to(device)

            _, layers = model(vol_a_t, vol_b_t, bvec_a_all, bvec_b_all, bvec_t_all, t_frac_all,
                               quality=quality_all, return_layers=True)
            # layers["pi"]: (n_target, K, ps, ps, ps) -> (ps,ps,ps,n_target,K)
            pi_np = layers["pi"].permute(2, 3, 4, 0, 1).cpu().numpy()

            pi_accum[sl] += pi_np
            weight_accum[sl] += 1.0

    weight_safe = np.where(weight_accum > 0, weight_accum, 1.0)
    pi_mean = pi_accum / weight_safe[..., None, None]  # (X,Y,Z,n_target,K)
    pi_mean[~mask_bool] = 0.0

    out_dir = Path(args.out_dir) / tag / f"shell{int(args.shell_b)}_n{args.n_level}_k{K}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (1) mapas pi^(k) por alvo escolhido (identidade de camada valida
    # dentro de um mesmo alvo, ver docstring do modulo) ---
    for ti in chosen_targets:
        pi_this = pi_mean[..., ti, :]  # (X,Y,Z,K)
        nib.save(nib.Nifti1Image(pi_this.astype(np.float32), affine),
                 out_dir / f"pi_target{ti:03d}.nii.gz")

    # --- (2) numero efetivo de camadas, por alvo escolhido e agregado ---
    efflayers_all = effective_num_layers(pi_mean)  # (X,Y,Z,n_target)
    efflayers_all[~mask_bool] = 0.0
    for ti in chosen_targets:
        nib.save(nib.Nifti1Image(efflayers_all[..., ti].astype(np.float32), affine),
                 out_dir / f"efflayers_target{ti:03d}.nii.gz")

    efflayers_mean = efflayers_all.mean(axis=-1)  # (X,Y,Z) -- media segura entre alvos
    efflayers_mean[~mask_bool] = 0.0
    nib.save(nib.Nifti1Image(efflayers_mean.astype(np.float32), affine),
             out_dir / "efflayers_mean_alltargets.nii.gz")

    # --- (3) resumo numerico, sem precisar abrir nenhum visualizador ---
    vals_in_mask = efflayers_mean[mask_bool]
    pcts = np.percentile(vals_in_mask, [5, 25, 50, 75, 95]) if vals_in_mask.size else \
        np.full(5, np.nan)
    frac_using_gt1 = float((vals_in_mask > 1.05).mean()) if vals_in_mask.size else float("nan")
    print(f"\n[resumo] numero efetivo de camadas (K={K}), agregado por todos os "
          f"{n_target} alvos, dentro da mascara:")
    print(f"  percentis [5,25,50,75,95] = {np.round(pcts, 3).tolist()}")
    print(f"  fracao de voxels com n_eff > 1.05 (evidencia de mistura real, nao so "
          f"ruido numerico perto de 1.0): {frac_using_gt1:.1%}")
    if frac_using_gt1 < 0.01:
        print("  [aviso] fracao muito baixa -- forte indicio de COLAPSO DE MODO (a rede "
              "praticamente sempre usa 1 camada so, em todo o cerebro). Ver protocolo secao "
              "13: proximo passo seria a supervisao auxiliar via CSD/peak-count.")
    else:
        print("  Ha evidencia de mistura de camadas em pelo menos parte do volume -- vale a "
              "pena abrir efflayers_mean_alltargets.nii.gz num visualizador (ex. FSLeyes) e "
              "comparar com a anatomia conhecida (crossing esperado em centrum semiovale, "
              "corona radiata; NAO esperado no meio do corpo caloso ou do trato "
              "corticoespinhal isolado).")

    print(f"\nArquivos salvos em: {out_dir}")
    print(f"  pi_target<idx>.nii.gz          -- pesos de camada (4D, K volumes) por alvo")
    print(f"  efflayers_target<idx>.nii.gz   -- numero efetivo de camadas (3D) por alvo")
    print(f"  efflayers_mean_alltargets.nii.gz -- media entre todos os {n_target} alvos "
          f"(a metrica agregada, permutation-invariant -- comece por este arquivo)")

    # --- (4) correlacao opcional com um mapa de GFA (ou similar) externo ---
    if args.gfa_path:
        gfa_img = nib.load(args.gfa_path)
        gfa = np.asarray(gfa_img.dataobj, dtype=np.float32)
        if gfa.shape != shape3d:
            print(f"[aviso] --gfa-path tem shape {gfa.shape}, esperado {shape3d} -- pulando "
                  f"correlacao (confira se esta no mesmo espaco/affine do DWI deste sujeito).")
        else:
            gfa_vals = gfa[mask_bool]
            eff_vals = efflayers_mean[mask_bool]
            finite = np.isfinite(gfa_vals) & np.isfinite(eff_vals)
            if finite.sum() > 10:
                r = float(np.corrcoef(gfa_vals[finite], eff_vals[finite])[0, 1])
                print(f"\n[gfa] correlacao de Pearson (dentro da mascara, n={finite.sum()}) "
                      f"entre --gfa-path e efflayers_mean_alltargets: r={r:.3f}")
                print(f"  (GFA calculado da aquisicao COMPLETA, nao do n_level subamostrado -- "
                      f"ver protocolo secao 13 sobre nao circularizar a evidencia. Uma "
                      f"correlacao positiva e clara e evidencia de que a rede aloca camadas de "
                      f"acordo com complexidade de fibra local, SEM ter sido supervisionada "
                      f"com isso.)")
            else:
                print("[aviso] poucos voxels finitos em comum entre GFA e mascara -- pulando "
                      "correlacao.")


if __name__ == "__main__":
    main()