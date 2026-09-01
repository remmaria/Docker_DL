#!/usr/bin/env python3
"""
Etapa 5h (baseline "burro" com o MESMO pool geometrico do ensemble em
estrela, mas SEM rede nenhuma): reconstroi as direcoes-alvo fazendo a
MEDIA SIMPLES (nao aprendida) dos M "blends ingenuos"
(1-t_frac_m)*vol_a_m + t_frac_m*vol_b_m calculados sobre CADA par do feixe
`{key}__ens_*` (scripts/02b_build_rrin_triplets.py --ensemble-m), em vez
de usar so o par unico canonico (isso e' o que scripts/05g_reconstruct_naive_blend.py
ja faz).

MOTIVACAO (pergunta da usuaria, 2026-08-31): comparando `naive_blend`
(par unico, sem pool nem rede) contra `rrin_n16_star610` (RRIN3DStar, pool
rico + fusao APRENDIDA por voxel via PairWeightHead3D, ver
model/rrin3d_star.py e addendum secao 13), nao da pra saber se o ganho do
`star610` vem (a) so de ter um pool de pares geometricamente mais diverso
disponivel (mesmo sem nenhuma rede pra escolher entre eles -- um efeito
tipo "bagging", reduzindo variancia so por ter mais de uma estimativa pra
mediar), (b) so da rede aprender a fundir/pesar esses pares por confianca
por voxel, ou (c) uma combinacao dos dois. Este script isola o fator (a):
usa o MESMO pool geometrico (mesmo --triplets-dir, mesmo teto de residuo
usado pra gerar os campos ens_* -- tipicamente ENSEMBLE_MAX_RESIDUAL_DEG=10
pra bater com o `star610`), mas funde os M blends com uma media UNIFORME
(nao aprendida) sobre os slots validos, em vez do softmax por voxel que a
RRIN3DStar aprende.

Com os tres metodos lado a lado no mesmo --extra-method de
06_evaluate_reconstruction.py/07_downstream_dti_noddi.py:
  naive_blend            (par unico, sem pool, sem rede)              -- piso
  naive_ensemble_blend   (pool rico, media uniforme, sem rede)        -- este script
  rrin_n16_star610       (pool rico, fusao aprendida por voxel)       -- teto atual
a diferenca (naive_ensemble_blend - naive_blend) isola o efeito do POOL
sozinho (mais candidatos geometricamente diversos pra mediar, mesmo sem
nenhum aprendizado); a diferenca (rrin_n16_star610 - naive_ensemble_blend)
isola o efeito da REDE (FlowNet3D fazendo warp de verdade em vez de blend
linear, + a fusao aprendida escolhendo/pesando os pares por confianca em
vez de media uniforme).

Slots de padding (`ens_valid=False`, quando o pool geometrico tem menos de
M candidatos pra um alvo -- ver addendum secao 14.1/15) sao EXCLUIDOS da
media (media so sobre os slots com `ens_valid=True`); se NENHUM slot for
valido para um alvo, cai pro slot 0 (mesmo fallback de
find_star_ensemble_batch/RRIN3DStar: "melhor candidato disponivel, mesmo
fora do teto", ver utils/gradients.py:find_star_ensemble_batch).

Nao precisa de GPU/torch (so numpy/nibabel, mesmo espirito de
05g_reconstruct_naive_blend.py) -- e so algebra de arrays inteiros, sem
patches/sliding-window.

Grava a mesma estrutura de saida de 05g/05b (<out_dir>/<tag>/shell<B>/n<N>/
recon_target.nii.gz + target_idx.npy + mask.npy em shell<B>/), pra poder
ser usado direto via --extra-method naive_ensemble_blend=<out_dir> em
06_evaluate_reconstruction.py e 07_downstream_dti_noddi.py.

Uso:
    python scripts/05h_reconstruct_naive_ensemble_blend.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --out-dir work_dir/naive_ensemble_blend_recon \
        --split test
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_dwi, split_shells, load_bval_bvec
from utils.masking import load_or_build_mask


def naive_ensemble_blend(vol_a_ens, vol_b_ens, t_frac_ens, ens_valid):
    """Media uniforme (nao aprendida) dos M blends ingenuos por alvo.

    vol_a_ens/vol_b_ens: (X,Y,Z,n_target,M) -- ja indexados pelos pares do
        feixe (fancy-indexing feito pelo chamador, mesmo padrao numerico
        ja verificado isoladamente pra 05f_reconstruct_rrin_star.py).
    t_frac_ens: (n_target,M)
    ens_valid: (n_target,M) bool -- False nos slots de padding (pool do
        alvo tinha menos de M candidatos, ver addendum secao 14.1/15).

    Retorna (X,Y,Z,n_target): media dos blends validos por alvo; se NENHUM
    slot for valido para um alvo, usa so o slot 0 (fallback, mesma
    semantica de find_star_ensemble_batch).
    """
    t = t_frac_ens.reshape(1, 1, 1, *t_frac_ens.shape).astype(np.float32)
    blend_m = (1.0 - t) * vol_a_ens + t * vol_b_ens  # (X,Y,Z,n_target,M)

    valid_f = ens_valid.astype(np.float32)  # (n_target,M)
    n_valid = valid_f.sum(axis=-1)  # (n_target,)
    fallback = n_valid == 0
    if fallback.any():
        # nenhum candidato no pool passou no teto para este alvo -- cai pro
        # slot 0 sozinho (mesmo fallback de find_star_ensemble_batch).
        valid_f = valid_f.copy()
        valid_f[fallback, 0] = 1.0
        n_valid = valid_f.sum(axis=-1)

    weights = (valid_f / n_valid[:, None]).reshape(1, 1, 1, *valid_f.shape)
    return (blend_m * weights).sum(axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="pasta com os <tag>_rrin_triplets.npz -- PRECISA ter os campos "
                          "'{key}__ens_*' (etapa 2b rodada com --ensemble-m>=1; use o MESMO "
                          "--triplets-dir/--ensemble-max-residual-deg usado pra gerar o "
                          "checkpoint do RRIN3DStar que voce quer comparar, pra isolar so o "
                          "efeito rede vs. media uniforme sobre o MESMO pool geometrico)")
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--subjects", default=None,
                     help="mesma convencao de --subjects em 05g_reconstruct_naive_blend.py")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import nibabel as nib

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

    print(f"Reconstruindo (blend ingenuo do FEIXE, media uniforme, sem rede) "
          f"{len(entries)} sujeito(s): {[_tag_of(e) for e in entries]}", flush=True)

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
            sys.exit(
                f"{tag}: {trip_path.name} nao tem os campos '{key}__ens_pair_a' etc. -- "
                f"rode scripts/02b_build_rrin_triplets.py de novo com --ensemble-m>=1 "
                f"apontando pro MESMO --triplets-dir usado aqui (aditivo, ver addendum "
                f"secao 13/15)."
            )

        target_idx = trip[f"{key}__target"]
        ens_pair_a = trip[f"{key}__ens_pair_a"]      # (n_target, M), -1 = padding
        ens_pair_b = trip[f"{key}__ens_pair_b"]
        ens_t_frac = trip[f"{key}__ens_t_frac"]
        ens_valid = trip[f"{key}__ens_valid"]        # (n_target, M) bool
        # par unico (canonico) so pra manter as mesmas colunas de
        # diagnostico que 05g/05b ja gravam (valid/residual/gap do par
        # UNICO, nao do feixe -- estratificacao posterior continua
        # comparavel entre os tres metodos).
        valid = trip[f"{key}__valid"]
        residual_deg = trip[f"{key}__residual_deg"]
        gap_deg = trip[f"{key}__gap_deg"]

        # indices de padding (-1) sao inofensivos pra indexar (viram o
        # ultimo indice do array bvec) pois sempre entram com peso 0 na
        # media (ens_valid=False nesses slots, exceto no raro fallback
        # tratado dentro de naive_ensemble_blend) -- mesmo cuidado de
        # clipping ja usado em 05f_reconstruct_rrin_star.py.
        n_bvecs = None  # resolvido apos carregar o DWI, abaixo

        data, affine, header = load_dwi(e.dwi_path)
        n_bvecs = data.shape[-1]
        ens_pair_a_clipped = np.clip(ens_pair_a, 0, n_bvecs - 1)
        ens_pair_b_clipped = np.clip(ens_pair_b, 0, n_bvecs - 1)

        bvals, _bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
        shells = split_shells(bvals, tol=args.shell_tol)
        b0_mean = data[..., shells[0]].mean(axis=-1)
        mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

        # fancy-indexing (X,Y,Z,n_target,M), mesmo padrao numerico ja
        # verificado isoladamente pra 05f_reconstruct_rrin_star.py.
        vol_a_ens = data[..., ens_pair_a_clipped]
        vol_b_ens = data[..., ens_pair_b_clipped]
        pred_dwi = naive_ensemble_blend(vol_a_ens, vol_b_ens, ens_t_frac, ens_valid)
        pred_dwi = pred_dwi.astype(np.float32)
        pred_dwi[~mask] = 0.0

        shell_out = out_dir / tag / f"shell{int(args.shell_b)}"
        sub_out = shell_out / f"n{args.n_level}"
        sub_out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred_dwi, affine), sub_out / "recon_target.nii.gz")
        np.save(sub_out / "target_idx.npy", target_idx)
        np.save(sub_out / "rrin_valid.npy", valid)
        np.save(sub_out / "rrin_residual_deg.npy", residual_deg)
        np.save(sub_out / "rrin_gap_deg.npy", gap_deg)
        mask_path = shell_out / "mask.npy"
        if not mask_path.exists():
            np.save(mask_path, mask)

        n_pool1 = int((ens_valid.sum(axis=-1) <= 1).sum())
        print(f"{tag}: {target_idx.shape[0]} alvos reconstruidos (blend ingenuo do feixe, "
              f"M={ens_valid.shape[1]}), {n_pool1} com pool<=1 (colapsado ao par unico) "
              f"-> {sub_out}", flush=True)


if __name__ == "__main__":
    main()