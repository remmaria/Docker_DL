#!/usr/bin/env python3
"""
Etapa 5g (baseline "burro", sem rede nenhuma): reconstroi as direcoes-alvo
do esquema de trincas (scripts/02b_build_rrin_triplets.py) usando so o
"blend ingenuo" (1-t_frac)*vol_a + t_frac*vol_b do par (a,b) JA ESCOLHIDO
pelo 02b -- a MESMA formula que scripts/07_visualize_triplet.py ja calcula
pra 1 trinca de exemplo (ver protocolo secao 10.1/11), aqui aplicada a
TODOS os voxels/alvos como um metodo de reconstrucao completo, pra servir
de "piso" de comparacao contra RRIN3D/AMT3D/HFD3D/baseline_sh: se um
metodo aprendido nao bate nem esse blend sem-rede, o aprendizado nao esta
agregando nada alem da propria geometria (a,b,t_frac) que ja alimenta a
rede.

Diferente de scripts/05b_reconstruct_rrin.py, NAO precisa de patches com
overlap/sliding-window nem GPU/torch: e uma combinacao linear ponto-a-ponto
entre dois volumes 3D inteiros (nenhum contexto espacial envolvido), entao
roda inteiro em memoria de uma vez, rapido, so numpy/nibabel.

Grava exatamente a mesma estrutura de saida de 05b_reconstruct_rrin.py
(<out_dir>/<tag>/shell<B>/n<N>/recon_target.nii.gz + target_idx.npy +
mask.npy em shell<B>/) para poder ser usado direto via
--extra-method naive_blend=<out_dir> em 06_evaluate_reconstruction.py e
07_downstream_dti_noddi.py.

Uso:
    python scripts/05g_reconstruct_naive_blend.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --out-dir work_dir/naive_blend_recon \
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--subjects", default=None,
                     help="mesma convencao de --subjects em 05b_reconstruct_rrin.py")
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

    print(f"Reconstruindo (blend ingenuo, sem rede) {len(entries)} sujeito(s): "
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
        target_idx = trip[f"{key}__target"]
        pair_a = trip[f"{key}__pair_a"]
        pair_b = trip[f"{key}__pair_b"]
        t_frac = trip[f"{key}__t_frac"]
        valid = trip[f"{key}__valid"]
        residual_deg = trip[f"{key}__residual_deg"]
        gap_deg = trip[f"{key}__gap_deg"]

        data, affine, header = load_dwi(e.dwi_path)
        # nota: nao precisamos de bvals/bvecs aqui alem de achar o b0 pra
        # mascara -- a,b,t_frac ja foram resolvidos geometricamente pelo
        # 02b e estao gravados no proprio .npz de trincas (indices e
        # fracao, nada de angulo precisa ser recalculado aqui).
        bvals, _bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
        shells = split_shells(bvals, tol=args.shell_tol)
        b0_mean = data[..., shells[0]].mean(axis=-1)
        mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

        # blend ingenuo: (1-t)*vol_a + t*vol_b, ponto-a-ponto, sem overlap
        # nem sliding-window -- e so algebra de arrays inteiros.
        vol_a = data[..., pair_a]                      # (X,Y,Z,n_target)
        vol_b = data[..., pair_b]
        t = t_frac.reshape(1, 1, 1, -1).astype(np.float32)
        pred_dwi = (1.0 - t) * vol_a + t * vol_b
        pred_dwi = pred_dwi.astype(np.float32)
        pred_dwi[~mask] = 0.0

        shell_out = out_dir / tag / f"shell{int(args.shell_b)}"
        sub_out = shell_out / f"n{args.n_level}"
        sub_out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred_dwi, affine), sub_out / "recon_target.nii.gz")
        np.save(sub_out / "target_idx.npy", target_idx)
        # mesmas colunas extras de 05b_reconstruct_rrin.py, mesma ordem de
        # target_idx -- permite estratificar por valid/residual/gap depois.
        np.save(sub_out / "rrin_valid.npy", valid)
        np.save(sub_out / "rrin_residual_deg.npy", residual_deg)
        np.save(sub_out / "rrin_gap_deg.npy", gap_deg)
        mask_path = shell_out / "mask.npy"
        if not mask_path.exists():
            np.save(mask_path, mask)
        n_invalid = int((~valid).sum())
        print(f"{tag}: {target_idx.shape[0]} alvos reconstruidos (blend ingenuo), "
              f"{n_invalid} invalidos ({n_invalid/target_idx.shape[0]:.1%}) -> {sub_out}",
              flush=True)


if __name__ == "__main__":
    main()