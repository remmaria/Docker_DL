#!/usr/bin/env python3
"""
Etapa 8 (opcional, a mais forte para a tese, tambem a mais fragil):
compara tratografia (CSD determinístico via MRtrix3) entre ground truth e
reconstrucoes (baseline SH e/ou RCAE), usando densidade de streamlines
(Dice entre mapas binarizados) como metrica agregada. Para tractometria por
feixe especifico, prefira `scilpy` ou `dipy.segment` num passo posterior --
aqui o objetivo e uma comparacao global rapida de plausibilidade.

Requer MRtrix3 instalado e no PATH (dwi2response, dwi2fod, tckgen, tckmap,
mrconvert). Chama tudo via subprocess; se algum passo falhar, o sujeito e
pulado com aviso (nao derruba o resto do lote).

Uso:
    python scripts/08_downstream_tractography.py \
        --manifest work_dir/manifest.csv \
        --baseline-dir work_dir/baseline_recon \
        --rcae-dir work_dir/rcae_recon \
        --shell-b 1000 --n-level 10 \
        --out-dir work_dir/tractography \
        --n-streamlines 200000
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_dwi, load_bval_bvec
from utils.masking import load_or_build_mask


def run(cmd: list[str], **kwargs):
    print(" $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def build_full_volume(gt_data, recon_dir, tag, shell_b, n_level):
    sub_dir = Path(recon_dir) / tag / f"shell{int(shell_b)}" / f"n{n_level}"
    recon_path = sub_dir / "recon_target.nii.gz"
    target_idx_path = sub_dir / "target_idx.npy"
    if not recon_path.exists() or not target_idx_path.exists():
        return None
    import nibabel as nib
    recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
    target_idx = np.load(target_idx_path)
    out = gt_data.copy()
    out[..., target_idx] = recon
    return out


def tractography_pipeline(work_dir: Path, data, affine, bvals, bvecs, mask, n_streamlines: int):
    import nibabel as nib
    work_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), work_dir / "dwi.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), work_dir / "mask.nii.gz")
    np.savetxt(work_dir / "dwi.bval", bvals.reshape(1, -1), fmt="%d")
    np.savetxt(work_dir / "dwi.bvec", bvecs.T, fmt="%.6f")

    dwi_mif = work_dir / "dwi.mif"
    run(["mrconvert", str(work_dir / "dwi.nii.gz"), str(dwi_mif),
         "-fslgrad", str(work_dir / "dwi.bvec"), str(work_dir / "dwi.bval"), "-force"])

    response = work_dir / "response.txt"
    run(["dwi2response", "tournier", str(dwi_mif), str(response), "-force"])

    fod = work_dir / "fod.mif"
    run(["dwi2fod", "csd", str(dwi_mif), str(response), str(fod),
         "-mask", str(work_dir / "mask.nii.gz"), "-force"])

    tck = work_dir / "tracks.tck"
    run(["tckgen", str(fod), str(tck), "-seed_image", str(work_dir / "mask.nii.gz"),
         "-mask", str(work_dir / "mask.nii.gz"), "-select", str(n_streamlines),
         "-force"])

    density = work_dir / "density.nii.gz"
    run(["tckmap", str(tck), str(density), "-template", str(work_dir / "mask.nii.gz"),
         "-force"])

    return nib.load(str(density)).get_fdata()


def dice_from_density(density_a, density_b, percentile: float = 50.0):
    """Binariza os mapas de densidade de streamlines por percentil (dentro
    dos voxels com streamline > 0) e calcula o coeficiente de Dice.
    """
    def binarize(d):
        nz = d[d > 0]
        if nz.size == 0:
            return np.zeros_like(d, dtype=bool)
        thr = np.percentile(nz, percentile)
        return d >= thr

    a = binarize(density_a)
    b = binarize(density_b)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2 * inter / denom)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--rcae-dir", default=None)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-streamlines", type=int, default=200_000)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    args = ap.parse_args()

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]
    out_dir = Path(args.out_dir)
    rows = []

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
        gt_data, affine, header = load_dwi(e.dwi_path)
        b0_mean = gt_data[..., bvals < 50].mean(axis=-1)
        mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

        variants = {"ground_truth": gt_data}
        for method, recon_dir in (("baseline_sh", args.baseline_dir), ("rcae", args.rcae_dir)):
            if recon_dir is None:
                continue
            full = build_full_volume(gt_data, recon_dir, tag, args.shell_b, args.n_level)
            if full is not None:
                variants[method] = full

        densities = {}
        for method, vol in variants.items():
            try:
                wd = out_dir / "mrtrix_tmp" / tag / method
                densities[method] = tractography_pipeline(wd, vol, affine, bvals, bvecs, mask,
                                                             args.n_streamlines)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"[aviso] tratografia falhou para {tag}/{method}: {exc}. "
                      f"Confira se MRtrix3 esta instalado e no PATH.")

        if "ground_truth" not in densities:
            continue
        for method in variants:
            if method == "ground_truth" or method not in densities:
                continue
            dice = dice_from_density(densities[method], densities["ground_truth"])
            rows.append({"subject": e.subject, "method": method, "shell": args.shell_b,
                         "n_level": args.n_level, "dice_streamline_density": dice})
        print(f"{tag}: tratografia comparada para {[m for m in variants if m in densities]}")

    if not rows:
        sys.exit("Nenhum resultado de tratografia (confira instalacao do MRtrix3 e os diretorios)")

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"tractography_metrics_shell{int(args.shell_b)}_n{args.n_level}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Metricas de tratografia salvas em", out_csv)
    print(df.groupby("method")["dice_streamline_density"].mean())


if __name__ == "__main__":
    main()
