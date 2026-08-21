#!/usr/bin/env python3
"""
Etapa 5: aplica um RCAE treinado (checkpoint da etapa 4) aos sujeitos de um
split (tipicamente "test"), reconstruindo as direcoes held-out volume
inteiro (processamento em blocos/patches com overlap, costurados por media).

Uso:
    python scripts/05_reconstruct_rcae.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --checkpoint work_dir/rcae_checkpoints/shell1000_n10/best.pt \
        --shell-b 1000 --n-level 10 \
        --out-dir work_dir/rcae_recon \
        --split test --patch-size 24 --stride 16

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
from model.rcae import RCAE


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
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--patch-size", type=int, default=24)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    args = ap.parse_args()

    import nibabel as nib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    # ATENCAO: checkpoints de ANTES da reproducao fiel da arquitetura do
    # paper (ConvGRU customizada, base_ch/encoder_out_ch/gru_hidden_ch)
    # NAO sao compativeis com o RCAE atual -- state_dict tem chaves/formas
    # completamente diferentes agora (Encoder3D/Decoder3D multi-ramo, ver
    # model/rcae.py). load_state_dict abaixo vai falhar alto e claro nesse
    # caso; e preciso retreinar do zero com o pipeline novo.
    model = RCAE(lstm_size=ckpt_args.get("lstm_size", 48)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint carregado (epoca {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss')})")

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]
    scheme_dir = Path(args.scheme_dir)
    out_dir = Path(args.out_dir)

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        scheme_path = scheme_dir / f"{tag}_scheme.npz"
        key = f"{args.shell_b}__{args.n_level}"
        if not scheme_path.exists():
            continue
        scheme = np.load(scheme_path)
        if f"{key}__input" not in scheme.files:
            print(f"[aviso] {tag}: sem esquema para shell={args.shell_b} n={args.n_level}")
            continue
        input_idx = scheme[f"{key}__input"]
        target_idx = scheme[f"{key}__target"]

        bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
        data, affine, header = load_dwi(e.dwi_path)
        shells = split_shells(bvals, tol=args.shell_tol)
        b0_mean = data[..., shells[0]].mean(axis=-1)
        mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

        # normalizacao por percentil DENTRO da mascara e da shell (xmax =
        # percentil 99), igual ao que o modelo viu no treino -- ver
        # utils/dataset.py:DWIPatchDataset._load_subject. NAO e mais
        # divisao por b0 por voxel (normalizacao antiga, incompativel com
        # os checkpoints treinados pelo pipeline atual).
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

        input_bvecs_t = torch.from_numpy(bvecs[input_idx].astype(np.float32)).unsqueeze(0).to(device)
        input_bvals_t = torch.from_numpy(bvals[input_idx].astype(np.float32)).unsqueeze(0).to(device)
        target_bvecs_t = torch.from_numpy(bvecs[target_idx].astype(np.float32)).unsqueeze(0).to(device)
        target_bvals_t = torch.from_numpy(bvals[target_idx].astype(np.float32)).unsqueeze(0).to(device)

        ps = args.patch_size
        origins = sliding_window_origins(shape3d, ps, args.stride)
        with torch.no_grad():
            for (ox, oy, oz) in origins:
                sl = (slice(ox, ox + ps), slice(oy, oy + ps), slice(oz, oz + ps))
                if not mask[sl].any():
                    continue
                patch = signal[sl][..., input_idx]  # (ps,ps,ps,n_in)
                input_vols = np.moveaxis(patch, -1, 0)[:, None].astype(np.float32)
                input_vols_t = torch.from_numpy(input_vols).unsqueeze(0).to(device)  # (1,n_in,1,ps,ps,ps)

                pred = model(input_vols_t, input_bvecs_t, input_bvals_t,
                             target_bvecs_t, target_bvals_t, b_ref=args.shell_b)
                # pred: (1, n_target, 1, ps, ps, ps) -> (ps,ps,ps,n_target)
                pred_np = pred[0, :, 0].permute(1, 2, 3, 0).cpu().numpy()

                pred_accum[sl] += pred_np
                weight_accum[sl] += 1.0

        weight_safe = np.where(weight_accum > 0, weight_accum, 1.0)
        pred_signal = pred_accum / weight_safe[..., None]
        pred_dwi = pred_signal * xmax  # desfaz a normalizacao por percentil (ver acima)
        pred_dwi[~mask] = 0.0

        shell_out = out_dir / tag / f"shell{int(args.shell_b)}"
        sub_out = shell_out / f"n{args.n_level}"
        sub_out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred_dwi.astype(np.float32), affine), sub_out / "recon_target.nii.gz")
        np.save(sub_out / "target_idx.npy", target_idx)
        # mascara e identica pra todo nivel dessa (sujeito, shell) -- grava uma vez em
        # shell_out (nivel acima), nao a cada n_level; se outro array task (nivel
        # diferente, mesma shell) ja escreveu, nao repete (so desperdicaria I/O)
        mask_path = shell_out / "mask.npy"
        if not mask_path.exists():
            np.save(mask_path, mask)
        print(f"{tag}: reconstrucao RCAE salva em {sub_out}")


if __name__ == "__main__":
    main()