#!/usr/bin/env python3
"""
Etapa 5e: aplica uma HFD3D treinada (checkpoint da etapa 4d, ver
model/hfd3d.py e scripts/04d_train_hfd.py) aos sujeitos de um split
(tipicamente "test"), reconstruindo as direcoes-alvo do esquema de trincas
(scripts/02b_build_rrin_triplets.py) volume inteiro (blocos/patches com
overlap, costurados por media). PORT quase 1:1 de
scripts/05d_reconstruct_amt.py -- unica coisa que muda de verdade e qual
modelo e carregado (model.hfd3d.build_hfd_model em vez de
model.amt3d.build_amt_model) e quais chaves de arquitetura sao lidas do
checkpoint (corr_radius, num_timesteps, num_sample_steps em vez de
num_fields). NAO precisa da AMT3D professora aqui -- ela so e usada durante
o TREINO (ver scripts/04d_train_hfd.py); na reconstrucao, a HFD3D ja
aprendeu a gerar fluxo sozinha via amostragem DDIM (model.hfd3d.HFD3D.forward
chama sample_flow internamente, ver docstring de model/hfd3d.py).

Reconstroi TODOS os alvos presentes no esquema de trincas para este
(shell,n_level), MESMO os marcados `valid=False` -- MESMO motivo/uso de
scripts/05b_reconstruct_rrin.py/scripts/05d_reconstruct_amt.py (comparar na
mesma cobertura de direcoes held-out, e permitir a
scripts/06_evaluate_reconstruction.py estratificar o erro por validade
geometrica da trinca).

IMPORTANTE sobre o nome do arquivo de validade (`rrin_valid.npy`): mesma
justificativa ja documentada em scripts/05d_reconstruct_amt.py -- o nome
"historico" e reaproveitado de proposito (nao e copiar-e-colar por engano)
para que scripts/06_evaluate_reconstruction.py estratifique
aggregate_valid/aggregate_invalid para HFD3D sem nenhuma modificacao no
script de avaliacao.

NOTA DE CUSTO: cada patch reconstruido roda o loop DDIM completo
(`--num-sample-steps`, tipicamente 6) do denoiser -- mais lento por patch
que RRIN3D/AMT3D (1 forward so). Considere `--stride` maior (menos patches
sobrepostos) se o tempo de reconstrucao for proibitivo, ao custo de mais
artefato de costura entre blocos (mesmo trade-off ja conhecido de
scripts/05b_reconstruct_rrin.py/05d_reconstruct_amt.py).

Uso:
    python scripts/05e_reconstruct_hfd.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --checkpoint work_dir/hfd_checkpoints/shell1000_n10/best.pt \
        --shell-b 1000 --n-level 10 \
        --out-dir work_dir/hfd_recon \
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
from model.hfd3d import build_hfd_model


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
    ap.add_argument("--triplets-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--num-sample-steps", type=int, default=None,
                     help="sobrescreve o num_sample_steps salvo no checkpoint (so afeta "
                          "custo/qualidade da amostragem, nunca shape/peso -- ver "
                          "model.hfd3d.HFD3D). Default: usa o valor do checkpoint.")
    ap.add_argument("--subjects", default=None,
                     help="mesma convencao de --subjects em 05_reconstruct_rcae.py/"
                          "05b_reconstruct_rrin.py/05d_reconstruct_amt.py")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import nibabel as nib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    use_quality_cond = ckpt_args.get("use_quality_cond", False)
    corr_radius = ckpt_args.get("corr_radius", 3)
    norm_type = ckpt_args.get("norm_type", "instance")
    num_timesteps = ckpt_args.get("num_timesteps", 1000)
    num_sample_steps = args.num_sample_steps if args.num_sample_steps is not None \
        else ckpt_args.get("num_sample_steps", 6)
    model = build_hfd_model(base_ch=ckpt_args.get("base_ch", 16),
                             max_disp=ckpt_args.get("max_disp", 0.5),
                             corr_radius=corr_radius,
                             use_quality_cond=use_quality_cond,
                             norm_type=norm_type,
                             num_timesteps=num_timesteps,
                             num_sample_steps=num_sample_steps).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint carregado (epoca {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss')}, "
          f"corr_radius={corr_radius}, norm_type={norm_type}, num_timesteps={num_timesteps}, "
          f"num_sample_steps={num_sample_steps} "
          f"{'(sobrescrito via --num-sample-steps)' if args.num_sample_steps is not None else '(do checkpoint)'})")

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

    print(f"Reconstruindo {len(entries)} sujeito(s): {[_tag_of(e) for e in entries]}", flush=True)

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
        bvec_a_all = bvecs_t[pair_a].to(device)          # (n_target, 3)
        bvec_b_all = bvecs_t[pair_b].to(device)
        bvec_t_all = bvecs_t[target_idx].to(device)
        t_frac_all = torch.from_numpy(t_frac.astype(np.float32)).to(device)
        quality_all = None
        if use_quality_cond:
            quality_np = np.stack([residual_deg / 90.0, gap_deg / 90.0], axis=1).astype(np.float32)
            quality_all = torch.from_numpy(quality_np).to(device)

        ps = args.patch_size
        origins = sliding_window_origins(shape3d, ps, args.stride)
        with torch.no_grad():
            for (ox, oy, oz) in origins:
                sl = (slice(ox, ox + ps), slice(oy, oy + ps), slice(oz, oz + ps))
                if not mask[sl].any():
                    continue
                vol_a_patch = signal[sl][..., pair_a]  # (ps,ps,ps,n_target)
                vol_b_patch = signal[sl][..., pair_b]
                vol_a_t = torch.from_numpy(np.moveaxis(vol_a_patch, -1, 0)[:, None]
                                            .astype(np.float32)).to(device)
                vol_b_t = torch.from_numpy(np.moveaxis(vol_b_patch, -1, 0)[:, None]
                                            .astype(np.float32)).to(device)

                pred = model(vol_a_t, vol_b_t, bvec_a_all, bvec_b_all, bvec_t_all, t_frac_all,
                             quality=quality_all)
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
        # MESMO nome de arquivo usado por scripts/05b_reconstruct_rrin.py/
        # scripts/05d_reconstruct_amt.py (rrin_valid.npy) -- ver docstring
        # do modulo.
        np.save(sub_out / "rrin_valid.npy", valid)
        np.save(sub_out / "rrin_residual_deg.npy", residual_deg)
        np.save(sub_out / "rrin_gap_deg.npy", gap_deg)
        mask_path = shell_out / "mask.npy"
        if not mask_path.exists():
            np.save(mask_path, mask)
        n_invalid = int((~valid).sum())
        print(f"{tag}: reconstrucao HFD3D salva em {sub_out} "
              f"({n_invalid}/{n_target} alvos com trinca invalida)")


if __name__ == "__main__":
    main()