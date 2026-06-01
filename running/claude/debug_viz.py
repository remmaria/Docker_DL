"""
debug_viz.py — corrigido: model.train() sempre restaurado via finally
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional
import torch

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    import dipy.reconst.dti as dti
    HAS_DIPY = True
except ImportError:
    HAS_DIPY = False


@torch.no_grad()
def save_debug_images(
    model,
    val_dataset,
    output_dir: str,
    epoch: int,
    device: torch.device,
    n_voxels: int = 6,
    seed: int = 0,
):
    """
    Salva PNGs de diagnóstico. Garante que model.train() é restaurado
    mesmo que ocorra uma exceção — bug que causava divergência no treino.
    """
    was_training = model.training   # guarda estado original
    model.eval()

    try:
        rng       = np.random.default_rng(seed)
        debug_dir = Path(output_dir) / "debug" / f"epoch_{epoch:03d}"
        debug_dir.mkdir(parents=True, exist_ok=True)

        n_subjects      = len(val_dataset.meta)
        chosen_subjects = rng.choice(n_subjects, min(n_voxels, n_subjects), replace=False)

        voxel_data = []
        for s_idx in chosen_subjects:
            meta     = val_dataset.meta[s_idx]
            S_norm   = val_dataset._get_S_norm(s_idx)
            n_vox    = len(meta["valid_voxels"])
            v_idx    = int(rng.integers(0, n_vox))
            xyz      = meta["valid_voxels"][v_idx]
            X, Y, Z  = meta["shape_xyz"]
            flat_idx = int(xyz[0]) * Y * Z + int(xyz[1]) * Z + int(xyz[2])

            bvals  = meta["bvals"]
            bvecs  = meta["bvecs"]
            b_norm = meta["bvals_norm"]
            S_all  = S_norm[flat_idx].astype(np.float32)

            vox_rng    = np.random.default_rng(seed * 1000 + s_idx)
            dwi_idx    = np.where(bvals >= 50)[0]
            n_mask     = max(1, int(len(dwi_idx) * 0.30))
            masked_idx = vox_rng.choice(dwi_idx, n_mask, replace=False)
            query_mask = np.zeros(len(bvals), dtype=bool)
            query_mask[masked_idx] = True
            context_mask = ~query_mask

            x_context = torch.tensor(
                np.concatenate([b_norm[context_mask][:,None], bvecs[context_mask],
                                S_all[context_mask][:,None]], axis=-1),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            q_query = torch.tensor(
                np.concatenate([b_norm[query_mask][:,None], bvecs[query_mask]], axis=-1),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            S_pred, z = model(x_context, q_query)
            S_pred_np = S_pred.squeeze().cpu().numpy()

            voxel_data.append({
                "s_idx":       int(s_idx),
                "subject":     Path(meta["subject_dir"]).name,
                "shells":      np.unique(np.round(bvals, -2).astype(int)).tolist(),
                "bvals_ctx":   bvals[context_mask],
                "S_ctx":       S_all[context_mask],
                "bvals_query": bvals[query_mask],
                "S_target":    S_all[query_mask],
                "S_pred":      S_pred_np,
                "mae":         float(np.abs(S_pred_np - S_all[query_mask]).mean()),
                "z_norm":      float(z.norm().item()),
            })

        _save_summary_panel(voxel_data, debug_dir, epoch)
        for i, vd in enumerate(voxel_data):
            _save_voxel_plot(vd, debug_dir / f"voxel_{i:02d}_{vd['subject'][:20]}.png", epoch)

        print(f"  Debug PNGs salvos em: {debug_dir}")
        return debug_dir

    finally:
        model.train(was_training)   # restaura independente de exceção


def _save_summary_panel(voxel_data, out_dir, epoch):
    n     = len(voxel_data)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(f"Epoch {epoch:03d} — Q-space reconstruction (val set)",
                 fontsize=13, fontweight="bold", y=1.01)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for ax, vd in zip(axes_flat, voxel_data):
        _plot_qspace(ax, vd)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    path = out_dir.parent / f"summary_epoch_{epoch:03d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_voxel_plot(vd, out_path, epoch):
    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    _plot_qspace(ax1, vd, detailed=True)

    ax2 = fig.add_subplot(gs[1])
    sc = ax2.scatter(vd["S_target"], vd["S_pred"],
                     c=vd["bvals_query"], cmap="viridis", s=60, alpha=0.85, zorder=3)
    lims = [min(vd["S_target"].min(), vd["S_pred"].min()) - 0.02,
            max(vd["S_target"].max(), vd["S_pred"].max()) + 0.02]
    ax2.plot(lims, lims, "k--", lw=1.5, alpha=0.4, label="y=x")
    ax2.set_xlabel("Target S", fontsize=11)
    ax2.set_ylabel("Predição S", fontsize=11)
    ax2.set_title(f"Pred vs Target  (MAE={vd['mae']:.4f})", fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="b-value (s/mm²)")

    fig.suptitle(
        f"Epoch {epoch:03d} | {vd['subject']} | shells={vd['shells']} | "
        f"MAE={vd['mae']:.4f} | |z|={vd['z_norm']:.2f}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_qspace(ax, vd, detailed=False):
    ax.scatter(vd["bvals_ctx"], vd["S_ctx"],
               c="steelblue", s=25, alpha=0.5, label="Contexto", zorder=2)
    ax.scatter(vd["bvals_query"], vd["S_target"],
               c="forestgreen", marker="^", s=55, alpha=0.9, label="Target", zorder=4)
    ax.scatter(vd["bvals_query"], vd["S_pred"],
               c="tomato", marker="x", s=70, linewidths=2, label="Predição", zorder=5)
    for bq, sp, st in zip(vd["bvals_query"], vd["S_pred"], vd["S_target"]):
        ax.plot([bq, bq], [sp, st], c="gray", alpha=0.35, lw=1)
    ax.set_xlabel("b-value" if detailed else "", fontsize=10)
    ax.set_ylabel("S norm" if detailed else "", fontsize=10)
    title = (f"Sinal vs b-value | shells={vd['shells']}\nMAE={vd['mae']:.4f}"
             if detailed else f"{vd['subject'][:18]}\nMAE={vd['mae']:.4f}")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.1)


def save_fa_comparison(model, val_dataset, output_dir, epoch, device,
                       subject_idx=0, slice_idx=None):
    """FA map comparison — requer dipy."""
    if not HAS_DIPY:
        print("  FA map: dipy não disponível.")
        return

    was_training = model.training
    model.eval()
    try:
        meta   = val_dataset.meta[subject_idx]
        S_norm = val_dataset._get_S_norm(subject_idx)
        bvals  = meta["bvals"]
        bvecs  = meta["bvecs"]
        b_norm = meta["bvals_norm"]
        X, Y, Z = meta["shape_xyz"]

        b1000_mask = np.abs(np.round(bvals, -2) - 1000) < 50
        b0_mask    = bvals < 50
        if b1000_mask.sum() < 6:
            print(f"  FA map: sujeito {subject_idx} sem b1000 suficiente.")
            return

        if slice_idx is None:
            slice_idx = Z // 2

        print(f"  Gerando FA map (sujeito {subject_idx}, slice {slice_idx})...")

        S_slice_orig = S_norm.reshape(X, Y, Z, -1)[:, :, slice_idx, :]
        fa_orig = _estimate_fa_slice(S_slice_orig, bvals, bvecs)

        S_slice_pred = _reconstruct_slice(
            model, S_norm, meta, b_norm, bvecs,
            b1000_mask, b0_mask, slice_idx, X, Y, Z, device
        )
        fa_pred = _estimate_fa_slice(
            S_slice_pred, bvals[b1000_mask | b0_mask], bvecs[b1000_mask | b0_mask]
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"Epoch {epoch:03d} | FA map | {Path(meta['subject_dir']).name} | slice {slice_idx}",
            fontsize=12, fontweight="bold"
        )
        kw = dict(cmap="hot", vmin=0, vmax=1)
        im0 = axes[0].imshow(fa_orig.T, origin="lower", **kw)
        axes[0].set_title("FA original")
        plt.colorbar(im0, ax=axes[0], fraction=0.04)
        im1 = axes[1].imshow(fa_pred.T, origin="lower", **kw)
        axes[1].set_title("FA reconstruída")
        plt.colorbar(im1, ax=axes[1], fraction=0.04)
        diff = np.abs(fa_orig - fa_pred)
        im2 = axes[2].imshow(diff.T, origin="lower", cmap="RdYlGn_r", vmin=0, vmax=0.3)
        axes[2].set_title(f"Diferença  MAE={diff[diff>0].mean():.4f}")
        plt.colorbar(im2, ax=axes[2], fraction=0.04)
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        path = Path(output_dir) / "debug" / f"fa_map_epoch_{epoch:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  FA map salvo: {path}")
    finally:
        model.train(was_training)


def _estimate_fa_slice(S_slice, bvals, bvecs):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel
    import dipy.reconst.dti as dti_module
    gtab  = gradient_table(bvals, bvecs)
    model = TensorModel(gtab)
    fit   = model.fit(S_slice[:, :, None, :])
    return np.clip(dti_module.fractional_anisotropy(fit.evals).squeeze(), 0, 1)


@torch.no_grad()
def _reconstruct_slice(model, S_norm, meta, b_norm, bvecs, b1000_mask, b0_mask,
                        slice_idx, X, Y, Z, device):
    b_norm_q = b_norm[b1000_mask][:, None]
    g_q      = bvecs[b1000_mask]
    q_query  = torch.tensor(
        np.concatenate([b_norm_q, g_q], axis=-1), dtype=torch.float32
    ).unsqueeze(0).to(device)

    N_b1000 = b1000_mask.sum()
    N_b0    = b0_mask.sum()
    S_out   = np.zeros((X, Y, N_b1000 + N_b0), dtype=np.float32)
    ctx_mask = ~b1000_mask

    for xi in range(X):
        for yi in range(Y):
            flat_idx = xi * Y * Z + yi * Z + slice_idx
            S_all    = S_norm[flat_idx].astype(np.float32)
            x_ctx    = torch.tensor(
                np.concatenate([b_norm[ctx_mask][:,None], bvecs[ctx_mask],
                                S_all[ctx_mask][:,None]], axis=-1),
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            S_pred, _ = model(x_ctx, q_query)
            S_out[xi, yi, :N_b1000] = S_pred.squeeze().cpu().numpy()
            S_out[xi, yi, N_b1000:] = S_all[b0_mask]
    return S_out