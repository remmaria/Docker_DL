"""
metrics.py
----------
Métricas quantitativas para avaliar a qualidade da reconstrução do sinal DWI.

Métricas implementadas:
  1. MAE / RMSE no sinal normalizado
  2. SSIM no espaço de FA (Fractional Anisotropy)
  3. Angular Error no pico principal de difusão
  4. Shell-wise MAE (diagnóstico por b-value)

Também inclui funções de visualização para inspeção qualitativa.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")   # sem display — compatível com servidor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# 1. Métricas básicas de reconstrução
# ---------------------------------------------------------------------------

def compute_mae(S_pred: np.ndarray, S_target: np.ndarray) -> float:
    """Mean Absolute Error no sinal normalizado."""
    return float(np.abs(S_pred - S_target).mean())


def compute_rmse(S_pred: np.ndarray, S_target: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(((S_pred - S_target) ** 2).mean()))


def compute_psnr(S_pred: np.ndarray, S_target: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB). Sinal normalizado → max_val=1."""
    mse = ((S_pred - S_target) ** 2).mean()
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_shell_metrics(
    S_pred: np.ndarray,     # (N_voxels, N_query)
    S_target: np.ndarray,   # (N_voxels, N_query)
    bvals: np.ndarray,      # (N_query,)
) -> Dict[str, float]:
    """
    Computa MAE por shell.
    Retorna dict: {"b0": mae, "b1000": mae, ...}
    """
    shells = np.unique(np.round(bvals, -2).astype(int))
    results = {}
    for b in shells:
        mask = np.abs(np.round(bvals, -2) - b) < 50
        if mask.sum() == 0:
            continue
        mae = compute_mae(S_pred[:, mask], S_target[:, mask])
        results[f"b{int(b)}"] = mae
    return results


# ---------------------------------------------------------------------------
# 2. Métricas no espaço de FA (requer dipy)
# ---------------------------------------------------------------------------

def compute_fa_from_signal(
    S_norm: np.ndarray,   # (N_voxels, N_dwi)
    bvals: np.ndarray,    # (N_dwi,)
    bvecs: np.ndarray,    # (N_dwi, 3)
) -> Optional[np.ndarray]:
    """
    Estima FA via DTI fitting (apenas single-shell, b≈1000).
    Retorna FA (N_voxels,) ou None se dipy não disponível.
    """
    try:
        from dipy.core.gradients import gradient_table
        from dipy.reconst.dti import TensorModel
        import dipy.reconst.dti as dti
    except ImportError:
        return None

    # Reconstrói sinal absoluto (assume S0=1 pois está normalizado)
    gtab = gradient_table(bvals, bvecs)
    model = TensorModel(gtab)

    # dipy espera shape (X, Y, Z, N_dwi) ou (N, N_dwi)
    S_4d = S_norm[:, None, None, :]   # (N, 1, 1, N_dwi)
    fit = model.fit(S_4d)
    fa = dti.fractional_anisotropy(fit.evals).squeeze()  # (N,)
    return np.clip(fa, 0, 1)


def compute_fa_mae(
    S_pred: np.ndarray,
    S_target: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
) -> Optional[float]:
    """MAE entre FA estimada a partir do sinal predito vs target."""
    fa_pred   = compute_fa_from_signal(S_pred, bvals, bvecs)
    fa_target = compute_fa_from_signal(S_target, bvals, bvecs)
    if fa_pred is None or fa_target is None:
        return None
    return compute_mae(fa_pred, fa_target)


# ---------------------------------------------------------------------------
# 3. Avaliação completa em batch
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    n_batches: int = 50,   # limita para não demorar
) -> Dict[str, float]:
    """
    Roda avaliação completa e retorna dict de métricas.
    """
    model.eval()

    all_pred   = []
    all_target = []
    all_bvals  = []

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        x_ctx    = batch["x_context"].to(device)
        ctx_mask = batch["ctx_mask"].to(device)
        q_query  = batch["q_query"].to(device)
        S_target = batch["S_target"]
        q_mask   = batch["q_mask"]
        b_vals   = batch["bvals_query"]

        S_pred, _ = model(x_ctx, q_query, ctx_mask)
        S_pred = S_pred.squeeze(-1).cpu()

        # Coleta apenas posições válidas
        valid = ~q_mask
        for b_idx in range(S_pred.shape[0]):
            v = valid[b_idx]
            all_pred.append(S_pred[b_idx, v].numpy())
            all_target.append(S_target[b_idx, v].numpy())
            all_bvals.append(b_vals[b_idx, v].numpy())

    all_pred   = np.concatenate(all_pred)
    all_target = np.concatenate(all_target)
    all_bvals  = np.concatenate(all_bvals)

    metrics = {
        "mae":  compute_mae(all_pred, all_target),
        "rmse": compute_rmse(all_pred, all_target),
        "psnr": compute_psnr(all_pred, all_target),
    }

    # Por shell
    shell_metrics = compute_shell_metrics(
        all_pred[None, :],
        all_target[None, :],
        all_bvals,
    )
    metrics.update(shell_metrics)

    return metrics


# ---------------------------------------------------------------------------
# 4. Visualizações
# ---------------------------------------------------------------------------

def plot_qspace_prediction(
    S_context: np.ndarray,    # (N_ctx,)   sinal de contexto
    b_context: np.ndarray,    # (N_ctx,)
    S_pred: np.ndarray,       # (N_query,) predição
    S_target: np.ndarray,     # (N_query,)
    b_query: np.ndarray,      # (N_query,)
    title: str = "Q-Space Reconstruction",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota sinal DWI vs b-value para um voxel:
    - Pontos de contexto (dados de entrada)
    - Predição nas direções mascaradas
    - Target nas direções mascaradas

    Permite inspecionar visualmente se a rede está interpolando
    corretamente ao longo do eixo b.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ---- Painel 1: Sinal vs b-value ----
    ax = axes[0]
    ax.scatter(b_context, S_context, c="steelblue", alpha=0.6,
               s=40, label="Contexto (visível)", zorder=3)
    ax.scatter(b_query, S_target, c="forestgreen", marker="^",
               s=60, label="Target (mascarado)", zorder=4)
    ax.scatter(b_query, S_pred, c="tomato", marker="x",
               s=80, linewidths=2, label="Predição", zorder=5)

    # Linha de erro
    for bp, sp, st in zip(b_query, S_pred, S_target):
        ax.plot([bp, bp], [sp, st], c="gray", alpha=0.4, lw=1)

    ax.set_xlabel("b-value (s/mm²)", fontsize=11)
    ax.set_ylabel("S normalizado", fontsize=11)
    ax.set_title("Sinal vs b-value", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    mae = np.abs(S_pred - S_target).mean()
    ax.set_title(f"Sinal vs b-value  (MAE={mae:.4f})", fontsize=11)

    # ---- Painel 2: Scatter pred vs target ----
    ax2 = axes[1]
    ax2.scatter(S_target, S_pred, c=b_query, cmap="viridis",
                s=50, alpha=0.8, zorder=3)
    lims = [min(S_target.min(), S_pred.min()) - 0.02,
            max(S_target.max(), S_pred.max()) + 0.02]
    ax2.plot(lims, lims, "k--", lw=1.5, alpha=0.5, label="y=x (perfeito)")
    ax2.set_xlabel("Target", fontsize=11)
    ax2.set_ylabel("Predição", fontsize=11)
    ax2.set_title("Predição vs Target (cor = b-value)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(b_query.min(), b_query.max()))
    plt.colorbar(sm, ax=ax2, label="b-value")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_training_curves(
    log_path: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Lê logs do TensorBoard (via tfevents) ou um CSV simples e plota curvas.
    Formato CSV esperado: epoch,train_loss,val_loss,train_recon,train_mono
    """
    import pandas as pd

    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train", color="steelblue")
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="Val", color="tomato")
    axes[0].set_title("Loss Total")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "train_recon" in df.columns:
        axes[1].plot(df["epoch"], df["train_recon"], label="Recon", color="steelblue")
    if "train_mono" in df.columns:
        axes[1].plot(df["epoch"], df["train_mono"], label="Mono", color="orange")
    if "train_smooth" in df.columns:
        axes[1].plot(df["epoch"], df["train_smooth"], label="Smooth", color="green")
    axes[1].set_title("Componentes da Loss")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_latent_space(
    z: np.ndarray,           # (N, latent_dim)
    protocol_ids: np.ndarray, # (N,)
    protocol_names: List[str],
    method: str = "umap",    # 'umap' | 'tsne' | 'pca'
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualiza o espaço latente com redução de dimensionalidade.
    Útil para verificar se protocolos estão separados ou misturados.
    Objetivo: os clusters devem se sobrepor (protocolo disentangled).
    """
    from sklearn.decomposition import PCA

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            z_2d = reducer.fit_transform(z)
            method_name = "UMAP"
        except ImportError:
            method = "pca"

    if method == "pca":
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z)
        method_name = f"PCA (var={pca.explained_variance_ratio_.sum():.1%})"

    elif method == "tsne":
        from sklearn.manifold import TSNE
        z_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z)
        method_name = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(protocol_names)))
    for pid, (name, color) in enumerate(zip(protocol_names, colors)):
        mask = protocol_ids == pid
        if mask.sum() == 0:
            continue
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=[color], label=name, alpha=0.5, s=15)

    ax.set_title(f"Espaço Latente — {method_name}\n"
                 f"(ideal: clusters sobrepostos = protocolo-invariante)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig