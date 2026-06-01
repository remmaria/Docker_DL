import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def save_debug_image(pred, target, epoch, folder="debug_images"):
    """Salva um painel Target | Pred | Erro no slice central do volume."""
    os.makedirs(folder, exist_ok=True)
    idx = pred.shape[-1] // 2
    p = pred[0, 0, :, :, idx].cpu().float().numpy()
    t = target[0, 0, :, :, idx].cpu().float().numpy()
    error    = np.abs(p - t)
    combined = np.hstack([t, p, error])
    plt.imsave(f"{folder}/val_full_epoch_{epoch}.png", combined, cmap='jet')


def plot_q_space_polar(bvecs, target_idx, neighbor_indices, save_path):
    """Vista plana local (projeção 2D) da seleção de vizinhos em torno do target."""
    target_v = bvecs[target_idx]

    v_ref  = np.array([1, 0, 0]) if abs(target_v[0]) < 0.9 else np.array([0, 1, 0])
    orto_x = v_ref - np.dot(v_ref, target_v) * target_v
    orto_x /= np.linalg.norm(orto_x)
    orto_y = np.cross(target_v, orto_x)

    coords_2d = []
    colors    = []

    for idx in neighbor_indices:
        v_neigh  = bvecs[idx]
        is_flipped = np.dot(v_neigh, target_v) < 0
        v_eff    = -v_neigh if is_flipped else v_neigh
        x = np.dot(v_eff, orto_x)
        y = np.dot(v_eff, orto_y)
        coords_2d.append([x, y])
        colors.append('cyan' if is_flipped else 'blue')

    coords_2d = np.array(coords_2d)

    plt.figure(figsize=(6, 6))
    plt.scatter(0, 0, c='red', s=200, marker='*', label='Target (Centro)', zorder=5)

    for i in range(len(coords_2d)):
        plt.scatter(
            coords_2d[i, 0], coords_2d[i, 1],
            c=colors[i], s=100, edgecolors='black',
            label='Vizinho' if i == 0 else ""
        )
        plt.plot([0, coords_2d[i, 0]], [0, coords_2d[i, 1]], 'k--', alpha=0.2)

    circle = plt.Circle((0, 0), 0.1, color='gray', fill=False, linestyle=':', alpha=0.5)
    plt.gca().add_patch(circle)
    plt.axhline(0, color='black', lw=0.5, alpha=0.3)
    plt.axvline(0, color='black', lw=0.5, alpha=0.3)

    plt.title("Vista Local do Target (Projeção Plana)")
    plt.xlabel("Eixo Ortogonal X")
    plt.ylabel("Eixo Ortogonal Y")
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.2)
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close('all')


def plot_q_space_selection_antipodal(bvals, bvecs, target_idx, neighbor_indices, save_path):
    """Plot 3D da seleção antipodal no espaço-q."""
    q_coords  = np.sqrt(bvals[:, None]) * bvecs
    q_target  = q_coords[target_idx]

    hemi_neighbors    = []
    rebatido_indices  = []

    for idx in neighbor_indices:
        q_neigh = q_coords[idx]
        if np.dot(q_neigh, q_target) < 0:
            hemi_neighbors.append(-q_neigh)
            rebatido_indices.append(True)
        else:
            hemi_neighbors.append(q_neigh)
            rebatido_indices.append(False)

    hemi_neighbors   = np.array(hemi_neighbors)
    rebatido_indices = np.array(rebatido_indices)

    direct_neighbors  = hemi_neighbors[~rebatido_indices]
    flipped_neighbors = hemi_neighbors[rebatido_indices]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(q_coords[:, 0], q_coords[:, 1], q_coords[:, 2],
               alpha=0.05, c='gray', s=5)
    ax.scatter(q_target[0], q_target[1], q_target[2],
               c='red', s=100, marker='*', label='Target')

    if len(direct_neighbors) > 0:
        ax.scatter(direct_neighbors[:, 0], direct_neighbors[:, 1], direct_neighbors[:, 2],
                   c='blue', s=50, label='Vizinhos Diretos')
    if len(flipped_neighbors) > 0:
        ax.scatter(flipped_neighbors[:, 0], flipped_neighbors[:, 1], flipped_neighbors[:, 2],
                   c='purple', s=50, edgecolors='black', label='Vizinhos Rebatidos')

    for q_n in hemi_neighbors:
        ax.plot([0, q_n[0]], [0, q_n[1]], [0, q_n[2]], 'k--', alpha=0.1)

    ax.set_title("Seleção Antipodal (Hemi-Esferizada) no Espaço-Q")
    ax.set_xlabel("qx"); ax.set_ylabel("qy"); ax.set_zlabel("qz")
    plt.legend()
    plt.savefig(save_path)
    plt.close('all')


def plot_q_space_selection(bvals, bvecs, target_idx, neighbor_indices, save_path):
    """Plot 3D simples da seleção de vizinhos no espaço-q."""
    q_coords = np.sqrt(bvals[:, None]) * bvecs

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(q_coords[:, 0], q_coords[:, 1], q_coords[:, 2],
               alpha=0.1, c='gray', s=10, label='Todas as Direções')
    ax.scatter(q_coords[target_idx, 0], q_coords[target_idx, 1], q_coords[target_idx, 2],
               c='red', s=100, marker='*', label=f'Target (b{int(bvals[target_idx])})')
    ax.scatter(q_coords[neighbor_indices, 0], q_coords[neighbor_indices, 1], q_coords[neighbor_indices, 2],
               c='blue', s=50, label='Vizinhos Selecionados')

    for idx in neighbor_indices:
        ax.plot([0, q_coords[idx, 0]], [0, q_coords[idx, 1]], [0, q_coords[idx, 2]],
                'b--', alpha=0.3)

    ax.set_title("Seleção de Vizinhos no Espaço-Q")
    ax.set_xlabel("qx"); ax.set_ylabel("qy"); ax.set_zlabel("qz")
    ax.legend()

    limit = np.sqrt(bvals.max()) + 5
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    plt.savefig(save_path)
    plt.close('all')

def save_debug_documentation_png(
    neighbors,
    target,
    output_final,
    res_predito,
    step,
    save_dir,
    origin_b,
    target_b,
    alpha,
    query_coord,
    neighbors_coords,
    media_ponderada=None,   # FIX 1: recebe a média já computada pelo modelo
):
    """
    Painel de debug 2×4 com patch fixo ao longo do treino.

    Correções aplicadas
    -------------------
    FIX 1 — media_ponderada vem do modelo (não recalculada com temperatura errada)
             Se None (retrocompatibilidade), recalcula com os mesmos hiperparâmetros
             do model_v2: temperatura 0.3 e penalidade de b-value.

    FIX 2 — vmax dinâmico por percentil (não hardcoded em 0.5)
             Vizinho, Média, Target e Predição usam o mesmo vmax baseado no
             percentil 99 do target, para que mudanças na predição sejam visíveis.

    FIX 3 — MAE numérico nas legendas de erro
             Facilita acompanhar a convergência sem precisar inspecionar os pixels.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    slice_idx = target.shape[-1] // 2
    B, K = neighbors.shape[0], neighbors.shape[1]

    # =====================================================
    # FIX 1: usa média do modelo se disponível;
    #        caso contrário recalcula com os mesmos parâmetros do model_v2
    # =====================================================
    if media_ponderada is not None:
        media_ponderada_vol = media_ponderada
        with torch.no_grad():
            target_v = query_coord[:, 1:].float()
            neigh_vs = neighbors_coords[:, :, 1:].float()
            neigh_bs = neighbors_coords[:, :, 0].float()
            target_b_coord = query_coord[:, 0:1].float()
            dot = torch.abs(torch.einsum('bi,bki->bk', target_v, neigh_vs))
            b_diff = torch.abs(neigh_bs - target_b_coord)
            combined = (dot / 0.3) - (b_diff * 1.0)   # mesmos hiperparâmetros do model_v2
            weights = torch.softmax(combined, dim=1)
    else:
        # Fallback: recalcula como o model_v2 faria (temperatura 0.3 + penalidade b)
        with torch.no_grad():
            target_v   = query_coord[:, 1:].float()
            neigh_vs   = neighbors_coords[:, :, 1:].float()
            neigh_bs   = neighbors_coords[:, :, 0].float()
            target_b_c = query_coord[:, 0:1].float()
            dot        = torch.abs(torch.einsum('bi,bki->bk', target_v, neigh_vs))
            b_diff     = torch.abs(neigh_bs - target_b_c)
            combined   = (dot / 0.3) - (b_diff * 1.0)
            weights    = torch.softmax(combined, dim=1)
            weights_vol = weights.view(B, K, 1, 1, 1, 1)
            media_ponderada_vol = torch.sum(neighbors.float() * weights_vol, dim=1)

    # =====================================================
    # dominant neighbor (vizinho com maior peso)
    # =====================================================
    top_w_idx = torch.argmax(weights[0]).item()

    dominant_neighbor = (
        neighbors[0, top_w_idx, 0, :, :, slice_idx]
        .detach().cpu().float().numpy()
    ) * alpha

    # =====================================================
    # slices 2D para o plot
    # =====================================================
    mean_img = (
        media_ponderada_vol[0, 0, :, :, slice_idx]
        .detach().cpu().float().numpy()
    ) * alpha

    target_img = (
        target[0, 0, :, :, slice_idx]
        .detach().cpu().float().numpy()
    ) * alpha

    pred_img = (
        output_final[0, 0, :, :, slice_idx]
        .detach().cpu().float().numpy()
    ) * alpha

    residual_img = (
        res_predito[0, 0, :, :, slice_idx]
        .detach().cpu().float().numpy()
    ) * alpha

    # =====================================================
    # mapas de erro
    # =====================================================
    diff_mean_target  = mean_img  - target_img
    diff_pred_target  = pred_img  - target_img
    diff_improvement  = diff_mean_target - diff_pred_target   # positivo = modelo melhorou

    mae_mean = np.mean(np.abs(diff_mean_target))
    mae_pred = np.mean(np.abs(diff_pred_target))
    delta_mae = mae_mean - mae_pred   # positivo = modelo melhorou

    # limites simétricos independentes para cada mapa de erro
    diff_mean_lim    = np.percentile(np.abs(diff_mean_target), 99)  + 1e-8
    diff_pred_lim    = np.percentile(np.abs(diff_pred_target), 99)  + 1e-8
    improvement_lim  = np.percentile(np.abs(diff_improvement), 99)  + 1e-8
    residual_lim     = np.percentile(np.abs(residual_img),     99)  + 1e-8

    # =====================================================
    # FIX 2: vmax dinâmico — percentil 99 do target (não 0.5 fixo)
    # =====================================================
    vmax = float(np.percentile(target_img, 99)) + 1e-8
    vmax = max(vmax, 1e-4)   # evita vmax=0 em patches ruins

    # =====================================================
    # labels de b-value
    # =====================================================
    b_in = (
        origin_b[0].item() if torch.is_tensor(origin_b)
        else (origin_b[0] if hasattr(origin_b, '__len__') else origin_b)
    )
    b_out = (
        target_b[0].item() if torch.is_tensor(target_b)
        else (target_b[0] if hasattr(target_b, '__len__') else target_b)
    )

    # =====================================================
    # plotting
    # =====================================================
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"Step {step} | b{int(b_in)} → b{int(b_out)} | "
        f"MAE Média={mae_mean:.4f}  MAE Pred={mae_pred:.4f}  ΔMAE={delta_mae:+.4f}",
        fontsize=11, y=1.01
    )

    # --- linha 0 ---

    axes[0, 0].imshow(dominant_neighbor, cmap='jet', vmin=0, vmax=vmax)
    axes[0, 0].set_title(
        f"Vizinho Dominante\n(b{int(b_in)})  w={weights[0, top_w_idx]:.2f}"
    )

    axes[0, 1].imshow(mean_img, cmap='jet', vmin=0, vmax=vmax)
    axes[0, 1].set_title("Média Ponderada (modelo)")

    im_mean = axes[0, 2].imshow(
        diff_mean_target, cmap='seismic',
        vmin=-diff_mean_lim, vmax=diff_mean_lim
    )
    axes[0, 2].set_title(f"Erro Média vs Target\nMAE={mae_mean:.4f}")
    fig.colorbar(im_mean, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_improve = axes[0, 3].imshow(
        diff_improvement, cmap='seismic',
        vmin=-improvement_lim, vmax=improvement_lim
    )
    axes[0, 3].set_title(
        f"Melhora do Modelo\n(Erro Média − Erro Pred)  ΔMAE={delta_mae:+.4f}"
    )
    fig.colorbar(im_improve, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # --- linha 1 ---

    axes[1, 0].imshow(target_img, cmap='jet', vmin=0, vmax=vmax)
    axes[1, 0].set_title(f"Target Real (b{int(b_out)})")

    axes[1, 1].imshow(pred_img, cmap='jet', vmin=0, vmax=vmax)
    axes[1, 1].set_title("Predição Final - {pred_img:.6f}")

    im_pred = axes[1, 2].imshow(
        diff_pred_target, cmap='seismic',
        vmin=-diff_pred_lim, vmax=diff_pred_lim
    )
    axes[1, 2].set_title(f"Erro Predição vs Target\nMAE={mae_pred:.4f}")
    fig.colorbar(im_pred, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im_res = axes[1, 3].imshow(
        residual_img, cmap='seismic',
        vmin=-residual_lim, vmax=residual_lim
    )
    axes[1, 3].set_title(
        f"Resíduo Aprendido\n|res| máx={residual_lim:.4f}"
    )
    fig.colorbar(im_res, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        if ax.has_data():
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"doc_step_{step}.png"),
        bbox_inches='tight', dpi=120
    )
    plt.close('all')


def save_comparison_png(
    neighbors, target, output_final, res_predito,
    step, save_dir, origin_b, target_b, alpha
):
    """Painel 1×5: Média Vizinhos | Target | Predição | Resíduo | Erro Final."""
    os.makedirs(save_dir, exist_ok=True)
    slice_idx = target.shape[-1] // 2

    t_img = target[0, 0, :, :, slice_idx].detach().cpu().float().numpy() * alpha
    o_img = output_final[0, 0, :, :, slice_idx].detach().cpu().float().numpy() * alpha
    r_img = res_predito[0, 0, :, :, slice_idx].detach().cpu().float().numpy() * alpha

    mean_neighbors = torch.mean(neighbors[0].float(), dim=0)
    n_img = mean_neighbors[0, :, :, slice_idx].detach().cpu().numpy() * alpha

    erro_real_vol = (target[0, 0] - output_final[0, 0]).detach().cpu().float().numpy() * alpha
    e_img = erro_real_vol[:, :, slice_idx]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    vmax_shared = max(n_img.max(), t_img.max(), o_img.max(), 1e-8)

    b_in  = origin_b[0].item() if torch.is_tensor(origin_b) else origin_b
    b_out = target_b[0].item() if torch.is_tensor(target_b) else target_b
    info_text = f"In: b{int(b_in)} | Out: b{int(b_out)}"

    axes[0].imshow(n_img, cmap='jet', vmin=0, vmax=vmax_shared)
    axes[0].set_title(f"Média Vizinhos\n{info_text}")
    axes[0].axis('off')

    axes[1].imshow(t_img, cmap='jet', vmin=0, vmax=vmax_shared)
    axes[1].set_title(f"Target Real\n(b{int(b_out)})")
    axes[1].axis('off')

    axes[2].imshow(o_img, cmap='jet', vmin=0, vmax=vmax_shared)
    axes[2].set_title("Predição\n(Média + Res)")
    axes[2].axis('off')

    res_lim = np.percentile(np.abs(r_img), 99) + 1e-8
    im4 = axes[3].imshow(r_img, cmap='seismic', vmin=-res_lim, vmax=res_lim)
    axes[3].set_title("Ajuste Residual\n(O que mudou)")
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis('off')

    im5 = axes[4].imshow(e_img, cmap='seismic', vmin=-res_lim, vmax=res_lim)
    axes[4].set_title("Erro Final\n(Target - Pred)")
    fig.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)
    axes[4].axis('off')

    plt.tight_layout()
    filename = f"step_{step:05d}_b{int(b_in)}_to_b{int(b_out)}.png"
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close('all')


def backup_code(folder_checkpoint):
    """Copia os arquivos principais para a pasta do experimento."""
    code_backup_dir = os.path.join(folder_checkpoint, "code_backup")
    os.makedirs(code_backup_dir, exist_ok=True)

    files_to_save = [
        "dataset.py",
        "inference.py",
        "losses.py",
        "metrics.py",
        "model.py",
        "model_v2.py"
        "run_inf.sh",
        "run_train.sh",
        "train_config.yaml",
        "train.py",
        "utils.py",
    ]

    for f in files_to_save:
        if os.path.exists(f):
            dest_path = os.path.join(code_backup_dir, f)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(f, dest_path)

    print(f"📦 Backup do código realizado em: {code_backup_dir}", flush=True)