"""Visualizacao de debug para patches de dMRI (entrada / alvo / predicao /
contexto). Usado tanto por scripts/debug_visualize_patches.py (inspecao
manual, sem modelo) quanto por scripts/04_train_rcae.py (snapshot da
predicao a cada N epocas/batches, pra acompanhar visualmente a convergencia
do treino).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return x


def save_patch_debug_png(out_path, input_vols, target_vols, pred_vols=None, context=None,
                          max_dirs: int = 6, title: str = ""):
    """Salva um PNG com fatia axial central (plano XY, Z no meio do patch)
    em cmap='jet'. Linhas: input, target, predicao (se fornecida) e
    contexto (se fornecido).

    input_vols/target_vols/pred_vols: (N_dirs, 1, ps, ps, ps) -- numpy ou
    torch.Tensor (com ou sem grad). target_vols e pred_vols devem ter o
    mesmo N_out (pred e a saida do modelo pras mesmas direcoes-alvo).

    context: (C, ps, ps, ps) opcional -- o hidden state do encoder
    (RCAE.encoder(...), ANTES do decoder condicionar numa direcao-alvo
    especifica). E O MESMO pra qualquer direcao-alvo pedida (o decoder so
    recebe esse tensor + o bvec-alvo) -- plotamos a media nos canais, na
    MESMA fatia z_mid, repetida em toda coluna de proposito: se a linha de
    "pred" acima ficar visualmente IGUAL a essa linha (ou proporcional a
    ela), e evidencia direta de que a rede esta so devolvendo o contexto e
    ignorando o bvec-alvo (o "colapso condicional" que motivou essa linha
    extra). Usa escala de cor PROPRIA (nao a mesma de input/target/pred --
    e uma grandeza diferente, um estado latente da ConvLSTM, nao sinal de
    dMRI, entao nao faz sentido compartilhar vmin/vmax).
    """
    import matplotlib
    matplotlib.use("Agg")  # sem display no cluster
    import matplotlib.pyplot as plt

    input_vols = _to_numpy(input_vols)
    target_vols = _to_numpy(target_vols)
    pred_vols = _to_numpy(pred_vols) if pred_vols is not None else None
    context = _to_numpy(context) if context is not None else None

    ps = input_vols.shape[-1]
    z_mid = ps // 2

    n_in_show = min(max_dirs, input_vols.shape[0])
    n_out_show = min(max_dirs, target_vols.shape[0])
    n_cols = max(n_in_show, n_out_show)

    row_labels = ["input", "target"]
    row_data = [input_vols, target_vols]
    row_n_show = [n_in_show, n_out_show]
    if pred_vols is not None:
        row_labels.append("pred")
        row_data.append(pred_vols)
        row_n_show.append(n_out_show)
    n_rows = len(row_data) + (1 if context is not None else 0)

    # escala de cor COMPARTILHADA so entre input/target/pred (mesma
    # grandeza: sinal de dMRI normalizado) -- o contexto (se houver) tem
    # escala propria, ver docstring.
    vals = [d[:n, 0, :, :, z_mid] for d, n in zip(row_data, row_n_show)]
    all_vals = np.concatenate([v.ravel() for v in vals])
    vmin, vmax = np.percentile(all_vals, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # desvio-padrao ENTRE DIRECOES (eixo 0 = direcao), por voxel, media no
    # patch inteiro -- numero objetivo de "o quanto essa linha varia de
    # coluna pra coluna", pra nao depender de olhar a cor na escala
    # compartilhada acima (que pode escamotear uma variacao real mas
    # pequena, especialmente no pred). Ver titulo do plot.
    def _cross_dir_std(vols, n_show):
        return float(vols[:n_show, 0, :, :, z_mid].std(axis=0).mean())

    row_cross_std = [_cross_dir_std(d, n) for d, n in zip(row_data, row_n_show)]

    context_map = None
    ctx_vmin = ctx_vmax = None
    if context is not None:
        # media nos canais, mesma fatia z_mid -- UM SO mapa 2D (o contexto
        # nao depende de direcao-alvo nenhuma).
        context_map = context[:, :, :, z_mid].mean(axis=0)  # (ps, ps)
        ctx_vmin, ctx_vmax = np.percentile(context_map, [1, 99])
        if ctx_vmax <= ctx_vmin:
            ctx_vmax = ctx_vmin + 1e-6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows),
                              squeeze=False)

    im = None
    for r, (label, data, n_show, cstd) in enumerate(
            zip(row_labels, row_data, row_n_show, row_cross_std)):
        for c in range(n_cols):
            ax = axes[r, c]
            if c < n_show:
                slice_2d = data[c, 0, :, :, z_mid]
                im = ax.imshow(slice_2d, cmap="jet", vmin=vmin, vmax=vmax)
                if r == 0:
                    ax.set_title(f"dir {c}", fontsize=8)
                # media individual DESSE patch/direcao especifico -- mais
                # casas decimais do que a cor deixa perceber a olho, pra
                # dar pra confirmar se duas colunas sao REALMENTE iguais
                # (nao so parecidas na escala de cor).
                ax.text(0.5, -0.06, f"μ={slice_2d.mean():.5f}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=6)
            ax.axis("off")
        # cross-dir std no label da linha -- numero objetivo de quanto essa
        # linha varia de coluna pra coluna, pra comparar direto pred vs
        # target sem depender so da cor.
        axes[r, 0].text(-0.15, 0.5, f"{label}\n(std entre dir={cstd:.4f})", fontsize=8,
                         rotation=90, va="center", ha="center", transform=axes[r, 0].transAxes)

    im_ctx = None
    if context_map is not None:
        r = len(row_data)
        for c in range(n_cols):
            ax = axes[r, c]
            im_ctx = ax.imshow(context_map, cmap="jet", vmin=ctx_vmin, vmax=ctx_vmax)
            ax.text(0.5, -0.06, f"μ={context_map.mean():.5f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6)
            ax.axis("off")
        axes[r, 0].text(-0.15, 0.5, "contexto\n(media canais --\nigual p/ qualquer dir)",
                         fontsize=8, rotation=90, va="center", ha="center",
                         transform=axes[r, 0].transAxes)

    if im is not None:
        fig.colorbar(im, ax=axes[: len(row_data)], fraction=0.02, pad=0.02,
                      label="sinal (input/target/pred)")
    if im_ctx is not None:
        fig.colorbar(im_ctx, ax=axes[len(row_data):], fraction=0.02, pad=0.02,
                      label="contexto (escala propria)")
    fig.suptitle(title, fontsize=10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # tambem no stdout (nao so no PNG) -- da pra grepar o .out do SLURM
    # atras da tendencia do cross-dir std do pred sem abrir imagem nenhuma,
    # ex.: `grep "cross-dir std" logs/train.*.out`
    if pred_vols is not None:
        cross_std_input, cross_std_target, cross_std_pred = row_cross_std
        print(f"[debug] cross-dir std -- input={cross_std_input:.4f} "
              f"target={cross_std_target:.4f} pred={cross_std_pred:.4f} "
              f"(pred/target={cross_std_pred / max(cross_std_target, 1e-8):.3f}) "
              f"em {out_path}", flush=True)
    return out_path