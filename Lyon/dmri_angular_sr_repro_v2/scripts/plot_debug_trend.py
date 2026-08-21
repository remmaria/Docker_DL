#!/usr/bin/env python3
"""
Le o .out do SLURM (stdout de scripts/04_train_rcae.py) e extrai a
TENDENCIA do "cross-dir std" (quanto a predicao varia entre
direcoes-alvo, comparado com o quanto o target varia de verdade) ao longo
do treino -- pra responder objetivamente "isso esta melhorando, so
devagar, ou estagnou?" sem precisar comparar PNGs a olho, um a um.

Le as linhas que utils/viz.py:save_patch_debug_png imprime a cada
snapshot de debug, no formato:
    [debug] cross-dir std -- input=0.0375 target=0.0327 pred=0.0085 (pred/target=0.260) em .../step_002600_epoch0001_batch002600.png

Separa em duas series (eixos X diferentes, nao dá pra misturar num so
grafico direto):
  - "por batch" (--debug-plot-every-batches): nome de arquivo
    step_NNNNNN_epochEEEE_batchBBBB.png -- eixo X = step (contador GLOBAL
    de batches de treino, continuo entre epocas). Patch/sujeito/direcoes
    mudam a cada ponto (e reamostrado a cada exemplo de treino, ver
    utils/dataset.py:_dynamic_split) -- serie mais "ruidosa".
  - "patch fixo por epoca" (--debug-plot-every): nome de arquivo
    epoch_EEEE.png -- eixo X = epoca (1 ponto por epoca marcada), SEMPRE
    o mesmo patch/direcoes de validacao -- serie mais limpa/comparavel
    pra ver tendencia real sem misturar "mudou o patch" com "o modelo
    aprendeu mais".

Uso:
    python scripts/plot_debug_trend.py --log logs/train.12345_0.out \
        --out work_dir/rcae_checkpoints/shell1000_n10/debug_trend.png

    # varios .out do mesmo run (ex.: job reiniciado, --requeue, etc.):
    python scripts/plot_debug_trend.py --log logs/train.12345_0.out logs/train.12346_0.out \
        --out debug_trend.png --csv debug_trend.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"\[debug\] cross-dir std -- input=([\d.eE+-]+) target=([\d.eE+-]+) "
    r"pred=([\d.eE+-]+) \(pred/target=([\d.eE+-]+)\) em (.+?)\s*$"
)
STEP_RE = re.compile(r"step_(\d+)_epoch(\d+)_batch(\d+)\.png$")
EPOCH_RE = re.compile(r"epoch_(\d+)\.png$")


def parse_logs(log_paths):
    """Le um ou mais .out e devolve (per_batch, per_epoch), cada um uma
    lista de tuplas ja ORDENADA pelo eixo X relevante (step ou epoca)."""
    per_batch = []   # (step, epoch, batch, input_std, target_std, pred_std, ratio)
    per_epoch = []   # (epoch, input_std, target_std, pred_std, ratio)
    for log_path in log_paths:
        text = Path(log_path).read_text(errors="replace")
        for line in text.splitlines():
            m = LINE_RE.search(line)
            if not m:
                continue
            in_std, tg_std, pr_std, ratio, path_str = m.groups()
            in_std, tg_std, pr_std, ratio = float(in_std), float(tg_std), float(pr_std), float(ratio)
            fname = Path(path_str).name
            m_step = STEP_RE.search(fname)
            if m_step:
                step, epoch, batch = (int(x) for x in m_step.groups())
                per_batch.append((step, epoch, batch, in_std, tg_std, pr_std, ratio))
                continue
            m_epoch = EPOCH_RE.search(fname)
            if m_epoch:
                epoch = int(m_epoch.group(1))
                per_epoch.append((epoch, in_std, tg_std, pr_std, ratio))
            # nomes de arquivo que nao batem nenhum dos dois padroes (ex.:
            # renomeado a mao) sao ignorados silenciosamente -- so afeta a
            # linha do stdout que ja tem os numeros, entao nao perde dado
            # nenhum que nao seja so "de qual serie/eixo X ele e".
    per_batch.sort(key=lambda r: r[0])
    per_epoch.sort(key=lambda r: r[0])
    return per_batch, per_epoch


def _plot_series(ax, x, target_std, pred_std, ratio, xlabel, title, marker=None):
    ax.plot(x, target_std, label="target std entre dir", color="tab:blue", alpha=0.6, marker=marker)
    ax.plot(x, pred_std, label="pred std entre dir", color="tab:red", marker=marker)
    ax2 = ax.twinx()
    ax2.plot(x, ratio, label="pred/target (razao)", color="tab:green", linestyle="--",
              alpha=0.8, marker=marker)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("pred/target (razao)", color="tab:green")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("std entre direcoes")
    ax.set_title(title, fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs="+", required=True,
                     help="um ou mais arquivos .out do SLURM (stdout do 04_train_rcae.py)")
    ap.add_argument("--out", default="debug_trend.png",
                     help="PNG de saida (default: debug_trend.png)")
    ap.add_argument("--csv", default=None,
                     help="opcional: tambem salva os dados extraidos como CSV nesse caminho "
                          "(util se quiser plotar/filtrar do seu jeito depois)")
    args = ap.parse_args()

    per_batch, per_epoch = parse_logs(args.log)
    if not per_batch and not per_epoch:
        print("Nenhuma linha '[debug] cross-dir std' encontrada nos logs informados -- "
              "confira se o job rodou com --debug-plot-every e/ou "
              "--debug-plot-every-batches > 0 (sem isso o script de treino nao gera "
              "snapshot nenhum, entao nao ha nada pra extrair).", file=sys.stderr)
        sys.exit(1)

    print(f"{len(per_batch)} snapshots 'por batch' e {len(per_epoch)} snapshots "
          f"'patch fixo por epoca' encontrados em {len(args.log)} arquivo(s) de log.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fonte", "step", "epoch", "batch", "input_std", "target_std",
                        "pred_std", "pred_target_ratio"])
            for step, epoch, batch, in_std, tg_std, pr_std, ratio in per_batch:
                w.writerow(["por_batch", step, epoch, batch, in_std, tg_std, pr_std, ratio])
            for epoch, in_std, tg_std, pr_std, ratio in per_epoch:
                w.writerow(["patch_fixo_epoca", "", epoch, "", in_std, tg_std, pr_std, ratio])
        print(f"CSV salvo em {args.csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = int(bool(per_batch)) + int(bool(per_epoch))
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels), squeeze=False)
    axes = axes[:, 0]
    ax_i = 0

    if per_batch:
        steps = [r[0] for r in per_batch]
        tg = [r[4] for r in per_batch]
        pr = [r[5] for r in per_batch]
        ratio = [r[6] for r in per_batch]
        _plot_series(axes[ax_i], steps, tg, pr, ratio,
                     xlabel="step (contador global de batches de treino)",
                     title="Snapshots por batch (--debug-plot-every-batches) -- "
                           "patch/sujeito/direcoes mudam a cada ponto (serie ruidosa)")
        ax_i += 1

    if per_epoch:
        epochs = [r[0] for r in per_epoch]
        tg = [r[2] for r in per_epoch]
        pr = [r[3] for r in per_epoch]
        ratio = [r[4] for r in per_epoch]
        _plot_series(axes[ax_i], epochs, tg, pr, ratio,
                     xlabel="epoca", marker="o",
                     title="Snapshots do patch FIXO de validacao (--debug-plot-every) -- "
                           "mesmo patch/direcoes toda vez (serie limpa/comparavel)")
        ax_i += 1

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Grafico salvo em {args.out}")


if __name__ == "__main__":
    main()