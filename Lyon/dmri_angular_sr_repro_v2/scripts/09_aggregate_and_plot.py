#!/usr/bin/env python3
"""
Etapa 9: agrega os CSVs de metricas gerados pelas etapas 06/07/08 (varios
niveis de subamostragem, possivelmente varias shells) em tabelas resumo e
figuras para a tese: qualidade de reconstrucao vs. numero de direcoes de
entrada, comparando baseline SH vs. RCAE, e o mesmo para as metricas de
microestrutura (DTI/NODDI) e tratografia quando disponiveis.

Uso:
    python scripts/09_aggregate_and_plot.py \
        --metrics-dir work_dir/metrics \
        --downstream-dir work_dir/downstream \
        --tractography-dir work_dir/tractography \
        --out-dir work_dir/figures

Esperado: os CSVs seguem o padrao de nome gerado pelos scripts 06/07/08
(signal_metrics_shell*_n*.csv, dti_noddi_metrics_shell*_n*.csv,
tractography_metrics_shell*_n*.csv). Basta apontar os diretorios; o script
concatena tudo que encontrar.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.metrics import paired_wilcoxon

METHOD_COLORS = {"baseline_sh": "#7a7a7a", "rcae": "#2f6db3"}
METHOD_LABELS = {"baseline_sh": "Baseline (SH)", "rcae": "RCAE"}


def load_concat(pattern_dir: str | None, glob_pattern: str) -> pd.DataFrame:
    if pattern_dir is None:
        return pd.DataFrame()
    files = sorted(Path(pattern_dir).glob(glob_pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def plot_metric_vs_level(df: pd.DataFrame, metric: str, out_path: Path, title: str,
                          ylabel: str, higher_is_better: bool = True):
    if df.empty or metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        grouped = sub.groupby("n_level")[metric].agg(["mean", "std"])
        grouped = grouped.sort_index()
        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker="o",
                    capsize=3, label=label, color=color)
    ax.set_xlabel("Numero de direcoes de entrada")
    ax.set_ylabel(ylabel)
    ax.set_title(title + ("  (maior = melhor)" if higher_is_better else "  (menor = melhor)"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Figura salva em", out_path)


def summarize_and_test(df: pd.DataFrame, metric: str, group_cols=("shell", "n_level")):
    """Tabela resumo (media/desvio por metodo/nivel) + teste de Wilcoxon
    pareado baseline vs rcae por sujeito, quando ambos os metodos existem
    para o mesmo (shell, n_level, subject).
    """
    if df.empty or metric not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    summary = df.groupby(["method", *group_cols])[metric].agg(["mean", "std", "count"]).reset_index()

    tests = []
    for (shell, n_level), group in df.groupby(list(group_cols)):
        pivot = group.pivot_table(index="subject", columns="method", values=metric)
        if "baseline_sh" in pivot.columns and "rcae" in pivot.columns:
            paired = pivot[["baseline_sh", "rcae"]].dropna()
            if len(paired) >= 5:
                stat, p = paired_wilcoxon(paired["baseline_sh"].values, paired["rcae"].values)
                tests.append({"shell": shell, "n_level": n_level, "metric": metric,
                              "n_subjects": len(paired), "wilcoxon_stat": stat, "p_value": p})
    return summary, pd.DataFrame(tests)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-dir", default=None, help="saida da etapa 06 (metricas de sinal)")
    ap.add_argument("--downstream-dir", default=None, help="saida da etapa 07 (DTI/NODDI)")
    ap.add_argument("--tractography-dir", default=None, help="saida da etapa 08 (opcional)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    signal_df = load_concat(args.metrics_dir, "signal_metrics_*.csv")
    downstream_df = load_concat(args.downstream_dir, "dti_noddi_metrics_*.csv")
    tract_df = load_concat(args.tractography_dir, "tractography_metrics_*.csv")

    # --- metricas de sinal (agregadas por voxel: nmse, rmse, acc_mean) ---
    if not signal_df.empty:
        agg = signal_df[signal_df.metric_scope == "aggregate"]
        for metric, higher_better in (("nmse", False), ("rmse", False), ("acc_mean", True)):
            plot_metric_vs_level(agg, metric, out_dir / f"signal_{metric}_vs_ndirs.png",
                                  f"Reconstrucao de sinal: {metric}", metric, higher_better)
            summary, tests = summarize_and_test(agg, metric)
            if not summary.empty:
                summary.to_csv(tables_dir / f"signal_{metric}_summary.csv", index=False)
            if not tests.empty:
                tests.to_csv(tables_dir / f"signal_{metric}_wilcoxon.csv", index=False)

        # PSNR/SSIM sao por volume; agregamos por sujeito antes de plotar
        per_vol = signal_df[signal_df.metric_scope == "per_volume"]
        if not per_vol.empty:
            per_subj = per_vol.groupby(["subject", "method", "shell", "n_level"])[
                ["psnr", "ssim"]].mean().reset_index()
            for metric in ("psnr", "ssim"):
                plot_metric_vs_level(per_subj, metric, out_dir / f"signal_{metric}_vs_ndirs.png",
                                      f"Reconstrucao de sinal: {metric.upper()}", metric.upper(), True)
                summary, tests = summarize_and_test(per_subj, metric)
                if not summary.empty:
                    summary.to_csv(tables_dir / f"signal_{metric}_summary.csv", index=False)
                if not tests.empty:
                    tests.to_csv(tables_dir / f"signal_{metric}_wilcoxon.csv", index=False)

    # --- metricas downstream (DTI/NODDI, em erro absoluto medio) ---
    if not downstream_df.empty:
        mae_cols = [c for c in downstream_df.columns if c.endswith("_mae")]
        for metric in mae_cols:
            plot_metric_vs_level(downstream_df, metric, out_dir / f"downstream_{metric}_vs_ndirs.png",
                                  f"Erro downstream: {metric}", metric, higher_is_better=False)
            summary, tests = summarize_and_test(downstream_df, metric)
            if not summary.empty:
                summary.to_csv(tables_dir / f"downstream_{metric}_summary.csv", index=False)
            if not tests.empty:
                tests.to_csv(tables_dir / f"downstream_{metric}_wilcoxon.csv", index=False)

    # --- tratografia (opcional) ---
    if not tract_df.empty:
        plot_metric_vs_level(tract_df, "dice_streamline_density",
                              out_dir / "tractography_dice_vs_ndirs.png",
                              "Tratografia: Dice de densidade de streamlines",
                              "Dice", higher_is_better=True)
        summary, tests = summarize_and_test(tract_df, "dice_streamline_density")
        if not summary.empty:
            summary.to_csv(tables_dir / "tractography_dice_summary.csv", index=False)
        if not tests.empty:
            tests.to_csv(tables_dir / "tractography_dice_wilcoxon.csv", index=False)

    if signal_df.empty and downstream_df.empty and tract_df.empty:
        print("Nenhum CSV de metricas encontrado nos diretorios informados.")
    else:
        print("Agregacao concluida. Figuras em", out_dir, "| tabelas em", tables_dir)


if __name__ == "__main__":
    main()
