#!/usr/bin/env python3
"""
Compara as metricas downstream (etapa 7) de duas variantes de treino do
RCAE para o MESMO (shell_b, n_level) -- tipicamente uma sem loss angular
(--angular-loss-weight 0.0) e outra com (--angular-loss-weight > 0), geradas
por slurm/06_compare_sh_loss.sh em pastas separadas
($WORK_DIR/downstream_semSH e $WORK_DIR/downstream_comSH).

So compara o metodo 'rcae' (o baseline_sh, se presente nos CSVs, e o mesmo
nos dois arquivos -- nao reconstruido de novo -- entao nao faz sentido
comparar baseline_sh vs baseline_sh aqui).

Junta os dois CSVs por (subject, roi) e roda um teste pareado (Wilcoxon
signed-rank, ver protocolo secao 5) em cada metrica *_mae -- valor menor de
MAE e melhor (mais perto do ground truth). Reporta tambem quantos sujeitos
melhoraram vs pioraram (sinal da diferenca), nao so o p-valor.

Uso:
    python scripts/compare_sh_loss_variants.py \
        --sem-sh work_dir/downstream_semSH/dti_noddi_metrics_shell1000_n10.csv \
        --com-sh work_dir/downstream_comSH/dti_noddi_metrics_shell1000_n10.csv \
        --roi whole_mask
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sem-sh", required=True, help="CSV do downstream sem loss angular")
    ap.add_argument("--com-sh", required=True, help="CSV do downstream com loss angular")
    ap.add_argument("--roi", default="whole_mask",
                     help="restringe a comparacao a essa ROI (default whole_mask -- "
                          "use o nome de um trato JHU-ICBM, ex. FX, se tiver rodado "
                          "com ROI_TRACTS na etapa 7)")
    ap.add_argument("--method", default="rcae",
                     help="metodo a comparar entre os dois CSVs (default 'rcae' -- "
                          "'baseline_sh' nao muda entre os dois runs, comparar ele "
                          "contra ele mesmo nao informa nada)")
    args = ap.parse_args()

    df_sem = pd.read_csv(args.sem_sh)
    df_com = pd.read_csv(args.com_sh)

    df_sem = df_sem[(df_sem["method"] == args.method) & (df_sem["roi"] == args.roi)]
    df_com = df_com[(df_com["method"] == args.method) & (df_com["roi"] == args.roi)]

    metric_cols = [c for c in df_sem.columns if c.endswith("_mae") or c.endswith("_corr")]

    merged = df_sem.merge(df_com, on="subject", suffixes=("_semSH", "_comSH"))
    n = len(merged)
    print(f"ROI={args.roi} | metodo={args.method} | sujeitos pareados: {n} "
          f"(sem_sh tinha {len(df_sem)}, com_sh tinha {len(df_com)})")
    if n == 0:
        print("Nenhum sujeito em comum -- confira se os dois CSVs sao do mesmo "
              "shell_b/n_level/split.")
        return

    print(f"\n{'metrica':<12} {'sem_SH (med)':>14} {'com_SH (med)':>14} "
          f"{'delta (com-sem)':>16} {'melhorou/total':>16} {'wilcoxon p':>12}")
    for col in metric_cols:
        a = merged[f"{col}_semSH"].to_numpy()
        b = merged[f"{col}_comSH"].to_numpy()
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]
        if len(a) < 2:
            continue
        diff = b - a  # negativo = com_SH melhor (MAE menor) / positivo = melhor pra _corr
        lower_is_better = col.endswith("_mae")
        if lower_is_better:
            n_better = int((diff < 0).sum())
        else:
            n_better = int((diff > 0).sum())
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = float("nan")  # ex.: todas as diferencas sao zero
        print(f"{col:<12} {np.median(a):>14.4f} {np.median(b):>14.4f} "
              f"{np.median(diff):>16.4f} {n_better:>8d}/{len(a):<7d} {p:>12.4f}")

    print("\nLeitura: pra colunas *_mae, delta NEGATIVO e p<0.05 = com_SH reduziu o "
          "erro de forma estatisticamente significativa (melhorou). Pra colunas "
          "*_corr, delta POSITIVO e melhor (correlacao mais alta com o ground truth). "
          "'melhorou/total' e o placar bruto por sujeito, independente de p-valor -- "
          "vale olhar junto, nao so o p (com poucos sujeitos o teste tem pouco poder).")


if __name__ == "__main__":
    main()