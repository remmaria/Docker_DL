#!/usr/bin/env python3
"""
Etapa 9c (auxiliar de 09b_build_basecurve.py): junta os CSVs por-shard
gerados por `scripts/09b_build_basecurve.py --shard-index i --shard-count N`
(rodados em paralelo via `sbatch --array`, ver slurm/09b_build_basecurve.sh)
num CSV final unico + a tabela resumida por n_level.

So faz sentido rodar isso DEPOIS que todas as N shards ja tiverem
terminado (confira nos logs/basecurve.*.out de cada task do array que
todas imprimiram "Metricas salvas em ...shard<i>.csv" com sucesso) --
se faltar algum shard, o merge roda mesmo assim mas o resultado fica
incompleto silenciosamente do ponto de vista deste script (ele so avisa
quantos arquivos de shard encontrou, nao sabe quantos DEVERIAM existir --
por isso o --shard-count abaixo e usado pra conferir isso).

Uso:
    python scripts/09c_merge_basecurve.py \
        --out-csv work_dir/basecurve_metrics_shell1000.csv \
        --shard-count 8
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Copia identica da funcao em scripts/09b_build_basecurve.py -- nao
    importada porque o nome do arquivo original comeca com digito e nao e
    importavel como modulo normal."""
    agg = df[df["metric_scope"] == "aggregate"]
    pv = df[df["metric_scope"] == "per_volume"]

    summary_agg = agg.groupby("n_level")[["nmse", "rmse", "acc_mean"]].agg(["mean", "std", "count"])
    summary_pv = pv.groupby("n_level")[["psnr", "ssim"]].agg(["mean", "std"])
    summary = summary_agg.join(summary_pv)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", required=True,
                     help="mesmo --out-csv passado a scripts/09b_build_basecurve.py "
                          "(o script procura os shards <out-csv-stem>.shard*.csv ao lado dele)")
    ap.add_argument("--shard-count", type=int, required=True,
                     help="numero de shards esperado, so pra conferencia (avisa se faltar algum)")
    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    shard_paths = sorted(
        out_csv.parent.glob(f"{out_csv.stem}.shard*{out_csv.suffix}"),
        key=lambda p: int(p.stem.split(".shard")[-1]),
    )

    if not shard_paths:
        sys.exit(f"Nenhum arquivo de shard encontrado (padrao {out_csv.stem}.shard*"
                  f"{out_csv.suffix} em {out_csv.parent}) -- confira se os jobs do array ja "
                  f"terminaram e se --out-csv bate com o que foi passado a 09b_build_basecurve.py")

    if len(shard_paths) != args.shard_count:
        print(f"[aviso] esperava {args.shard_count} shard(s), encontrei {len(shard_paths)}: "
              f"{[p.name for p in shard_paths]} -- confira se algum job do array ainda nao "
              f"terminou ou falhou antes de confiar no resultado final.")

    dfs = [pd.read_csv(p) for p in shard_paths]
    n_rows_each = [len(d) for d in dfs]
    df = pd.concat(dfs, ignore_index=True)

    n_subjects = df.loc[df["metric_scope"] == "aggregate", "subject"].nunique()
    print(f"Juntando {len(shard_paths)} shard(s) ({n_rows_each} linhas cada) -- "
          f"{len(df)} linhas totais, {n_subjects} sujeito(s) unicos.")

    dup = df.duplicated(subset=["subject", "n_level", "target_volume_idx", "metric_scope"])
    if dup.any():
        print(f"[aviso] {dup.sum()} linha(s) duplicada(s) entre shards (mesmo sujeito/nivel/"
              f"volume processado em mais de uma shard) -- removendo duplicatas, mas confira "
              f"se o --shard-index/--shard-count usado em cada task bateu com o esperado.")
        df = df[~dup]

    df.to_csv(out_csv, index=False)
    print(f"CSV final salvo em {out_csv}")

    summary_df = compute_summary(df)
    summary_csv = out_csv.with_name(f"{out_csv.stem}_summary{out_csv.suffix}")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Tabela resumida (uma linha por n_level) salva em {summary_csv}")

    print("\nCurva (media entre sujeitos, amostra fixa) -- erro do baseline_sh vs. n_level:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()