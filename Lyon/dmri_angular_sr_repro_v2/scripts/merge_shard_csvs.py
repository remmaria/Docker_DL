#!/usr/bin/env python3
"""
Junta os CSVs de shards gerados por 06_evaluate_reconstruction.py /
07_downstream_dti_noddi.py quando rodados com --shard-count>1 (paralelizando
por SUJEITO dentro do mesmo combo shell/n_level via SLURM array -- ver
slurm/05_evaluate_and_downstream.sh).

Cada shard escreve um arquivo "<nome>.shardIofN.csv" (I=indice do shard,
N=total de shards) na mesma pasta. Este script encontra todos os shards de
um mesmo "<nome>.csv" base, concatena, e grava o CSV final sem o sufixo de
shard -- pronto pra usar em 09_aggregate_and_plot.py como se tivesse sido
gerado num job so.

Uso:
    # um arquivo base especifico (recomendado -- evita ambiguidade se tiver
    # varios combos shell/n_level com shards na mesma pasta):
    python scripts/merge_shard_csvs.py \
        work_dir/metrics/signal_metrics_shell1000_n10.csv

    # ou aponte pra pasta inteira: junta TODOS os grupos de shard encontrados
    python scripts/merge_shard_csvs.py --dir work_dir/metrics

Depois de juntar com sucesso (todos os N shards esperados encontrados), os
arquivos .shardIofN.csv originais NAO sao apagados automaticamente -- confira
o CSV final e apague-os manualmente (ou rode com --delete-shards) se quiser
liberar espaco.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SHARD_RE = re.compile(r"^(?P<base>.+)\.shard(?P<idx>\d+)of(?P<count>\d+)\.csv$")


def find_shard_groups(directory: Path):
    """Agrupa arquivos *.shardIofN.csv na pasta por nome-base, devolvendo
    {base_csv_path: {idx: path}}."""
    groups = {}
    for f in directory.glob("*.shard*of*.csv"):
        m = SHARD_RE.match(f.name)
        if not m:
            continue
        base = directory / f"{m.group('base')}.csv"
        idx, count = int(m.group("idx")), int(m.group("count"))
        groups.setdefault((base, count), {})[idx] = f
    return groups


def merge_one(base_csv: Path, count: int, shard_paths: dict, delete_shards: bool):
    missing = [i for i in range(count) if i not in shard_paths]
    if missing:
        print(f"[aviso] {base_csv.name}: faltam os shards {missing} de {count} -- "
              f"pulando (junte de novo quando todos tiverem terminado)", file=sys.stderr)
        return False
    dfs = [pd.read_csv(shard_paths[i]) for i in range(count)]
    df = pd.concat(dfs, ignore_index=True)
    base_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_csv, index=False)
    print(f"{base_csv}: {count} shards -> {len(df)} linhas juntadas")
    if delete_shards:
        for p in shard_paths.values():
            p.unlink()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_csv", nargs="?", default=None,
                     help="caminho do CSV final desejado (sem sufixo de shard), ex.: "
                          "work_dir/metrics/signal_metrics_shell1000_n10.csv -- procura "
                          "'<mesmo nome>.shardIofN.csv' na mesma pasta.")
    ap.add_argument("--dir", default=None,
                     help="em vez de um arquivo especifico, procura TODOS os grupos de "
                          "shard nesta pasta e junta cada um.")
    ap.add_argument("--delete-shards", action="store_true",
                     help="apaga os arquivos .shardIofN.csv depois de juntar com sucesso.")
    args = ap.parse_args()

    if args.base_csv is None and args.dir is None:
        sys.exit("Passe um base_csv especifico ou --dir")

    if args.base_csv is not None:
        base_csv = Path(args.base_csv)
        groups = find_shard_groups(base_csv.parent)
        matches = {(b, c): paths for (b, c), paths in groups.items() if b == base_csv}
        if not matches:
            sys.exit(f"Nenhum shard '*.shardIofN.csv' encontrado para {base_csv.name} "
                      f"em {base_csv.parent}")
    else:
        matches = find_shard_groups(Path(args.dir))
        if not matches:
            sys.exit(f"Nenhum grupo de shard encontrado em {args.dir}")

    any_fail = False
    for (base, count), paths in matches.items():
        ok = merge_one(base, count, paths, args.delete_shards)
        any_fail = any_fail or not ok
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()