#!/usr/bin/env python3
"""
Etapa 1b (QC, rodar depois da etapa 1): para cada b-value candidato,
reporta quantos sujeitos tem essa shell disponivel -- separando quem tem
ela como protocolo "single-shell nativo" de quem tem ela dentro de uma
aquisicao multi-shell (a shell pode ser extraida e tratada como um
experimento de b-value unico do mesmo jeito) -- e a distribuicao de
numero de direcoes e de b0s nesses sujeitos.

Isso existe porque, no seu caso, os protocolos sao bem heterogeneos: single
e multi-shell variam entre si em b-value (500/700/750/1000/1500/2000, nem
todo sujeito tem todos), numero de direcoes e numero de b0s. Antes de
decidir quais experimentos rodar (baseline + RCAE por b-value), vale ver
onde da pra formar um grupo com N razoavel.

Uso:
    python scripts/01b_shell_availability_report.py \
        --manifest work_dir/manifest.csv \
        --candidate-bvalues 500 700 750 1000 1500 2000 \
        --shell-match-tol 25 \
        --out-csv work_dir/shell_availability.csv

Saida: um CSV com uma linha por (b_value_candidato, sujeito) que possui
aquela shell, mais um resumo impresso no console por b_value.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--candidate-bvalues", type=float, nargs="+",
                     default=[500, 700, 750, 1000, 1500, 2000])
    ap.add_argument("--shell-match-tol", type=float, default=25.0,
                     help="tolerancia (s/mm^2) para casar um b-value candidato com a shell "
                          "detectada no sujeito -- mantenha isso menor que a menor diferenca "
                          "real entre protocolos (ex.: 700 vs 750 => tol < 25)")
    ap.add_argument("--min-n-directions", type=int, default=6,
                     help="ignora shells com menos direcoes que isso (nao da pra fazer nem "
                          "o baseline SH de ordem minima)")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    entries = load_manifest(args.manifest)

    rows = []
    for e in entries:
        for cand in args.candidate_bvalues:
            if not e.has_shell(cand, tol=args.shell_match_tol):
                continue
            n_dirs = e.n_dirs_for_shell(cand, tol=args.shell_match_tol)
            if n_dirs is None or n_dirs < args.min_n_directions:
                continue
            rows.append({
                "subject": e.subject,
                "split": e.split,
                "b_value_candidate": cand,
                "n_directions": n_dirs,
                "n_b0": e.n_b0,
                "n_shells_total": e.n_shells,
                "acquisition_context": "from_multishell" if e.is_multishell else "native_single_shell",
            })

    if not rows:
        sys.exit("Nenhuma shell encontrada para os b-values candidatos -- confira "
                  "--shell-match-tol e --candidate-bvalues")

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"Detalhe salvo em {args.out_csv}\n")
    print("Resumo por b-value candidato:\n")
    summary_rows = []
    for cand, group in df.groupby("b_value_candidate"):
        n_total = group["subject"].nunique()
        n_native = group[group.acquisition_context == "native_single_shell"]["subject"].nunique()
        n_multi = group[group.acquisition_context == "from_multishell"]["subject"].nunique()
        n_train = group[group.split == "train"]["subject"].nunique()
        n_val = group[group.split == "val"]["subject"].nunique()
        n_test = group[group.split == "test"]["subject"].nunique()
        med_dirs = group["n_directions"].median()
        min_dirs = group["n_directions"].min()
        max_dirs = group["n_directions"].max()
        med_b0 = group["n_b0"].median()
        summary_rows.append({
            "b_value": cand, "n_subjects": n_total, "n_native_single": n_native,
            "n_from_multishell": n_multi, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "n_directions_median": med_dirs, "n_directions_min": min_dirs,
            "n_directions_max": max_dirs, "n_b0_median": med_b0,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("n_subjects", ascending=False)
    print(summary_df.to_string(index=False))

    summary_csv = str(Path(args.out_csv).with_name(Path(args.out_csv).stem + "_summary.csv"))
    summary_df.to_csv(summary_csv, index=False)
    print("\nResumo salvo em", summary_csv)

    weak = summary_df[summary_df.n_subjects < 20]
    if not weak.empty:
        print("\n[aviso] b-values com poucos sujeitos (<20) -- considere descartar "
              "ou tratar so como analise exploratoria, nao como experimento principal:")
        print(weak[["b_value", "n_subjects"]].to_string(index=False))

    mixed = summary_df[(summary_df.n_native_single > 0) & (summary_df.n_from_multishell > 0)]
    if not mixed.empty:
        print("\n[nota] os b-values abaixo misturam sujeitos nativamente single-shell "
              "com shells extraidas de multi-shell -- o pipeline pode pooling os dois "
              "(ambos viram so 'uma shell + seus b0' na hora de treinar), mas convem "
              "reportar as metricas tambem separadas por acquisition_context, para "
              "checar se ha vies sistematico entre os dois tipos de aquisicao:")
        print(mixed[["b_value", "n_native_single", "n_from_multishell"]].to_string(index=False))


if __name__ == "__main__":
    main()
