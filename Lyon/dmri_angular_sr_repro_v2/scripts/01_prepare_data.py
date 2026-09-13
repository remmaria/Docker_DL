#!/usr/bin/env python3
"""
Etapa 1: descobre sujeitos numa arvore de diretorios (layout proprio, nao
precisa ser BIDS -- ex.: studies/<estudo>/<pasta_sessao>/<nome_base>_geomcorr.{nii,bval,bvec}),
registra a assinatura de shells de cada um (quais b-values, quantas direcoes
por shell, quantos b0 -- sem exigir um protocolo uniforme entre sujeitos) e
gera o split treino/val/teste GLOBAL por sujeito (reusado depois em todos os
experimentos por b-value, mesmo que um sujeito multi-shell participe de
varios experimentos diferentes -- ver utils/manifest.assign_splits).

A descoberta procura, recursivamente a partir de --data-root, qualquer
arquivo terminando em "<name-suffix>.bval" que tenha um ".bvec" e um
".nii"/".nii.gz" companheiros (mesmo nome, mesma pasta). Outros arquivos na
mesma pasta (mascaras, mapas de FA/MD, dados brutos sem bval/bvec) sao
ignorados automaticamente. O identificador do sujeito e derivado do caminho
("<estudo>__<pasta_sessao>"), sem exigir prefixo "sub-".

Demograficos/acquisicao (--demo-tsv-name, default "info.tsv"), DOIS MODOS:
  1) TSV GLOBAL UNICO (use isso se so existe um arquivo de demograficos pra
     tudo): passe o CAMINHO COMPLETO do arquivo, ex.:
     --demo-tsv-name /ix1/tibrahim/rmm270/DATA/DWIs/7TBRP/info.tsv
     (ou a env var DEMO_TSV_NAME=<caminho completo> no slurm/01_prepare_data.sh).
     Esse TSV e lido uma vez e casado por SessionID==pasta_sessao em TODOS
     os sujeitos, independente de <estudo>/subpasta -- inclusive quando
     `study` fica vazio (data-root ja aponta direto pra pasta que contem as
     sessoes, sem nivel de subpasta de estudo no meio).
  2) TSV POR SUBESTUDO (comportamento original): passe so um NOME de arquivo
     (ex.: "info.tsv", o default) e o script procura
     "<data-root>/<estudo>/<demo-tsv-name>" pra cada subestudo separadamente.
     So funciona se `study` (a subpasta logo abaixo de --data-root) for
     no'-vazia pra cada sujeito.
Em ambos os casos: colunas viram patient_sex/patient_age/acquisition_date/
scanner_protocol no manifest.csv (scanner_protocol vem da coluna "Study" do
TSV -- nao confundir com a coluna "study" do manifest, que e' o nome da
subpasta de estudo). Sujeitos sem SessionID casado ficam com esses 4 campos
vazios, sem erro -- e' opcional, nao um requisito de todo o dataset.

Uso:
    python scripts/01_prepare_data.py \
        --data-root /caminho/para/studies \
        --out-dir /caminho/para/work_dir \
        --name-suffix _geomcorr \
        --train-frac 0.7 --val-frac 0.15 --seed 42

Saida: <out-dir>/manifest.csv

Depois de rodar isso, rode scripts/01b_shell_availability_report.py para
ver quantos sujeitos tem cada b-value de interesse (nativo ou extraido de
multi-shell) antes de decidir quais experimentos valem a pena.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import build_manifest, assign_splits, save_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True,
                     help="raiz da arvore de dados (ex.: .../DATA/DWIs/studies)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name-suffix", default="_geomcorr",
                     help="sufixo (antes da extensao) que identifica o dwi pre-processado "
                          "final, ex.: 'bgpdwis_PA_geomcorr.nii' -> sufixo '_geomcorr'")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shell-tol", type=float, default=100.0,
                     help="tolerancia (s/mm^2) para agrupar bvals na mesma shell")
    ap.add_argument("--demo-tsv-name", default="info.tsv",
                     help="nome do TSV opcional de demograficos/acquisicao dentro de cada "
                          "subpasta de estudo (<data-root>/<estudo>/<este-nome>) -- ver "
                          "docstring do modulo para as colunas esperadas")
    ap.add_argument("--no-stratify-sex", action="store_true",
                     help="desliga a estratificacao do split treino/val/teste por sexo "
                          "(default: estratifica por protocol+sexo+estudo, ver "
                          "utils.manifest.assign_splits)")
    ap.add_argument("--no-stratify-study", action="store_true",
                     help="desliga a estratificacao do split por estudo/protocolo de scanner "
                          "de origem (mantem so protocol+sexo, se --no-stratify-sex tambem nao "
                          "for passado)")
    ap.add_argument("--min-stratum-size", type=int, default=8,
                     help="grupos (protocol,sexo,estudo) menores que isso recaem pra uma "
                          "estratificacao mais grosseira -- ver docstring de assign_splits")
    args = ap.parse_args()

    entries = build_manifest(args.data_root, tol=args.shell_tol, name_suffix=args.name_suffix,
                              demo_tsv_name=args.demo_tsv_name)
    if not entries:
        print(f"Nenhum trio nii+bval+bvec terminando em '{args.name_suffix}' "
              f"encontrado em {args.data_root}")
        sys.exit(1)

    entries = assign_splits(entries, train=args.train_frac, val=args.val_frac, seed=args.seed,
                             stratify_sex=not args.no_stratify_sex,
                             stratify_study=not args.no_stratify_study,
                             min_stratum_size=args.min_stratum_size)

    out_csv = str(Path(args.out_dir) / "manifest.csv")
    save_manifest(entries, out_csv)

    n_single = sum(1 for e in entries if e.protocol == "single_shell")
    n_multi = sum(1 for e in entries if e.protocol == "multi_shell")
    n_studies = len({e.study for e in entries if e.study})
    print(f"Sujeitos encontrados: {len(entries)} em {n_studies} subestudo(s) "
          f"(single-shell: {n_single}, multi-shell: {n_multi})")
    split_counts = {}
    for split in ("train", "val", "test"):
        n = sum(1 for e in entries if e.split == split)
        split_counts[split] = n
        print(f"  {split}: {n}")

    # Composicao do split por sexo e por estudo, pra conferir visualmente que a
    # estratificacao (ver utils.manifest.assign_splits) manteve as proporcoes
    # parecidas entre treino/val/teste -- nao so o tamanho total de cada um.
    sexes_present = sorted({(e.patient_sex.strip().upper() or "UNK") for e in entries})
    if any(s in ("M", "F") for s in sexes_present):
        print("\nComposicao por sexo (%, dentro de cada split):")
        for split in ("train", "val", "test"):
            split_entries = [e for e in entries if e.split == split]
            n_split = len(split_entries)
            if n_split == 0:
                continue
            counts = {s: 0 for s in sexes_present}
            for e in split_entries:
                counts[e.patient_sex.strip().upper() or "UNK"] += 1
            pct_str = ", ".join(f"{s}={100*c/n_split:.1f}%" for s, c in counts.items())
            print(f"  {split} (n={n_split}): {pct_str}")

    studies_present = sorted({e.study for e in entries if e.study})
    if studies_present:
        print(f"\nComposicao por estudo (top 5 maiores, %, dentro de cada split; "
              f"{len(studies_present)} estudo(s) distinto(s) no total):")
        top_studies = sorted(studies_present,
                              key=lambda s: -sum(1 for e in entries if e.study == s))[:5]
        for split in ("train", "val", "test"):
            split_entries = [e for e in entries if e.split == split]
            n_split = len(split_entries)
            if n_split == 0:
                continue
            pct_str = ", ".join(
                f"{s}={100 * sum(1 for e in split_entries if e.study == s) / n_split:.1f}%"
                for s in top_studies)
            print(f"  {split} (n={n_split}): {pct_str}")

    # Resumo demografico (so' entra se algum sujeito tiver os campos preenchidos --
    # ver utils.manifest.load_study_info/build_manifest).
    summary_rows = [
        ("subjects_total", len(entries)),
        ("subestudos_distintos", n_studies),
        ("single_shell", n_single),
        ("multi_shell", n_multi),
        ("split_train", split_counts["train"]),
        ("split_val", split_counts["val"]),
        ("split_test", split_counts["test"]),
    ]

    with_demo = [e for e in entries if e.patient_sex or e.patient_age or e.scanner_protocol]
    n_m = n_f = n_sex_other = 0
    ages = []
    protocols = []
    if with_demo:
        n_m = sum(1 for e in with_demo if e.patient_sex.strip().upper() == "M")
        n_f = sum(1 for e in with_demo if e.patient_sex.strip().upper() == "F")
        n_sex_other = len(with_demo) - n_m - n_f
        for e in with_demo:
            try:
                ages.append(float(e.patient_age))
            except (TypeError, ValueError):
                pass
        protocols = sorted({e.scanner_protocol for e in with_demo if e.scanner_protocol})
        print(f"\nDemograficos: {len(with_demo)}/{len(entries)} sujeitos com info de "
              f"'{args.demo_tsv_name}' (M: {n_m}, F: {n_f}, outro/vazio: {n_sex_other})")
        if ages:
            print(f"  idade (anos): min={min(ages):.1f} mediana={sorted(ages)[len(ages)//2]:.1f} "
                  f"max={max(ages):.1f} (n={len(ages)}/{len(with_demo)} com idade parseavel)")
        if protocols:
            print(f"  scanner_protocol (coluna 'Study' do TSV) distintos: {len(protocols)} "
                  f"-> {', '.join(protocols[:8])}{' ...' if len(protocols) > 8 else ''}")
    else:
        print(f"\nDemograficos: nenhum sujeito casado via '{args.demo_tsv_name}' "
              f"(0/{len(entries)}) -- confira se --demo-tsv-name aponta certo (caminho "
              f"completo pra TSV global, ou nome de arquivo dentro de cada subpasta de "
              f"estudo) e se SessionID no TSV bate exatamente com o nome da pasta de sessao")

    summary_rows += [
        ("demo_tsv_name", args.demo_tsv_name),
        ("demo_subjects_matched", len(with_demo)),
        ("demo_sex_M", n_m),
        ("demo_sex_F", n_f),
        ("demo_sex_outro_vazio", n_sex_other),
        ("demo_age_min", f"{min(ages):.2f}" if ages else ""),
        ("demo_age_mediana", f"{sorted(ages)[len(ages)//2]:.2f}" if ages else ""),
        ("demo_age_max", f"{max(ages):.2f}" if ages else ""),
        ("demo_age_n_parseavel", len(ages)),
        ("demo_scanner_protocols_distintos", len(protocols)),
        ("demo_scanner_protocols", "|".join(protocols)),
    ]

    summary_csv = str(Path(args.out_dir) / "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    # Resumo por estudo (1 linha por valor distinto de `study` -- que agora
    # vem da subpasta em disco quando existe, ou senao da coluna "Study" do
    # TSV de demograficos, ver utils.manifest.build_manifest) com contagem
    # de sujeitos por split. Sujeitos sem `study` nenhum (nem pasta nem TSV)
    # caem em uma linha "(sem study)".
    by_study: dict[str, dict[str, int]] = {}
    for e in entries:
        key = e.study or "(sem study)"
        d = by_study.setdefault(key, {"train": 0, "val": 0, "test": 0, "outro": 0})
        d[e.split if e.split in ("train", "val", "test") else "outro"] += 1

    study_summary_csv = str(Path(args.out_dir) / "study_summary.csv")
    with open(study_summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study", "n_subjects", "n_train", "n_val", "n_test"])
        for study in sorted(by_study):
            d = by_study[study]
            n_total = d["train"] + d["val"] + d["test"] + d["outro"]
            writer.writerow([study, n_total, d["train"], d["val"], d["test"]])

    print("Manifesto salvo em:", out_csv)
    print("Resumo salvo em:", summary_csv)
    print("Resumo por estudo salvo em:", study_summary_csv)
    print("\nProxima etapa recomendada: "
          "python scripts/01b_shell_availability_report.py --manifest", out_csv)


if __name__ == "__main__":
    main()