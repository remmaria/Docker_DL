"""
make_splits.py
==============
Gera os CSVs de treino / validação / teste para o pipeline VFI-Harmonization.

Estratégia
----------
- Nível de estratificação : (protocolo, era_coil)
- Era da coil             : 'pre2022' se data < 2022-05-09, 'post2022' caso contrário
- Proporção               : 70 / 15 / 15  (train / val / test)
- Garantia                : nenhum sujeito aparece em mais de um split

Uso
---
python make_splits.py \
    --master_csv /caminho/para/todos_sujeitos.csv \
    --out_dir    /caminho/para/COORDS_VFI_MASKED_NEW \
    --seed       42

O CSV master deve ter pelo menos as colunas:
    subject, dwi_path, center_x, center_y, center_z, protocol

A coluna 'SessionID' deve conter o nome da pasta no formato
    YYYYMMDDHHMMSS_XXXXXX-descricao
onde os 8 primeiros caracteres (YYYYMMDD) são a data do scan.
"""

import argparse
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────────────
# 1. UTILITÁRIOS
# ──────────────────────────────────────────────────────────────────────────────

COIL_CUTOFF = pd.Timestamp("2022-05-09")


def extract_date(subject_id: str) -> pd.Timestamp | None:
    """
    Extrai a data do nome do subject (primeiros 8 dígitos = YYYYMMDD).
    Exemplos aceitos:
        '20160914203805_160914-volunteer'  → 2016-09-14
        '20220510_sub001'                  → 2022-05-10
    """
    match = re.match(r"(\d{8})", str(subject_id))
    if match:
        try:
            return pd.to_datetime(match.group(1), format="%Y%m%d")
        except ValueError:
            return None
    return None


def assign_coil_era(subject_id: str) -> str:
    date = extract_date(subject_id)
    if date is None:
        warnings.warn(
            f"Não foi possível extrair data de '{subject_id}'. "
            "Será atribuído 'unknown_era'."
        )
        return "unknown_era"
    return "pre2022" if date < COIL_CUTOFF else "post2022"


# Insira este bloco antes de:  def assign_strata(row: pd.Series) -> str:

def filter_small_protocols(df: pd.DataFrame, min_subjects: int = 10) -> pd.DataFrame:
    """
    Remove sujeitos cujo protocolo tenha menos de `min_subjects` sujeitos únicos.
    Imprime um relatório dos protocolos descartados.
    """
    subjects_per_protocol = (
        df.groupby("protocol")["SessionID"]
        .nunique()
        .rename("n_subjects")
    )

    small = subjects_per_protocol[subjects_per_protocol < min_subjects]
    keep  = subjects_per_protocol[subjects_per_protocol >= min_subjects]

    if small.empty:
        print(f"✅  Todos os protocolos têm ≥ {min_subjects} sujeitos. Nenhum descartado.")
    else:
        print(f"\n⚠️  Protocolos descartados (< {min_subjects} sujeitos):")
        for proto, n in small.items():
            print(f"   → '{proto}': {n} sujeito(s)")
        print(f"   Protocolos mantidos: {list(keep.index)}\n")

    return df[df["protocol"].isin(keep.index)].copy()

def assign_strata(row: pd.Series) -> str:
    """Combina protocolo + era para formar o estrato."""
    return f"{row['protocol']}|{row['coil_era']}"

def load_outliers(outliers_txt: str) -> set[str]:
    """
    Carrega SessionIDs de um TXT/CSV.
    Aceita:
        - apenas SessionID
        - SessionID + Reason
        - separados por TAB ou múltiplos espaços
    """

    df_out = pd.read_csv(
        outliers_txt,
        sep=r"\s+|\t+",
        engine="python",
        usecols=[0],
        header=0,
    )

    df_out.columns = ["SessionID"]

    outliers = set(
        df_out["SessionID"]
        .astype(str)
        .str.strip()
    )

    print(f"⚠️  Outliers carregados: {len(outliers)}")

    return outliers    
# ──────────────────────────────────────────────────────────────────────────────
# 2. SPLIT ESTRATIFICADO POR SUBJECT (não por patch)
# ──────────────────────────────────────────────────────────────────────────────

def stratified_subject_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide os SUJEITOS (não patches) de forma estratificada.

    Retorna (df_train, df_val, df_test) com todos os patches
    de cada sujeito no split correto.
    """
    # Uma linha por sujeito, mantendo o estrato
    subjects = (
        df[["SessionID", "strata"]]
        .drop_duplicates(subset="SessionID")
        .reset_index(drop=True)
    )

    strata_counts = subjects["strata"].value_counts()
    small_strata  = strata_counts[strata_counts < 3].index.tolist()

    if small_strata:
        warnings.warn(
            f"Os seguintes estratos têm < 3 sujeitos e não podem ser "
            f"estratificados: {small_strata}. Serão alocados manualmente."
        )

    # Separa estratos pequenos (serão colocados no treino)
    mask_small = subjects["strata"].isin(small_strata)
    subjects_small  = subjects[mask_small].copy()
    subjects_normal = subjects[~mask_small].copy()

    # ── Primeira divisão: (train+val) vs test ────────────────────────────────
    train_val_subj, test_subj = train_test_split(
        subjects_normal,
        test_size=test_size,
        stratify=subjects_normal["strata"],
        random_state=seed,
    )

    # ── Segunda divisão: train vs val ────────────────────────────────────────
    # val_size relativo ao conjunto original → ajuste proporcional
    val_rel = val_size / (1.0 - test_size)
    train_subj, val_subj = train_test_split(
        train_val_subj,
        test_size=val_rel,
        stratify=train_val_subj["strata"],
        random_state=seed,
    )

    # Pequenos estratos → treino
    train_subj = pd.concat([train_subj, subjects_small], ignore_index=True)

    # ── Expande de volta para os patches ─────────────────────────────────────
    train_ids = set(train_subj["SessionID"])
    val_ids   = set(val_subj["SessionID"])
    test_ids  = set(test_subj["SessionID"])

    df_train = df[df["SessionID"].isin(train_ids)].copy()
    df_val   = df[df["SessionID"].isin(val_ids)].copy()
    df_test  = df[df["SessionID"].isin(test_ids)].copy()

    return df_train, df_val, df_test


# ──────────────────────────────────────────────────────────────────────────────
# 3. RELATÓRIO
# ──────────────────────────────────────────────────────────────────────────────

def print_report(
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """Imprime um resumo do split por estrato."""
    print("\n" + "=" * 72)
    print(f"{'SPLIT REPORT':^72}")
    print("=" * 72)

    # Contagem de sujeitos únicos por split e estrato
    def count_subjects(d: pd.DataFrame) -> pd.Series:
        return d.groupby("strata")["SessionID"].nunique()

    all_strata = sorted(df["strata"].unique())
    counts = {
        "Total" : count_subjects(df),
        "Train" : count_subjects(df_train),
        "Val"   : count_subjects(df_val),
        "Test"  : count_subjects(df_test),
    }
    summary = pd.DataFrame(counts, index=all_strata).fillna(0).astype(int)
    summary.index.name = "Strata (protocol|coil_era)"

    print(summary.to_string())
    print("-" * 72)

    totals = summary.sum()
    print(f"\n{'TOTAL SUBJECTS':30s}  {totals['Total']:>5}  "
          f"Train:{totals['Train']:>4}  Val:{totals['Val']:>4}  "
          f"Test:{totals['Test']:>4}")

    total_patches = len(df)
    print(f"{'TOTAL PATCHES':30s}  {total_patches:>5}  "
          f"Train:{len(df_train):>4}  Val:{len(df_val):>4}  "
          f"Test:{len(df_test):>4}")

    # Verificação de leakage
    train_ids = set(df_train["SessionID"])
    val_ids   = set(df_val["SessionID"])
    test_ids  = set(df_test["SessionID"])
    leaks = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if leaks:
        print(f"\n⚠️  LEAKAGE DETECTADO: {len(leaks)} sujeito(s) em múltiplos splits!")
        for s in sorted(leaks):
            print(f"   → {s}")
    else:
        print("\n✅  Nenhum leakage detectado. Todos os sujeitos estão em um único split.")

    # Distribuição de eras
    print("\nDistribuição de coil_era por split:")
    for name, d in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        era_counts = d.drop_duplicates("SessionID")["coil_era"].value_counts().to_dict()
        print(f"  {name:5s}: {era_counts}")

    print("=" * 72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gera splits estratificados para o dataset VFI-Harmonization."
    )
    parser.add_argument(
        "--master_csv",
        required=True,
        help="CSV com todos os sujeitos e patches (subject, dwi_path, "
             "center_x, center_y, center_z, protocol).",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Pasta onde os CSVs de split serão salvos.",
    )
    parser.add_argument(
        "--min_subjects_per_protocol", 
        type=int, 
        default=10,
        help="Protocolos com menos sujeitos que este valor são descartados."
    )
    parser.add_argument(
        "--outliers_txt",
        type=str,
        default=None,
        help="TXT/CSV contendo coluna SessionID com subjects a excluir.",
    )
    parser.add_argument("--val_size",  type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    # ── Leitura ────────────────────────────────────────────────────────────────
    df = pd.read_csv(args.master_csv)


    # ── Remove outliers ─────────────────────────────────────────────────────────
    if args.outliers_txt is not None:
        outliers = load_outliers(args.outliers_txt)

        n_before = df["SessionID"].nunique()

        # salva antes de remover
        removed_ids = sorted(
            set(df["SessionID"].astype(str)) & outliers
        )

        df = df[~df["SessionID"].astype(str).isin(outliers)].copy()

        n_after = df["SessionID"].nunique()

        print(
            f"⚠️  Subjects removidos por outlier: {n_before - n_after}",
            flush=True,
        )

        if removed_ids:
            print("\nSubjects descartados:")
            for sid in removed_ids:
                print(f"  - {sid}")
        else:
            print("\nNenhum SessionID do arquivo de outliers foi encontrado no dataset.")

    required_cols = {"SessionID", "dwi_path", "center_x", "center_y", "center_z", "protocol"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    df = filter_small_protocols(df, min_subjects=args.min_subjects_per_protocol)
    if df.empty:
        raise ValueError("Nenhum sujeito restou após o filtro de protocolos. "
                        "Reduza --min_subjects_per_protocol.")

    # ── Feature engineering ────────────────────────────────────────────────────
    df["coil_era"] = df["SessionID"].apply(assign_coil_era)
    df["strata"]   = df.apply(assign_strata, axis=1)

    print(f"Sujeitos únicos  : {df['SessionID'].nunique()}", flush=True)
    print(f"Patches totais   : {len(df)}", flush=True)
    print(f"Estratos únicos  : {df['strata'].nunique()}", flush=True)
    print(f"Distribuição era : {df.drop_duplicates('SessionID')['coil_era'].value_counts().to_dict()}", flush=True)

    # ── Split ──────────────────────────────────────────────────────────────────
    df_train, df_val, df_test = stratified_subject_split(
        df,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # ── Relatório ──────────────────────────────────────────────────────────────
    print_report(df, df_train, df_val, df_test)

    # ── Salvamento ─────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols_to_save = [
        "SessionID", "dwi_path", "center_x", "center_y", "center_z",
        "protocol", "coil_era", "strata",
    ]

    df_train.to_csv(out_dir / "train.csv", index=False, columns=cols_to_save)
    df_val.to_csv(  out_dir / "val.csv",   index=False, columns=cols_to_save)
    df_test.to_csv( out_dir / "test.csv",  index=False, columns=cols_to_save)

    print(f"CSVs salvos em: {out_dir}", flush=True)
    print(f"  train.csv  : {len(df_train)} linhas", flush=True)
    print(f"  val.csv    : {len(df_val)} linhas", flush=True)
    print(f"  test.csv   : {len(df_test)} linhas", flush=True)


if __name__ == "__main__":
    main()