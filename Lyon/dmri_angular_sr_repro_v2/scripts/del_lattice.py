#!/usr/bin/env python3
"""
Deleta (ou lista, em modo dry-run) as pastas em
/ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias
cujo nome de pasta corresponde ao SessionID de linhas do info.tsv
onde Study == "BRAIN^WPC-7317".

Uso no cluster:
    # 1) Dry-run (padrão) -- só lista o que seria apagado, NADA é deletado
    python3 delete_wpc7317.py

    # 2) Deleção de fato
    python3 delete_wpc7317.py --apply

Caminhos podem ser sobrescritos via --tsv e --root se necessário.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

DEFAULT_TSV = "/ix1/tibrahim/rmm270/DATA/DWIs/7TBRP/info.tsv"
DEFAULT_ROOT = "/ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias"
TARGET_STUDY = "BRAIN^WPC-7317"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", default=DEFAULT_TSV, help="Caminho do info.tsv")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Pasta all_bias onde ficam as subpastas por SessionID")
    ap.add_argument("--study", default=TARGET_STUDY, help="Valor de Study a filtrar")
    ap.add_argument("--apply", action="store_true", help="Executa a deleção de fato (sem isso, é só dry-run)")
    args = ap.parse_args()

    tsv_path = Path(args.tsv)
    root = Path(args.root)

    if not tsv_path.exists():
        sys.exit(f"ERRO: info.tsv não encontrado em: {tsv_path}")
    if not root.exists():
        sys.exit(f"ERRO: pasta raiz não encontrada em: {root}")

    matching_sessions = []
    with tsv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # normaliza nomes de coluna (tira BOM/espaços residuais, se houver)
        reader.fieldnames = [fn.strip().lstrip("﻿") for fn in reader.fieldnames]
        if "SessionID" not in reader.fieldnames or "Study" not in reader.fieldnames:
            sys.exit(f"ERRO: colunas esperadas não encontradas. Colunas lidas: {reader.fieldnames}")
        for row in reader:
            if row["Study"].strip() == args.study:
                matching_sessions.append(row["SessionID"].strip())

    print(f"Encontradas {len(matching_sessions)} SessionID(s) com Study == '{args.study}':")
    for s in matching_sessions:
        print(f"  {s}")
    print()

    to_delete = []
    missing = []
    for session_id in matching_sessions:
        folder = root / session_id
        if folder.exists() and folder.is_dir():
            to_delete.append(folder)
        else:
            missing.append(folder)

    if missing:
        print(f"Aviso: {len(missing)} pasta(s) correspondente(s) NÃO encontrada(s) em {root} (ignoradas):")
        for m in missing:
            print(f"  {m}")
        print()

    if not to_delete:
        print("Nenhuma pasta para deletar. Encerrando.")
        return

    print(f"Pastas a deletar em {root}:")
    for d in to_delete:
        print(f"  {d}")
    print()

    if not args.apply:
        print(f"[DRY-RUN] Nada foi deletado. Rode novamente com --apply para deletar de fato as {len(to_delete)} pasta(s) acima.")
        return

    print(f"Deletando {len(to_delete)} pasta(s)...")
    for d in to_delete:
        shutil.rmtree(d)
        print(f"  deletada: {d}")
    print("Concluído.")


if __name__ == "__main__":
    main()