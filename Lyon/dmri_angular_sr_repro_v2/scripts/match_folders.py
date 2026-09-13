#!/usr/bin/env python3
"""
Move pastas de sessão de
  /ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias/<SessionID>
para
  /ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias_clinical/<SessionID>

Regras de seleção (a partir do info.tsv):
  - Study em {BRAIN^WPC-6566, BRAIN^WPC-7452, BRAIN^WPC-7707, BRAIN^GILIBR_AD}:
      move TODAS as sessions desses estudos.
  - Study == BRAIN^NOV-SCD:
      move só as sessions cujo último grupo de dígitos após o "_" final
      esteja entre 300 e 499 (ex: ..._312 move; ..._609 fica).

Uso no cluster:
    # 1) Dry-run (padrão) -- só lista o que seria movido, NADA é movido
    python3 move_clinical.py

    # 2) Execução de fato
    python3 move_clinical.py --apply
"""

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

DEFAULT_TSV = "/ix1/tibrahim/rmm270/DATA/DWIs/7TBRP/info.tsv"
DEFAULT_ROOT = "/ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias"
DEFAULT_DEST = "/ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias_clinical"

STUDIES_ALL = {
    "BRAIN^WPC-6566",
    "BRAIN^WPC-7452",
    "BRAIN^WPC-7707",
    "BRAIN^GILIBR_AD",
}
STUDY_PARTIAL = "BRAIN^NOV-SCD"
PARTIAL_MIN = 300
PARTIAL_MAX = 499


def nov_scd_suffix_in_range(session_id: str) -> bool:
    """Último grupo de dígitos após o '_' final deve estar entre PARTIAL_MIN e PARTIAL_MAX."""
    last_part = session_id.rsplit("_", 1)[-1]
    m = re.search(r"\d+", last_part)
    if not m:
        return False
    n = int(m.group(0))
    return PARTIAL_MIN <= n <= PARTIAL_MAX


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", default=DEFAULT_TSV, help="Caminho do info.tsv")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Pasta all_bias (origem)")
    ap.add_argument("--dest", default=DEFAULT_DEST, help="Pasta all_bias_clinical (destino)")
    ap.add_argument("--apply", action="store_true", help="Executa o move de fato (sem isso, é só dry-run)")
    args = ap.parse_args()

    tsv_path = Path(args.tsv)
    root = Path(args.root)
    dest = Path(args.dest)

    if not tsv_path.exists():
        sys.exit(f"ERRO: info.tsv não encontrado em: {tsv_path}")
    if not root.exists():
        sys.exit(f"ERRO: pasta raiz (origem) não encontrada em: {root}")

    with tsv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        reader.fieldnames = [fn.strip().lstrip("﻿") for fn in reader.fieldnames]
        if "SessionID" not in reader.fieldnames or "Study" not in reader.fieldnames:
            sys.exit(f"ERRO: colunas esperadas não encontradas. Colunas lidas: {reader.fieldnames}")

        selected = []  # (session_id, study, reason)
        for row in reader:
            study = row["Study"].strip()
            session_id = row["SessionID"].strip()
            if study in STUDIES_ALL:
                selected.append((session_id, study, "estudo completo"))
            elif study == STUDY_PARTIAL:
                if nov_scd_suffix_in_range(session_id):
                    selected.append((session_id, study, "NOV-SCD, sufixo 300-499"))

    print(f"Encontradas {len(selected)} SessionID(s) selecionadas para mover:")
    for sid, study, reason in selected:
        print(f"  {sid}  [{study}] ({reason})")
    print()

    to_move = []
    missing = []
    already_at_dest = []
    for sid, study, reason in selected:
        src = root / sid
        dst = dest / sid
        if dst.exists():
            already_at_dest.append(sid)
        elif src.exists() and src.is_dir():
            to_move.append((src, dst))
        else:
            missing.append(src)

    if missing:
        print(f"Aviso: {len(missing)} pasta(s) NÃO encontrada(s) em {root} (ignoradas):")
        for m in missing:
            print(f"  {m}")
        print()

    if already_at_dest:
        print(f"Aviso: {len(already_at_dest)} SessionID(s) já existem em {dest} (ignoradas, não sobrescritas):")
        for sid in already_at_dest:
            print(f"  {sid}")
        print()

    if not to_move:
        print("Nenhuma pasta para mover. Encerrando.")
        return

    print(f"Pastas a mover de {root} para {dest}:")
    for src, dst in to_move:
        print(f"  {src.name}")
    print()

    if not args.apply:
        print(f"[DRY-RUN] Nada foi movido. Rode novamente com --apply para mover de fato as {len(to_move)} pasta(s) acima.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Movendo {len(to_move)} pasta(s)...")
    for src, dst in to_move:
        shutil.move(str(src), str(dst))
        print(f"  movida: {src.name}")
    print("Concluído.")


if __name__ == "__main__":
    main()