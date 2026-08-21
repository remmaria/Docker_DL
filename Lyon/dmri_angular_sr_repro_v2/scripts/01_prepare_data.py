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
    args = ap.parse_args()

    entries = build_manifest(args.data_root, tol=args.shell_tol, name_suffix=args.name_suffix)
    if not entries:
        print(f"Nenhum trio nii+bval+bvec terminando em '{args.name_suffix}' "
              f"encontrado em {args.data_root}")
        sys.exit(1)

    entries = assign_splits(entries, train=args.train_frac, val=args.val_frac, seed=args.seed)

    out_csv = str(Path(args.out_dir) / "manifest.csv")
    save_manifest(entries, out_csv)

    n_single = sum(1 for e in entries if e.protocol == "single_shell")
    n_multi = sum(1 for e in entries if e.protocol == "multi_shell")
    n_studies = len({e.study for e in entries})
    print(f"Sujeitos encontrados: {len(entries)} em {n_studies} subestudo(s) "
          f"(single-shell: {n_single}, multi-shell: {n_multi})")
    for split in ("train", "val", "test"):
        n = sum(1 for e in entries if e.split == split)
        print(f"  {split}: {n}")
    print("Manifesto salvo em:", out_csv)
    print("\nProxima etapa recomendada: "
          "python scripts/01b_shell_availability_report.py --manifest", out_csv)


if __name__ == "__main__":
    main()
