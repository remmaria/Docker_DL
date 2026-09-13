#!/bin/bash
#SBATCH --job-name=dmri_sub_diag
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/sub_diag.%J.err
#SBATCH --output=logs/sub_diag.%J.out
#
# Etapa 2d (diagnostico, opcional): audita a qualidade geometrica (angulo
# minimo, |centro de massa|, energia eletrostatica) dos esquemas de
# subamostragem (.npz) ja gerados pela etapa 2 (scripts/02_subsample_directions.py
# / slurm/02_baseline_sh.sh) -- so LE os .npz prontos, rapido e barato (sem
# GPU, sem volume). Ver docstring de scripts/02d_diagnose_subsampling.py.
#
# Uso (canonico, sem overrides -- le $WORK_DIR/subsampling, grava
# $WORK_DIR/subsampling_quality.csv):
#   sbatch slurm/02d_diagnose_subsampling.sh <work_dir>
#
# SCHEME_DIR=<pasta> (variavel de ambiente, default "$WORK_DIR/subsampling"):
# aponta pra uma pasta de esquemas DIFERENTE da de sempre -- use isso pra
# comparar dois metodos lado a lado (ex.: rodar a etapa 2 uma vez com
# METHOD=fps num OUT_DIR e outra com METHOD=electrostatic noutro, depois
# rodar este diagnostico apontando SCHEME_DIR pra cada um e comparar as
# tabelas impressas).
#   SCHEME_DIR=$WORK_DIR/subsampling_fps sbatch slurm/02d_diagnose_subsampling.sh <work_dir>
#
# OUT_CSV=<caminho> (variavel de ambiente, default "$WORK_DIR/subsampling_quality.csv"):
# onde gravar o detalhe por (sujeito,shell,n_level) -- passe "-" pra nao
# gravar nenhum CSV (so a tabela resumo impressa no log).

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02d_diagnose_subsampling.sh <work_dir>}"

source "./00_env_common.sh"

SCHEME_DIR="${SCHEME_DIR:-$WORK_DIR/subsampling}"
if [[ "$SCHEME_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "SCHEME_DIR=$SCHEME_DIR -- lendo de pasta SEPARADA (nao a de sempre)"
fi

OUT_CSV="${OUT_CSV:-$WORK_DIR/subsampling_quality.csv}"

if [[ "$OUT_CSV" == "-" ]]; then
    python scripts/02d_diagnose_subsampling.py \
        --manifest "$WORK_DIR/manifest.csv" \
        --scheme-dir "$SCHEME_DIR"
else
    python scripts/02d_diagnose_subsampling.py \
        --manifest "$WORK_DIR/manifest.csv" \
        --scheme-dir "$SCHEME_DIR" \
        --out-csv "$OUT_CSV"
fi