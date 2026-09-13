#!/bin/bash
#SBATCH --job-name=dmri_split_check
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:15:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/split_check.%J.err
#SBATCH --output=logs/split_check.%J.out
#
# Etapa 1c (diagnostico, opcional): verifica se o split treino/val/teste do
# manifest.csv (etapa 1, scripts/01_prepare_data.py / slurm/01_prepare_data.sh)
# ficou bem balanceado em protocolo, sexo, estudo de origem e idade entre os
# 3 conjuntos -- cobre TODOS os estudos (nao so top-5) e adiciona testes
# qui-quadrado formais. So LE o manifest.csv ja pronto, rapido e barato (sem
# GPU, sem volume) -- pode ser rodado a qualquer momento depois da etapa 1.
# Ver docstring de scripts/01c_check_split_stratification.py.
#
# Uso (canonico, sem overrides -- le $WORK_DIR/manifest.csv, grava
# $WORK_DIR/split_stratification_check.csv):
#   sbatch slurm/01c_check_split_stratification.sh <work_dir>
#
# OUT_CSV=<caminho> (variavel de ambiente, default
# "$WORK_DIR/split_stratification_check.csv"): onde gravar a tabela completa
# estudo x split (contagens e %). Passe "-" pra nao gravar nenhum CSV (so a
# tabela resumo impressa no log).

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 01c_check_split_stratification.sh <work_dir>}"

source "./00_env_common.sh"

OUT_CSV="${OUT_CSV:-$WORK_DIR/split_stratification_check.csv}"

if [[ "$OUT_CSV" == "-" ]]; then
    python scripts/01c_check_split_stratification.py \
        --manifest "$WORK_DIR/manifest.csv"
else
    python scripts/01c_check_split_stratification.py \
        --manifest "$WORK_DIR/manifest.csv" \
        --out-csv "$OUT_CSV"
fi