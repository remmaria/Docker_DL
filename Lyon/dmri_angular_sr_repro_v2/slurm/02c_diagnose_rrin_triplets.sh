#!/bin/bash
#SBATCH --job-name=dmri_rrin_diag
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/rrin_diag.%J.err
#SBATCH --output=logs/rrin_diag.%J.out
#
# Etapa 2c (diagnostico, opcional): resume gap_deg/t_frac das trincas
# validas (scripts/02c_diagnose_rrin_triplets.py) -- so le os .npz da etapa
# 2b, rapido e barato (sem GPU, sem volume). Ver protocolo secao 10.1.
#
# Uso:
#   sbatch slurm/02c_diagnose_rrin_triplets.sh <work_dir>
#
# TRIPLETS_DIR=<pasta> (variavel de ambiente, default "$WORK_DIR/subsampling"):
# aponta pra uma pasta de trincas DIFERENTE da de sempre -- use isso pra
# inspecionar o resultado de um 02b de teste rodado com OUT_DIR diferente
# (ver slurm/02b_build_rrin_triplets.sh), sem afetar nem depender do
# subsampling/ que um treino ja em andamento esta lendo.
#   TRIPLETS_DIR=$WORK_DIR/subsampling_ens_test sbatch slurm/02c_diagnose_rrin_triplets.sh <work_dir>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02c_diagnose_rrin_triplets.sh <work_dir>}"

source "./00_env_common.sh"

TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
if [[ "$TRIPLETS_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "TRIPLETS_DIR=$TRIPLETS_DIR -- lendo de pasta SEPARADA (nao a de sempre)"
fi

python scripts/02c_diagnose_rrin_triplets.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR"