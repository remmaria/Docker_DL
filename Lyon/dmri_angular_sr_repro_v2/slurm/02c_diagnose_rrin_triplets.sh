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

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02c_diagnose_rrin_triplets.sh <work_dir>}"

source "./00_env_common.sh"

python scripts/02c_diagnose_rrin_triplets.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling"