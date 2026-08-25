#!/bin/bash
#SBATCH --job-name=dmri_rrin_triplets
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/rrin_trip.%J.err
#SBATCH --output=logs/rrin_trip.%J.out
#
# Etapa 2b: constroi as trincas (par de entrada + alvo) para a linha
# RRIN/VFI-por-triplets, a partir do esquema de subamostragem ja gerado
# pela etapa 2 (scripts/02_subsample_directions.py). Ver
# scripts/02b_build_rrin_triplets.py e protocolo secao 10.1.
#
# Uso:
#   sbatch slurm/02b_build_rrin_triplets.sh <work_dir>
#   MAX_RESIDUAL_DEG=8.0 sbatch slurm/02b_build_rrin_triplets.sh <work_dir>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02b_build_rrin_triplets.sh <work_dir>}"

source "./00_env_common.sh"

MAX_RESIDUAL_DEG="${MAX_RESIDUAL_DEG:-5.0}"
echo "MAX_RESIDUAL_DEG=$MAX_RESIDUAL_DEG"

python scripts/02b_build_rrin_triplets.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/subsampling" \
    --max-residual-deg "$MAX_RESIDUAL_DEG" 
    #--limit 2