#!/bin/bash
#SBATCH --job-name=dmri_poc_csd
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/poc_csd.%j.err
#SBATCH --output=logs/poc_csd.%j.out
#
# Prova de conceito: CSD em poucas direcoes vs. aquisicao densa vs.
# preenchido por SH (e por RCAE, se ja tiver reconstrucao) -- ver
# scripts/poc_csd_direction_count.py pro racional completo. Amostra
# pequena de sujeitos por padrao (--n-subjects 8), entao roda como job
# unico (sem array/sharding) -- se quiser rodar em mais sujeitos e achar
# lento, aumente --cpus-per-task/--time ou peca pra eu adicionar sharding
# igual aos outros scripts do pipeline.
#
# Uso:
#   sbatch slurm/poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> [n_subjects]

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> [n_subjects]}"
SHELL_B="${2:?uso: sbatch poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> [n_subjects]}"
N_LEVEL="${3:?uso: sbatch poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> [n_subjects]}"
N_SUBJECTS="${4:-8}"

echo "Prova de conceito CSD para shell_b=$SHELL_B, n_level=$N_LEVEL, n_subjects=$N_SUBJECTS"

source "./00_env_common.sh"

python scripts/poc_csd_direction_count.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --baseline-dir "$WORK_DIR/baseline_recon" \
    --rcae-dir "$WORK_DIR/rcae_recon" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --split test --n-subjects "$N_SUBJECTS" --seed 0 \
    --out-csv "$WORK_DIR/metrics/poc_csd_shell${SHELL_B%.*}_n${N_LEVEL}.csv"