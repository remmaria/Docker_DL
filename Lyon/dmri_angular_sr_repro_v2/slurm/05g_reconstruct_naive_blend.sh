#!/bin/bash
#SBATCH --job-name=dmri_naive_blend
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/naive_blend.%J.err
#SBATCH --output=logs/naive_blend.%J.out
#
# Etapa 5g (baseline "burro", sem rede): reconstroi via
# scripts/05g_reconstruct_naive_blend.py -- blend ingenuo
# (1-t_frac)*vol_a + t_frac*vol_b do par (a,b) ja escolhido pelo 02b, sem
# nenhuma rede neural. So numpy/nibabel, sem GPU -- roda rapido (nao tem
# sliding-window, e so algebra de arrays inteiros). Depois de rodar, use
# --extra-method naive_blend=<out_dir> em 06_evaluate_reconstruction.py /
# 07_downstream_dti_noddi.py pra comparar com RRIN/AMT/HFD/baseline_sh.
#
# Uso:
#   sbatch slurm/05g_reconstruct_naive_blend.sh <work_dir> <shell_b> <n_level>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05g_reconstruct_naive_blend.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 05g_reconstruct_naive_blend.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 05g_reconstruct_naive_blend.sh <work_dir> <shell_b> <n_level>}"

source "./00_env_common.sh"

SPLIT="${SPLIT:-test}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/naive_blend_recon}"
TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"

python scripts/05g_reconstruct_naive_blend.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$OUT_DIR" \
    --split "$SPLIT"