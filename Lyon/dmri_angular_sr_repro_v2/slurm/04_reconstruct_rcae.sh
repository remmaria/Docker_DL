#!/bin/bash
#SBATCH --job-name=dmri_rcae_recon
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/recon.%A_%a.err
#SBATCH --output=logs/recon.%A_%a.out
#
# Reconstrucao do conjunto de teste com o RCAE treinado (etapa 5). Mesmo
# esquema de array de slurm/03_train_rcae.sh -- roda depois que o
# checkpoint correspondente ja existe.
#
# Por padrao usa o checkpoint "canonico" (out_dir/best.pt), que e sempre o
# do treino MAIS RECENTE daquele combo (shell_b, n_level) -- rodar
# 03_train_rcae.sh de novo pro mesmo combo sobrescreve esse arquivo. Se
# quiser um checkpoint de um run especifico mais antigo (cada treino
# tambem salva uma copia permanente em rcae_checkpoints/shell*_n*/runs/
# <job_id>/best.pt, nunca sobrescrita), passe o job_id ou um caminho
# explicito via variavel de ambiente:
#
# Uso:
#   sbatch --array=1-N slurm/04_reconstruct_rcae.sh <work_dir>
#   sbatch slurm/04_reconstruct_rcae.sh <work_dir> <shell_b> <n_level>   # sem array
#   CKPT_JOB_ID=10972424_0 sbatch slurm/04_reconstruct_rcae.sh <work_dir> <shell_b> <n_level>
#   CKPT_PATH=/caminho/explicito/best.pt sbatch slurm/04_reconstruct_rcae.sh <work_dir> <shell_b> <n_level>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04_reconstruct_rcae.sh <work_dir> [shell_b n_level]}"

EXPERIMENTS_TSV="configs/experiments.tsv"

if [[ -n "${2:-}" && -n "${3:-}" ]]; then
    SHELL_B="$2"
    N_LEVEL="$3"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LINE=$(grep -v '^#' "$EXPERIMENTS_TSV" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    SHELL_B=$(echo "$LINE" | cut -f1)
    N_LEVEL=$(echo "$LINE" | cut -f2)
else
    echo "Erro: informe shell_b/n_level ou submeta com --array=1-N"
    exit 1
fi

echo "Reconstruindo (RCAE) para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/rcae_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"
if [[ -n "${CKPT_PATH:-}" ]]; then
    CKPT="$CKPT_PATH"
    echo "Usando checkpoint explicito (CKPT_PATH): $CKPT"
elif [[ -n "${CKPT_JOB_ID:-}" ]]; then
    CKPT="$CKPT_DIR/runs/$CKPT_JOB_ID/best.pt"
    echo "Usando checkpoint do run job_id=$CKPT_JOB_ID: $CKPT"
else
    CKPT="$CKPT_DIR/best.pt"
    echo "Usando checkpoint canonico (mais recente): $CKPT"
fi
if [[ ! -f "$CKPT" ]]; then
    echo "Erro: checkpoint nao encontrado em $CKPT (rode o treino primeiro, ou confira "
    echo "CKPT_JOB_ID/CKPT_PATH -- runs disponiveis em: $CKPT_DIR/runs/)"
    exit 1
fi

python scripts/05_reconstruct_rcae.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$WORK_DIR/rcae_recon" \
    --split test --patch-size 24 --stride 16