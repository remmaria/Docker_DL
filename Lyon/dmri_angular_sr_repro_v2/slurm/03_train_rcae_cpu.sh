#!/bin/bash
#SBATCH --job-name=dmri_rcae_train
#SBATCH --cluster=htc
#SBATCH --partition=preempt
# SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train.%A_%a.err
#SBATCH --output=logs/train.%A_%a.out
#
# Treino do RCAE (etapa 4), no mesmo padrao do seu script de treino
# original (mesma partition/conta/GPU/cpus). Cada linha de
# slurm/configs/experiments.tsv (shell_b, n_level) vira um item do array --
# assim um `sbatch --array=1-N` roda todos os experimentos em paralelo,
# um job por combinacao, cada um na sua GPU.
#
# Uso:
#   1) edite slurm/configs/experiments.tsv com os (shell_b, n_level) que quer rodar
#   2) conte quantas linhas uteis (nao-comentario) tem no arquivo, ex.:
#        N=$(grep -vc '^#' slurm/configs/experiments.tsv)
#   3) sbatch --array=1-$N slurm/03_train_rcae.sh <work_dir>
#
# Para rodar so uma combinacao especifica (sem array), passe shell_b e
# n_level direto:
#   sbatch slurm/03_train_rcae.sh <work_dir> <shell_b> <n_level>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 03_train_rcae.sh <work_dir> [shell_b n_level]}"

EXPERIMENTS_TSV="configs/experiments.tsv"

if [[ -n "${2:-}" && -n "${3:-}" ]]; then
    SHELL_B="$2"
    N_LEVEL="$3"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LINE=$(grep -v '^#' "$EXPERIMENTS_TSV" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    if [[ -z "$LINE" ]]; then
        echo "Erro: nao ha linha $SLURM_ARRAY_TASK_ID em $EXPERIMENTS_TSV (confira --array=1-N)"
        exit 1
    fi
    SHELL_B=$(echo "$LINE" | cut -f1)
    N_LEVEL=$(echo "$LINE" | cut -f2)
else
    echo "Erro: informe shell_b/n_level como argumentos OU submeta com --array=1-N"
    echo "(N = numero de linhas uteis em $EXPERIMENTS_TSV)"
    exit 1
fi

echo "Treinando RCAE para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

python scripts/04_train_rcae.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/rcae_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 2 --batch-size 2 --patch-size 16 --patches-per-subject 4 --torch-threads 12 \
    --lr 1e-4 --num-workers 6 --max-cached-subjects 6 \
    --debug-plot-every 1 --debug-plot-every-batches 20 \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
