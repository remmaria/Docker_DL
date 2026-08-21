#!/bin/bash
#SBATCH --job-name=train_rcae
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
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

# --val-num-workers/--val-max-cached-subjects explicitos (em vez de contar
# so com o default do script): com persistent_workers=True os workers do
# train_loader (--num-workers 8, --max-cached-subjects 6) E do val_loader
# ficam residentes ao MESMO TEMPO a partir da 1a epoca (nenhum dos dois
# morre entre epocas -- ver comentario em scripts/04_train_rcae.py:main),
# entao o pico de RAM e a SOMA dos dois caches. Val nao precisa do mesmo
# paralelismo do treino (passa 1x por epoca, sem shuffle) -- foi um job
# assim (workers de treino + workers de val subindo por cima na 1a
# validacao) que estourou o --mem=32G original (subimos pra 64G abaixo
# como margem extra, mas o ajuste de fundo continua sendo nao duplicar
# paralelismo desnecessario no val_loader).
# patch-size 10 + q-out 10 (era 24 + "resto da shell") e batch-size 4 + lr
# 1e-3 (era 2 + 1e-4) -- ajustados pra bater com os hiperparametros do
# paper (ver utils/dataset.py e scripts/04_train_rcae.py, reproducao
# completa dos 8 itens revisados contra a implementacao oficial). Patch
# menor (10^3 vs 24^3) reduz bastante o uso de memoria por patch, entao a
# folga de --mem=64G/--max-cached-subjects abaixo continua valendo.
python scripts/04_train_rcae.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/rcae_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size 8 --patch-size 10 --q-out 10 \
    --lr 1e-3 --num-workers 8 --max-cached-subjects 10 --patience 15 \
    --debug-plot-every 1 --debug-plot-every-batches 200 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    --min-tile-coverage 0.15 \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"