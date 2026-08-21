#!/bin/bash
#SBATCH --job-name=dmri_baseline_recon
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/baseline_recon.%A_%a.err
#SBATCH --output=logs/baseline_recon.%A_%a.out
#

# sbatch --array=1-145 02b_baseline_reconstruct.sh /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir 1000 10
# Reconstrucao do baseline SH (etapa 3) para UM combo (shell_b, n_level) por
# vez -- roda por array, igual treino/reconstrucao do RCAE, em vez de gerar
# os 30 combos de uma vez num job so. So CPU (nao pedimos --gres=gpu).
# Isso existe pra nao acumular todos os recon_target.nii.gz de todos os
# combos em disco simultaneamente -- combine com CLEANUP_AFTER=1 no job 05
# (avaliacao) pra manter o pico de disco baixo o tempo todo.
#
# Roda DEPOIS de 02_baseline_sh.sh (que gera o esquema de subamostragem,
# pre-requisito) e pode rodar em paralelo com 03_train_rcae.sh (sao
# independentes -- baseline nao depende do RCAE nem vice-versa).
#
# Uso (array = 1 combo shell/nivel por task, le de configs/experiments.tsv):
#   sbatch --array=1-N slurm/02b_baseline_reconstruct.sh <work_dir>
#
# Uso (1 combo especifico, sem array -- todos os sujeitos do split num job so):
#   sbatch slurm/02b_baseline_reconstruct.sh <work_dir> <shell_b> <n_level>
#
# Uso (1 combo especifico, paralelizando por SUJEITO via array -- cada task
# processa so uma fatia dos sujeitos do split; nao precisa de merge depois,
# cada sujeito grava na sua propria pasta em baseline_recon/<tag>/...):
#   sbatch --array=1-20 slurm/02b_baseline_reconstruct.sh <work_dir> <shell_b> <n_level>
#
# ATENCAO: passar <shell_b> <n_level> junto com --array=1-N SEM esse modo de
# shard fazia cada task rodar o combo inteiro (todos os sujeitos) de forma
# DUPLICADA -- mesmo bug que existia em 05_evaluate_and_downstream.sh antes
# da correcao. Agora, se voce passar shell_b/n_level E estiver dentro de um
# array, o script entende que o array e sharding por sujeito.

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02b_baseline_reconstruct.sh <work_dir> [shell_b n_level]}"

EXPERIMENTS_TSV="configs/experiments.tsv"

SHARD_INDEX=0
SHARD_COUNT=1

if [[ -n "${2:-}" && -n "${3:-}" ]]; then
    SHELL_B="$2"
    N_LEVEL="$3"
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        # ver comentario extenso equivalente em slurm/05_evaluate_and_downstream.sh:
        # SHARD_INDEX usa SEMPRE base 1 (nao SLURM_ARRAY_TASK_MIN desta
        # submissao) pra sobreviver a resubmissoes parciais de tasks que
        # falharam (--array=28,36,... preserva os IDs originais). Numa
        # resubmissao parcial, passe SHARD_COUNT_OVERRIDE=<total original>
        # explicitamente, senao o fallback MAX desta submissao fica errado.
        SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
        if [[ -n "${SHARD_COUNT_OVERRIDE:-}" ]]; then
            SHARD_COUNT="$SHARD_COUNT_OVERRIDE"
        elif [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${SLURM_ARRAY_TASK_MIN:-1}" == "1" ]]; then
            SHARD_COUNT="$SLURM_ARRAY_TASK_COUNT"
        else
            SHARD_COUNT=$((${SLURM_ARRAY_TASK_MAX:-1}))
        fi
        echo "[shard] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID MIN=${SLURM_ARRAY_TASK_MIN:-?} MAX=${SLURM_ARRAY_TASK_MAX:-?} -> SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT (numa resubmissao parcial, confira que SHARD_COUNT bate com o total original -- use SHARD_COUNT_OVERRIDE=<N> se nao bater)"
    fi
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LINE=$(grep -v '^#' "$EXPERIMENTS_TSV" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    SHELL_B=$(echo "$LINE" | cut -f1)
    N_LEVEL=$(echo "$LINE" | cut -f2)
else
    echo "Erro: informe shell_b/n_level ou submeta com --array=1-N"
    exit 1
fi

if [[ "$SHARD_COUNT" -gt 1 ]]; then
    echo "Reconstruindo baseline SH para shell_b=$SHELL_B, n_level=$N_LEVEL -- shard $SHARD_INDEX/$SHARD_COUNT"
else
    echo "Reconstruindo baseline SH para shell_b=$SHELL_B, n_level=$N_LEVEL"
fi

source "./00_env_common.sh"

python scripts/03_baseline_sh_interpolation.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/baseline_recon" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
    --split test