#!/bin/bash
#SBATCH --job-name=rrin_recon
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/recon_rrin.%A_%a.err
#SBATCH --output=logs/recon_rrin.%A_%a.out
#
# Reconstrucao do conjunto de teste com a RRIN3D treinada (etapa 5b, ver
# scripts/05b_reconstruct_rrin.py e protocolo secao 10.1). Mesmo esquema de
# CKPT_JOB_ID/CKPT_PATH/RECON_TAG/RECON_SUBJECTS/RECON_LIMIT de
# slurm/04_reconstruct_rcae.sh (ver comentarios la para o detalhe de cada
# variavel -- reproduzido aqui so o essencial).
#
# Uso:
#   sbatch --array=1-N slurm/05b_reconstruct_rrin.sh <work_dir>
#   sbatch slurm/05b_reconstruct_rrin.sh <work_dir> <shell_b> <n_level>
#   CKPT_JOB_ID=10972424_0 sbatch slurm/05b_reconstruct_rrin.sh <work_dir> <shell_b> <n_level>
#   RECON_TAG=algumnome sbatch slurm/05b_reconstruct_rrin.sh <work_dir> <shell_b> <n_level>
#
# USE_QUALITY_COND=1 -- reconstroi com o checkpoint da variante consciente
# da qualidade da trinca (treinada com USE_QUALITY_COND=1 em
# slurm/04b_train_rrin.sh, salva em rrin_checkpoints/shell<B>_n<N>_qc/, ver
# scripts/04b_train_rrin.py). Sem isso, le do checkpoint "cego" padrao
# (shell<B>_n<N>/, sem sufixo). O proprio scripts/05b_reconstruct_rrin.py ja
# le use_quality_cond de dentro do checkpoint (nao precisa passar de novo
# na linha de comando) -- esta variavel so ajuda a achar o CKPT_DIR certo.

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05b_reconstruct_rrin.sh <work_dir> [shell_b n_level]}"

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

echo "Reconstruindo (RRIN3D) para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/rrin_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_qc"
    echo "USE_QUALITY_COND=1 -- lendo checkpoint da variante consciente da qualidade: $CKPT_DIR"
fi
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

SUBJECTS_FLAG=()
if [[ -n "${RECON_SUBJECTS:-}" ]]; then
    SUBJECTS_FLAG=(--subjects "$RECON_SUBJECTS")
    echo "RECON_SUBJECTS=$RECON_SUBJECTS -- restringindo reconstrucao a esse(s) sujeito(s)"
fi
LIMIT_FLAG=()
if [[ -n "${RECON_LIMIT:-}" ]]; then
    LIMIT_FLAG=(--limit "$RECON_LIMIT")
    echo "RECON_LIMIT=$RECON_LIMIT -- restringindo reconstrucao aos primeiros $RECON_LIMIT sujeito(s)"
fi

RECON_OUT_DIR="$WORK_DIR/rrin_recon"
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    RECON_OUT_DIR="${RECON_OUT_DIR}_qc"
fi
if [[ -n "${RECON_TAG:-}" ]]; then
    RECON_OUT_DIR="$WORK_DIR/rrin_recon_${RECON_TAG}"
    echo "RECON_TAG=$RECON_TAG -- gravando em $RECON_OUT_DIR"
else
    echo "Gravando reconstrucao em $RECON_OUT_DIR"
fi

python scripts/05b_reconstruct_rrin.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$RECON_OUT_DIR" \
    --split test --patch-size 10 --stride 8 \
    "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}"