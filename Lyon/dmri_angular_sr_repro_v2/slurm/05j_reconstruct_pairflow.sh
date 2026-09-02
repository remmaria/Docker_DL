#!/bin/bash
#SBATCH --job-name=dmri_pairflow_recon
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/recon_pairflow.%A_%a.err
#SBATCH --output=logs/recon_pairflow.%A_%a.out
#
# Reconstrucao do conjunto de teste com a PairFlowInterp3D treinada (etapa
# 5j, ver scripts/05j_reconstruct_pairflow.py e addendum secao 20.15).
# Mesmo esquema de CKPT_JOB_ID/CKPT_PATH/RECON_TAG/RECON_SUBJECTS/
# RECON_LIMIT/STRIDE/PATCH_SIZE de slurm/05b_reconstruct_rrin.sh -- ver
# comentarios la para o detalhe de cada variavel, reproduzido aqui so o
# essencial.
#
# Uso:
#   sbatch slurm/05j_reconstruct_pairflow.sh <work_dir> <shell_b> <n_level>
#   CKPT_JOB_ID=10972424_0 sbatch slurm/05j_reconstruct_pairflow.sh <work_dir> <shell_b> <n_level>
#   RECON_TAG=algumnome sbatch slurm/05j_reconstruct_pairflow.sh <work_dir> <shell_b> <n_level>
#
# PRETRAINED=1 (default 0) / FREEZE_FLOW=1 (default 0) / ONLY_VALID=0
# (default 1) / NORM_TYPE=batch (default instance) -- so ajudam a achar o
# CKPT_DIR certo (sufixos _pretrained/_frozen/_inclinv/_bn, ver
# scripts/04h_train_pairflow_finetune.py) -- o proprio
# scripts/05j_reconstruct_pairflow.py ja le norm_type/freeze_flow de
# dentro do checkpoint (nao precisa passar de novo na linha de comando).
#   PRETRAINED=1 sbatch slurm/05j_reconstruct_pairflow.sh <work_dir> <shell_b> <n_level>
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05j_reconstruct_pairflow.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 05j_reconstruct_pairflow.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 05j_reconstruct_pairflow.sh <work_dir> <shell_b> <n_level>}"

echo "Reconstruindo (PairFlowInterp3D) para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/pairflow_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    CKPT_DIR="${CKPT_DIR}_inclinv"
    echo "ONLY_VALID=0 -- lendo checkpoint treinado tambem com trincas invalidas: $CKPT_DIR"
fi
if [[ "${NORM_TYPE:-instance}" == "batch" ]]; then
    CKPT_DIR="${CKPT_DIR}_bn"
    echo "NORM_TYPE=batch -- lendo checkpoint da variante com BatchNorm3d: $CKPT_DIR"
fi
if [[ "${PRETRAINED:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_pretrained"
    echo "PRETRAINED=1 -- lendo checkpoint inicializado do pre-treino da etapa 4g: $CKPT_DIR"
fi
if [[ "${FREEZE_FLOW:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_frozen"
    echo "FREEZE_FLOW=1 -- lendo checkpoint com flow_net congelado: $CKPT_DIR"
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
    echo "Erro: checkpoint nao encontrado em $CKPT (rode o treino primeiro -- "
    echo "scripts/04h_train_pairflow_finetune.py -- ou confira CKPT_JOB_ID/CKPT_PATH/"
    echo "PRETRAINED/FREEZE_FLOW/ONLY_VALID/NORM_TYPE. Runs disponiveis em: $CKPT_DIR/runs/)"
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

RECON_OUT_DIR="$WORK_DIR/pairflow_recon"
if [[ -n "${RECON_TAG:-}" ]]; then
    RECON_OUT_DIR="$WORK_DIR/pairflow_recon_${RECON_TAG}"
    echo "RECON_TAG=$RECON_TAG -- gravando em $RECON_OUT_DIR"
else
    echo "Gravando reconstrucao em $RECON_OUT_DIR"
fi

STRIDE="${STRIDE:-8}"
PATCH_SIZE="${PATCH_SIZE:-10}"
if [[ "$STRIDE" != "8" || "$PATCH_SIZE" != "10" ]]; then
    echo "STRIDE=$STRIDE PATCH_SIZE=$PATCH_SIZE (default seria patch-size=10 stride=8)"
fi

python scripts/05j_reconstruct_pairflow.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$RECON_OUT_DIR" \
    --split test --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
    "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}"