#!/bin/bash
#SBATCH --job-name=dmri_pairflow_star_recon
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/recon_pairflow_star.%A_%a.err
#SBATCH --output=logs/recon_pairflow_star.%A_%a.out
#
# Reconstrucao do conjunto de teste com a PairFlowStar treinada (etapa 5k,
# ver scripts/05k_reconstruct_pairflow_star.py e model/pairflow_star.py).
# Mesmo esquema de CKPT_JOB_ID/CKPT_PATH/RECON_TAG/RECON_SUBJECTS/RECON_LIMIT
# de slurm/05f_reconstruct_rrin_star.sh (ver comentarios la para o detalhe de
# cada variavel -- reproduzido aqui so o essencial). --time igual ao do
# RRIN-star (6h vs 4h do pair-unico) pelo mesmo motivo: cada patch roda M
# forwards do pipeline de fluxo (um por par do feixe) em vez de 1.
#
# Uso:
#   sbatch --array=1-N slurm/05k_reconstruct_pairflow_star.sh <work_dir>
#   sbatch slurm/05k_reconstruct_pairflow_star.sh <work_dir> <shell_b> <n_level>
#   CKPT_JOB_ID=10972424_0 sbatch slurm/05k_reconstruct_pairflow_star.sh <work_dir> <shell_b> <n_level>
#   RECON_TAG=algumnome sbatch slurm/05k_reconstruct_pairflow_star.sh <work_dir> <shell_b> <n_level>
#
# ENSEMBLE_M=<M> (default 3) -- so ajuda a achar o CKPT_DIR certo (sufixo
# _pfstar<M>, ver ENSEMBLE_M em slurm/04i_train_pairflow_star.sh). O proprio
# scripts/05k_reconstruct_pairflow_star.py ja le ensemble_m de DENTRO do
# checkpoint.
#
# WEIGHT_QUALITY_COND=1 / NORM_TYPE=batch / ONLY_VALID=0 / FREEZE_FLOW=1 /
# PRETRAINED=1 -- so ajudam a achar o CKPT_DIR certo (sufixos _wqc/_bn/
# _inclinv/_frozen/_pretrained -- ver run_tag em
# scripts/04i_train_pairflow_star.py). NAO ha USE_QUALITY_COND aqui (ao
# contrario de slurm/05f_reconstruct_rrin_star.sh) -- a linha PairFlow nao
# tem esse grau de liberdade (ver model/pairflow_ssl.py:PairFlowNet3D). Estas
# variaveis so servem pra montar o caminho certo do CKPT_DIR (o script Python
# le weight_quality_cond/norm_type/freeze_flow de dentro do proprio
# checkpoint mesmo assim).
#
# STRIDE=<N> / PATCH_SIZE=<N> -- mesma semantica de slurm/05f_reconstruct_rrin_star.sh.

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05k_reconstruct_pairflow_star.sh <work_dir> [shell_b n_level]}"

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

ENSEMBLE_M="${ENSEMBLE_M:-3}"
echo "Reconstruindo (PairFlowStar, M=$ENSEMBLE_M) para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/pairflow_star_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}_pfstar${ENSEMBLE_M}"
if [[ "${WEIGHT_QUALITY_COND:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_wqc"
    echo "WEIGHT_QUALITY_COND=1 -- lendo checkpoint da variante com gap_deg/residual_deg na fusao (PairFlowWeightHead3D): $CKPT_DIR"
fi
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    CKPT_DIR="${CKPT_DIR}_inclinv"
    echo "ONLY_VALID=0 -- lendo checkpoint treinado tambem com alvos de par-unico invalido: $CKPT_DIR"
fi
if [[ "${NORM_TYPE:-instance}" == "batch" ]]; then
    CKPT_DIR="${CKPT_DIR}_bn"
    echo "NORM_TYPE=batch -- lendo checkpoint da variante com BatchNorm3d: $CKPT_DIR"
fi
if [[ "${PRETRAINED:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_pretrained"
    echo "PRETRAINED=1 -- lendo checkpoint inicializado a partir da etapa 4g (flow_net pre-treinado): $CKPT_DIR"
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
    echo "Erro: checkpoint nao encontrado em $CKPT (rode o treino primeiro, ou confira "
    echo "CKPT_JOB_ID/CKPT_PATH/ENSEMBLE_M -- runs disponiveis em: $CKPT_DIR/runs/)"
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

RECON_OUT_DIR="$WORK_DIR/pairflow_star_recon"
if [[ -n "${RECON_TAG:-}" ]]; then
    RECON_OUT_DIR="$WORK_DIR/pairflow_star_recon_${RECON_TAG}"
    echo "RECON_TAG=$RECON_TAG -- gravando em $RECON_OUT_DIR"
else
    echo "Gravando reconstrucao em $RECON_OUT_DIR"
fi

STRIDE="${STRIDE:-8}"
PATCH_SIZE="${PATCH_SIZE:-10}"
if [[ "$STRIDE" != "8" || "$PATCH_SIZE" != "10" ]]; then
    echo "STRIDE=$STRIDE PATCH_SIZE=$PATCH_SIZE (default seria patch-size=10 stride=8)"
fi

TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
if [[ "$TRIPLETS_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "TRIPLETS_DIR=$TRIPLETS_DIR -- lendo trincas de pasta SEPARADA da producao (subsampling/)"
fi

python scripts/05k_reconstruct_pairflow_star.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$RECON_OUT_DIR" \
    --split test --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
    "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}"