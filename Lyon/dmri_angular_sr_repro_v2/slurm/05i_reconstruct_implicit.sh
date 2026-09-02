#!/bin/bash
#SBATCH --job-name=dmri_implicit_recon
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/recon_implicit.%A_%a.err
#SBATCH --output=logs/recon_implicit.%A_%a.out
#
# Reconstrucao do conjunto de teste com o modelo de representacao angular
# IMPLICITA treinado (etapa 5i, ver scripts/05i_reconstruct_implicit.py).
# Le DIRETAMENTE de <tag>_scheme.npz (scheme-dir, saida da etapa 2) -- mesmo
# esquema de slurm/04_reconstruct_rcae.sh, NAO precisa de TRIPLETS_DIR nem
# de scripts/02b_build_rrin_triplets.py.
#
# Uso:
#   sbatch --array=1-N slurm/05i_reconstruct_implicit.sh <work_dir>
#   sbatch slurm/05i_reconstruct_implicit.sh <work_dir> <shell_b> <n_level>
#   CKPT_JOB_ID=10972424_0 sbatch slurm/05i_reconstruct_implicit.sh <work_dir> <shell_b> <n_level>
#   RECON_TAG=algumnome sbatch slurm/05i_reconstruct_implicit.sh <work_dir> <shell_b> <n_level>
#
# L_MAX=<inteiro> / BASE_CH=<inteiro> / NORM_TYPE=batch -- so ajudam a achar
# o CKPT_DIR certo (sufixos _lmax<N>/_ch<N>/_bn, ver run_tag em
# scripts/04f_train_implicit.py). O proprio scripts/05i_reconstruct_
# implicit.py ja le l_max/base_ch/norm_type de DENTRO do checkpoint (nao
# precisa bater exatamente com estas variaveis aqui pra reconstruir --
# so precisam apontar pro CKPT_DIR certo).
#
# STRIDE=<N> / PATCH_SIZE=<N> -- mesma semantica de slurm/05_reconstruct_rcae.sh.
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05i_reconstruct_implicit.sh <work_dir> [shell_b n_level]}"

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

echo "Reconstruindo (modelo implicito) para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/implicit_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"
if [[ -n "${L_MAX:-}" ]]; then
    CKPT_DIR="${CKPT_DIR}_lmax${L_MAX}"
    echo "L_MAX=$L_MAX -- lendo checkpoint da variante com ordem SH explicita: $CKPT_DIR"
fi
if [[ -n "${BASE_CH:-}" && "${BASE_CH}" != "16" ]]; then
    CKPT_DIR="${CKPT_DIR}_ch${BASE_CH}"
    echo "BASE_CH=$BASE_CH -- lendo checkpoint da variante com essa largura: $CKPT_DIR"
fi
if [[ "${NORM_TYPE:-instance}" == "batch" ]]; then
    CKPT_DIR="${CKPT_DIR}_bn"
    echo "NORM_TYPE=batch -- lendo checkpoint da variante com BatchNorm3d: $CKPT_DIR"
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
    echo "CKPT_JOB_ID/CKPT_PATH/L_MAX/BASE_CH/NORM_TYPE -- runs disponiveis em: $CKPT_DIR/runs/)"
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

RECON_OUT_DIR="$WORK_DIR/implicit_recon"
if [[ -n "${RECON_TAG:-}" ]]; then
    RECON_OUT_DIR="$WORK_DIR/implicit_recon_${RECON_TAG}"
    echo "RECON_TAG=$RECON_TAG -- gravando em $RECON_OUT_DIR"
else
    echo "Gravando reconstrucao em $RECON_OUT_DIR"
fi

STRIDE="${STRIDE:-8}"
PATCH_SIZE="${PATCH_SIZE:-10}"
if [[ "$STRIDE" != "8" || "$PATCH_SIZE" != "10" ]]; then
    echo "STRIDE=$STRIDE PATCH_SIZE=$PATCH_SIZE (default seria patch-size=10 stride=8)"
fi

SCHEME_DIR="${SCHEME_DIR:-$WORK_DIR/subsampling}"
if [[ "$SCHEME_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "SCHEME_DIR=$SCHEME_DIR -- lendo esquema de pasta SEPARADA da producao (subsampling/)"
fi

python scripts/05i_reconstruct_implicit.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$SCHEME_DIR" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$RECON_OUT_DIR" \
    --split test --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
    "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}"