#!/bin/bash
#SBATCH --job-name=dmri_amt_recon
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/recon_amt.%A_%a.err
#SBATCH --output=logs/recon_amt.%A_%a.out
#
# Reconstrucao do conjunto de teste com a AMT3D treinada (etapa 5d, ver
# scripts/05d_reconstruct_amt.py e protocolo secao 10.3/13). Mesmo esquema
# de CKPT_JOB_ID/CKPT_PATH/RECON_TAG/RECON_SUBJECTS/RECON_LIMIT de
# slurm/05b_reconstruct_rrin.sh (ver comentarios la para o detalhe de cada
# variavel -- reproduzido aqui so o essencial). Recursos SBATCH copiados de
# slurm/05b_reconstruct_rrin.sh como ponto de partida razoavel.
#
# Uso:
#   sbatch --array=1-N slurm/05d_reconstruct_amt.sh <work_dir>
#   sbatch slurm/05d_reconstruct_amt.sh <work_dir> <shell_b> <n_level>
#   CKPT_JOB_ID=10972424_0 sbatch slurm/05d_reconstruct_amt.sh <work_dir> <shell_b> <n_level>
#   RECON_TAG=algumnome sbatch slurm/05d_reconstruct_amt.sh <work_dir> <shell_b> <n_level>
#
# USE_QUALITY_COND=1 / ONLY_VALID=0 -- so ajudam a achar o CKPT_DIR certo
# (sufixos _qc/_inclinv, ver scripts/04c_train_amt.py) -- o proprio
# scripts/05d_reconstruct_amt.py ja le use_quality_cond de dentro do
# checkpoint (nao precisa passar de novo na linha de comando).
#
# NUM_FIELDS=<K> (variavel de ambiente, default 3) -- so ajuda a achar o
# CKPT_DIR certo (sufixo _k<K>, ver NUM_FIELDS em slurm/04c_train_amt.sh).
# O proprio script python le num_fields de dentro do checkpoint.
#   NUM_FIELDS=5 sbatch slurm/05d_reconstruct_amt.sh <work_dir> <shell_b> <n_level>
#
# CORR_RADIUS=<r> (variavel de ambiente, default 3) -- idem, sufixo _r<r>.
#   CORR_RADIUS=2 sbatch slurm/05d_reconstruct_amt.sh <work_dir> <shell_b> <n_level>
#
# NORM_TYPE=batch (variavel de ambiente, default instance) -- idem, sufixo
# _bn (ver NORM_TYPE em slurm/04c_train_amt.sh e model/rrin3d.py:_norm3d,
# reaproveitada por model/amt3d.py).
#   NORM_TYPE=batch sbatch slurm/05d_reconstruct_amt.sh <work_dir> <shell_b> <n_level>
#
# ANGULAR_LOSS=1 (variavel de ambiente, default 0) -- idem, sufixo _sh (a
# loss angular so afeta o treino, nao muda a arquitetura nem o forward).
#   ANGULAR_LOSS=1 sbatch slurm/05d_reconstruct_amt.sh <work_dir> <shell_b> <n_level>
#
# STRIDE=<N> / PATCH_SIZE=<N> (variaveis de ambiente, defaults 8/10) --
# mesma convencao de slurm/05b_reconstruct_rrin.sh.
#   STRIDE=4 sbatch slurm/05d_reconstruct_amt.sh <work_dir> <shell_b> <n_level>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05d_reconstruct_amt.sh <work_dir> [shell_b n_level]}"

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

echo "Reconstruindo (AMT3D) para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/amt_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_qc"
    echo "USE_QUALITY_COND=1 -- lendo checkpoint da variante consciente da qualidade: $CKPT_DIR"
fi
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    CKPT_DIR="${CKPT_DIR}_inclinv"
    echo "ONLY_VALID=0 -- lendo checkpoint treinado tambem com trincas invalidas: $CKPT_DIR"
fi
if [[ -n "${NUM_FIELDS:-}" && "${NUM_FIELDS}" != "3" ]]; then
    CKPT_DIR="${CKPT_DIR}_k${NUM_FIELDS}"
    echo "NUM_FIELDS=$NUM_FIELDS -- lendo checkpoint com K=$NUM_FIELDS campos: $CKPT_DIR"
fi
if [[ -n "${CORR_RADIUS:-}" && "${CORR_RADIUS}" != "3" ]]; then
    CKPT_DIR="${CKPT_DIR}_r${CORR_RADIUS}"
    echo "CORR_RADIUS=$CORR_RADIUS -- lendo checkpoint com raio de lookup $CORR_RADIUS: $CKPT_DIR"
fi
if [[ "${NORM_TYPE:-instance}" == "batch" ]]; then
    CKPT_DIR="${CKPT_DIR}_bn"
    echo "NORM_TYPE=batch -- lendo checkpoint da variante com BatchNorm3d: $CKPT_DIR"
fi
if [[ "${ANGULAR_LOSS:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_sh"
    echo "ANGULAR_LOSS=1 -- lendo checkpoint da variante treinada com a loss angular/SH: $CKPT_DIR"
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

RECON_OUT_DIR="$WORK_DIR/amt_recon"
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    RECON_OUT_DIR="${RECON_OUT_DIR}_qc"
fi
if [[ -n "${RECON_TAG:-}" ]]; then
    RECON_OUT_DIR="$WORK_DIR/amt_recon_${RECON_TAG}"
    echo "RECON_TAG=$RECON_TAG -- gravando em $RECON_OUT_DIR"
else
    echo "Gravando reconstrucao em $RECON_OUT_DIR"
fi

STRIDE="${STRIDE:-8}"
PATCH_SIZE="${PATCH_SIZE:-10}"
if [[ "$STRIDE" != "8" || "$PATCH_SIZE" != "10" ]]; then
    echo "STRIDE=$STRIDE PATCH_SIZE=$PATCH_SIZE (default seria patch-size=10 stride=8)"
fi

python scripts/05d_reconstruct_amt.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$RECON_OUT_DIR" \
    --split test --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
    "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}"