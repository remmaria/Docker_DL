#!/bin/bash
#SBATCH --job-name=dmri_hfd_recon
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/recon_hfd.%A_%a.err
#SBATCH --output=logs/recon_hfd.%A_%a.out
#
# Reconstrucao do conjunto de teste com a HFD3D treinada (etapa 5e, ver
# scripts/05e_reconstruct_hfd.py e protocolo/addendum 2026-08-27 secao 8).
# Mesmo esquema de CKPT_JOB_ID/CKPT_PATH/RECON_TAG/RECON_SUBJECTS/RECON_LIMIT
# de slurm/05d_reconstruct_amt.sh (ver comentarios la para o detalhe de
# cada variavel -- reproduzido aqui so o essencial). --time maior que
# slurm/05d_reconstruct_amt.sh (0-08:00:00 vs 0-04:00:00) porque cada patch
# roda o loop DDIM completo (mais lento por patch, ver docstring de
# scripts/05e_reconstruct_hfd.py) -- ajuste se necessario.
#
# Uso:
#   sbatch --array=1-N slurm/05e_reconstruct_hfd.sh <work_dir>
#   sbatch slurm/05e_reconstruct_hfd.sh <work_dir> <shell_b> <n_level>
#   CKPT_JOB_ID=10972424_0 sbatch slurm/05e_reconstruct_hfd.sh <work_dir> <shell_b> <n_level>
#   RECON_TAG=algumnome sbatch slurm/05e_reconstruct_hfd.sh <work_dir> <shell_b> <n_level>
#
# USE_QUALITY_COND=1 / ONLY_VALID=0 / CORR_RADIUS=<r> / NUM_TIMESTEPS=<T> /
# NORM_TYPE=batch -- so ajudam a achar o CKPT_DIR certo (mesmos sufixos de
# slurm/04d_train_hfd.sh) -- o proprio scripts/05e_reconstruct_hfd.py ja le
# a arquitetura de dentro do checkpoint (nao precisa passar de novo na
# linha de comando).
#
# NUM_SAMPLE_STEPS=<K> (variavel de ambiente, opcional) -- SOBRESCREVE o
# num_sample_steps salvo no checkpoint so pra esta reconstrucao (custo vs.
# qualidade da amostragem -- nao precisa bater com o valor usado no treino,
# ver --num-sample-steps em scripts/05e_reconstruct_hfd.py). NAO afeta qual
# CKPT_DIR e procurado (isso e controlado pelo NUM_SAMPLE_STEPS usado no
# TREINO, ver DSTEP_TRAIN abaixo, nome deliberadamente diferente pra nao
# confundir os dois usos).
#   NUM_SAMPLE_STEPS=12 sbatch slurm/05e_reconstruct_hfd.sh <work_dir> <shell_b> <n_level>
#
# DSTEP_TRAIN=<K> (variavel de ambiente, opcional) -- so ajuda a achar o
# CKPT_DIR certo quando o TREINO usou --num-sample-steps != 6 (sufixo
# _dstep<K>, ver slurm/04d_train_hfd.sh). NAO afeta a amostragem em si
# (isso e NUM_SAMPLE_STEPS acima) -- os dois sao independentes de proposito.
#   DSTEP_TRAIN=10 sbatch slurm/05e_reconstruct_hfd.sh <work_dir> <shell_b> <n_level>
#
# DIFFUSION_LOSS_WEIGHT_TRAIN=<valor> (variavel de ambiente, opcional) -- so
# ajuda a achar o CKPT_DIR certo (sufixo _dw<valor>, ver slurm/04d_train_hfd.sh).
#
# STRIDE=<N> / PATCH_SIZE=<N> (variaveis de ambiente, defaults 8/10) --
# mesma convencao de slurm/05d_reconstruct_amt.sh.
#   STRIDE=4 sbatch slurm/05e_reconstruct_hfd.sh <work_dir> <shell_b> <n_level>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05e_reconstruct_hfd.sh <work_dir> [shell_b n_level]}"

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

echo "Reconstruindo (HFD3D) para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/hfd_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_qc"
    echo "USE_QUALITY_COND=1 -- lendo checkpoint da variante consciente da qualidade: $CKPT_DIR"
fi
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    CKPT_DIR="${CKPT_DIR}_inclinv"
    echo "ONLY_VALID=0 -- lendo checkpoint treinado tambem com trincas invalidas: $CKPT_DIR"
fi
if [[ -n "${CORR_RADIUS:-}" && "${CORR_RADIUS}" != "3" ]]; then
    CKPT_DIR="${CKPT_DIR}_r${CORR_RADIUS}"
    echo "CORR_RADIUS=$CORR_RADIUS -- lendo checkpoint com raio de lookup $CORR_RADIUS: $CKPT_DIR"
fi
if [[ -n "${NUM_TIMESTEPS:-}" && "${NUM_TIMESTEPS}" != "1000" ]]; then
    CKPT_DIR="${CKPT_DIR}_t${NUM_TIMESTEPS}"
    echo "NUM_TIMESTEPS=$NUM_TIMESTEPS -- lendo checkpoint com schedule de $NUM_TIMESTEPS passos: $CKPT_DIR"
fi
if [[ -n "${DSTEP_TRAIN:-}" && "${DSTEP_TRAIN}" != "6" ]]; then
    CKPT_DIR="${CKPT_DIR}_dstep${DSTEP_TRAIN}"
    echo "DSTEP_TRAIN=$DSTEP_TRAIN -- lendo checkpoint treinado com num_sample_steps=$DSTEP_TRAIN: $CKPT_DIR"
fi
if [[ -n "${DIFFUSION_LOSS_WEIGHT_TRAIN:-}" && "${DIFFUSION_LOSS_WEIGHT_TRAIN}" != "1.0" && "${DIFFUSION_LOSS_WEIGHT_TRAIN}" != "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_dw${DIFFUSION_LOSS_WEIGHT_TRAIN}"
    echo "DIFFUSION_LOSS_WEIGHT_TRAIN=$DIFFUSION_LOSS_WEIGHT_TRAIN -- lendo checkpoint: $CKPT_DIR"
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
NUM_SAMPLE_STEPS_FLAG=()
if [[ -n "${NUM_SAMPLE_STEPS:-}" ]]; then
    NUM_SAMPLE_STEPS_FLAG=(--num-sample-steps "$NUM_SAMPLE_STEPS")
    echo "NUM_SAMPLE_STEPS=$NUM_SAMPLE_STEPS -- sobrescrevendo passos DDIM so nesta reconstrucao"
fi

RECON_OUT_DIR="$WORK_DIR/hfd_recon"
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    RECON_OUT_DIR="${RECON_OUT_DIR}_qc"
fi
if [[ -n "${RECON_TAG:-}" ]]; then
    RECON_OUT_DIR="$WORK_DIR/hfd_recon_${RECON_TAG}"
    echo "RECON_TAG=$RECON_TAG -- gravando em $RECON_OUT_DIR"
else
    echo "Gravando reconstrucao em $RECON_OUT_DIR"
fi

STRIDE="${STRIDE:-8}"
PATCH_SIZE="${PATCH_SIZE:-10}"
if [[ "$STRIDE" != "8" || "$PATCH_SIZE" != "10" ]]; then
    echo "STRIDE=$STRIDE PATCH_SIZE=$PATCH_SIZE (default seria patch-size=10 stride=8)"
fi

python scripts/05e_reconstruct_hfd.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$RECON_OUT_DIR" \
    --split test --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
    "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}" "${NUM_SAMPLE_STEPS_FLAG[@]}"