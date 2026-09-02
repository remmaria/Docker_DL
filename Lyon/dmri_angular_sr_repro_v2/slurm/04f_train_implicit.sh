#!/bin/bash
#SBATCH --job-name=implicit_angular
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train_implicit.%A_%a.err
#SBATCH --output=logs/train_implicit.%A_%a.out
#
# Treino do modelo de representacao angular IMPLICITA (NeRF/LIIF-style,
# etapa 4f, ver model/implicit_angular.py e scripts/04f_train_implicit.py --
# addendum secao 20.11) para um (shell_b, n_level) especifico.
#
# DIFERENCA IMPORTANTE em relacao a TODOS os wrappers 04b/04c/04d/04e
# (RRIN/AMT/HFD/estrela): este le DIRETAMENTE de <tag>_scheme.npz (saida da
# etapa 2, scripts/02_subsample_directions.py) -- NAO precisa que
# scripts/02b_build_rrin_triplets.py tenha rodado, e por isso NAO tem
# TRIPLETS_DIR/ENSEMBLE_M aqui (nao ha pareamento nenhum nesta linha). Mesmo
# esquema de scheme-dir que scripts/04_train_rcae.py ja usa (ver
# slurm/03_train_rcae.sh).
#
# Uso:
#   sbatch --array=1-N slurm/04f_train_implicit.sh <work_dir>
#   sbatch slurm/04f_train_implicit.sh <work_dir> <shell_b> <n_level>
#
# RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1 -- mesmo mecanismo de resume
# automatico dos demais treinos (ver scripts/04f_train_implicit.py).
#
# LR=<valor> (default 1e-3).
#
# L_MAX=<inteiro> -- ordem par maxima da base SH usada para codificar
# direcao (ver model/implicit_angular.py:sh_positional_encoding). Default
# vazio = automatico (max_order_for_n_directions(n_level), mesma convencao
# do baseline_sh). BLOQUEANTE para resume (muda o shape dos pesos) -- ver
# checagem em scripts/04f_train_implicit.py.
#
# BASE_CH=<inteiro> (default 16) -- largura dos blocos conv. NORM_TYPE=batch
# (default instance) -- mesma semantica/mesmo custo (exige treino do zero)
# de NORM_TYPE em slurm/04b_train_rrin.sh/04e_train_rrin_star.sh.
#
# SCHEME_DIR=<caminho> -- por padrao usa <work_dir>/subsampling (saida da
# etapa 2), mesma convencao de TRIPLETS_DIR nos outros wrappers.
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04f_train_implicit.sh <work_dir> [shell_b n_level]}"
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
    exit 1
fi
echo "Treinando modelo implicito (angular, estilo NeRF/LIIF) para shell_b=$SHELL_B, n_level=$N_LEVEL"
source "./00_env_common.sh"

RESUME_FLAG=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_FLAG=(--resume-checkpoint "$RESUME_CHECKPOINT")
    echo "RESUME_CHECKPOINT=$RESUME_CHECKPOINT -- retomando explicitamente deste checkpoint"
elif [[ "${NO_RESUME:-0}" == "1" ]]; then
    RESUME_FLAG=(--no-resume)
    echo "NO_RESUME=1 -- ignorando qualquer last.pt existente, comecando do zero"
fi

LR="${LR:-1e-3}"
echo "LR=$LR (default 1e-3)"

LMAX_FLAG=()
if [[ -n "${L_MAX:-}" ]]; then
    LMAX_FLAG=(--l-max "$L_MAX")
    echo "L_MAX=$L_MAX -- fixando ordem SH explicita (default seria automatico por n_level)"
fi

BASE_CH="${BASE_CH:-16}"
BASE_CH_FLAG=()
if [[ "$BASE_CH" != "16" ]]; then
    BASE_CH_FLAG=(--base-ch "$BASE_CH")
    echo "BASE_CH=$BASE_CH (default seria 16)"
fi

NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (exige treino do zero)"
fi

SCHEME_DIR="${SCHEME_DIR:-$WORK_DIR/subsampling}"
if [[ "$SCHEME_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "SCHEME_DIR=$SCHEME_DIR -- lendo esquema de pasta SEPARADA da producao (subsampling/)"
fi

python scripts/04f_train_implicit.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$SCHEME_DIR" \
    --out-dir "$WORK_DIR/implicit_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size 8 --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    "${RESUME_FLAG[@]}" "${LMAX_FLAG[@]}" "${BASE_CH_FLAG[@]}" "${NORM_TYPE_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"