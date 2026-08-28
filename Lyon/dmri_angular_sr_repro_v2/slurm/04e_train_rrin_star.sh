#!/bin/bash
#SBATCH --job-name=rrin_star
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
# SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train_rrin_star.%A_%a.err
#SBATCH --output=logs/train_rrin_star.%A_%a.out
#
# Treino da RRIN3DStar ("ensemble em estrela", etapa 4e, ver
# scripts/04e_train_rrin_star.py e protocolo secao 14.5 item 1/addendum
# 2026-08-27) para um (shell_b, n_level) especifico -- requer que
# scripts/02b_build_rrin_triplets.py ja tenha rodado COM --ensemble-m>=M
# pra esse work_dir (ver ENSEMBLE_M abaixo). Mesmo padrao de
# slurm/04b_train_rrin.sh (array de configs/experiments.tsv OU
# shell_b/n_level explicitos).
#
# Uso:
#   sbatch --array=1-N slurm/04e_train_rrin_star.sh <work_dir>
#   sbatch slurm/04e_train_rrin_star.sh <work_dir> <shell_b> <n_level>
#
# ENSEMBLE_M=<M> (variavel de ambiente, default 3) -- quantos pares diversos
# por alvo o ensemble usa (ver utils/gradients.py:find_star_ensemble_batch e
# model/rrin3d_star.py). PRECISA bater (ou ser <=) o M usado ao rodar
# scripts/02b_build_rrin_triplets.py --ensemble-m nesse work_dir -- se o
# npz nao tiver os campos '__ens_*', o treino falha cedo com uma mensagem
# clara (ver utils/rrin_dataset.py:RRINTripletDataset.ensemble_m) em vez de
# um erro confuso no meio do loop de treino.
#   ENSEMBLE_M=5 sbatch slurm/04e_train_rrin_star.sh <work_dir> <shell_b> <n_level>
# Grava em out_dir/shell<B>_n<N>_star<M>/ (sufixo automatico, nao colide com
# nenhuma variante de scripts/04b_train_rrin.py nem entre M diferentes).
#
# Resume automatico (ver scripts/04e_train_rrin_star.py) -- mesmo mecanismo
# do RRIN3D/RCAE: RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1.
#
# LR=<valor> (default 1e-3, mesma logica de sufixo automatico _lr<valor> de
# slurm/04b_train_rrin.sh -- so aplica sufixo se != 1e-3).
#
# USE_QUALITY_COND=1 / NORM_TYPE=batch / ONLY_VALID=0 -- mesmo espirito e
# mesmos sufixos automaticos (_qc/_bn/_inclinv) de slurm/04b_train_rrin.sh,
# ver docstring de scripts/04e_train_rrin_star.py para o detalhe de cada um
# aplicado ao ensemble em estrela.
#
# NAO tem ANGULAR_LOSS_WEIGHT/SH_LOSS_* aqui -- scripts/04e_train_rrin_star.py
# ainda nao porta a loss angular/SH pro ensemble em estrela (TODO, ver
# docstring do script).
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04e_train_rrin_star.sh <work_dir> [shell_b n_level]}"
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
ENSEMBLE_M="${ENSEMBLE_M:-3}"
echo "Treinando RRIN3DStar (ensemble em estrela, M=$ENSEMBLE_M) para shell_b=$SHELL_B, n_level=$N_LEVEL"
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
QC_FLAG=()
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    QC_FLAG=(--use-quality-cond)
    echo "USE_QUALITY_COND=1 -- treinando a variante consciente da qualidade de cada par do feixe"
fi
ONLY_VALID_FLAG=()
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    ONLY_VALID_FLAG=(--no-only-valid)
    echo "ONLY_VALID=0 -- treinando/validando tambem com alvos cujo par-unico e invalido"
fi
NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (exige treino do zero)"
fi
python scripts/04e_train_rrin_star.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/rrin_star_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" --ensemble-m "$ENSEMBLE_M" \
    --epochs 150 --batch-size 8 --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    "${RESUME_FLAG[@]}" "${QC_FLAG[@]}" "${ONLY_VALID_FLAG[@]}" "${NORM_TYPE_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"