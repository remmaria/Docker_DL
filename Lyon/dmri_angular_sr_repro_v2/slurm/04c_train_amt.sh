#!/bin/bash
#SBATCH --job-name=amt3d
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
#SBATCH --error=logs/train_amt.%A_%a.err
#SBATCH --output=logs/train_amt.%A_%a.out
#
# Treino da AMT3D (etapa 4c, ver scripts/04c_train_amt.py, model/amt3d.py e
# protocolo secao 10.3/13) para um (shell_b, n_level) especifico -- requer
# que scripts/02b_build_rrin_triplets.py ja tenha rodado pra esse work_dir
# (AMT3D reusa o MESMO esquema de trincas que RRIN3D, so muda o modelo).
# Recursos SBATCH copiados de slurm/04b_train_rrin.sh como ponto de partida
# razoavel (AMT3D tem mais operacoes por forward -- correlacao all-pairs +
# lookup em 2 escalas -- mas ainda opera em patches pequenos, ver
# model/amt3d.py; ajuste --time/--mem se o throughput real for bem
# diferente do RRIN).
#
# Uso:
#   sbatch --array=1-N slurm/04c_train_amt.sh <work_dir>
#   sbatch slurm/04c_train_amt.sh <work_dir> <shell_b> <n_level>
#
# Resume automatico (ver scripts/04c_train_amt.py) -- mesmo mecanismo do
# RRIN/RCAE: RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1 (variaveis de
# ambiente).
#
# LR=<valor> (variavel de ambiente, default 1e-3) -- mesmo default "canonico"
# usado por slurm/04b_train_rrin.sh e slurm/03_train_rcae.sh (ver historico
# la sobre por que os tres devem comecar do mesmo LR pra comparacao justa).
# Qualquer LR != 1e-3 grava em out_dir/shell<B>_n<N>[_variante]_lr<valor>/
# (sufixo automatico, ver scripts/04c_train_amt.py) -- nao colide com o
# checkpoint do LR canonico.
#   LR=2e-3 sbatch slurm/04c_train_amt.sh <work_dir> <shell_b> <n_level>
#
# NUM_FIELDS=<K> (variavel de ambiente, default 3) -- liga --num-fields K
# (ver model/amt3d.py:AMT3D, "multi-field" do AMT original). BLOQUEANTE em
# resume (muda o shape das camadas de saida do decoder fino + fusao) --
# grava em out_dir/shell<B>_n<N>[_variante]_k<K>/ quando K!=3.
#   NUM_FIELDS=5 sbatch slurm/04c_train_amt.sh <work_dir> <shell_b> <n_level>
#
# CORR_RADIUS=<r> (variavel de ambiente, default 3) -- liga --corr-radius r
# (ver model/amt3d.py:_corr_lookup_3d). Tambem BLOQUEANTE em resume (muda o
# numero de canais de entrada das cabecas de fluxo) -- grava em
# out_dir/shell<B>_n<N>[_variante]_r<r>/ quando r!=3.
#   CORR_RADIUS=2 sbatch slurm/04c_train_amt.sh <work_dir> <shell_b> <n_level>
#
# NORM_TYPE=batch (variavel de ambiente, default instance) -- liga
# --norm-type batch (ver model/rrin3d.py:_norm3d, reaproveitada por
# model/amt3d.py -- MESMO motivo/artefato de "costura" entre patches na
# reconstrucao com sliding-window). Exige treinar do ZERO. Grava em
# out_dir/shell<B>_n<N>[_qc][_inclinv][_k<K>][_r<r>]_bn/.
#   NORM_TYPE=batch NO_RESUME=1 sbatch slurm/04c_train_amt.sh <work_dir> <shell_b> <n_level>
#
# USE_QUALITY_COND=1 (variavel de ambiente) -- liga --use-quality-cond (mesma
# ideia/convencao de USE_QUALITY_COND em slurm/04b_train_rrin.sh e
# model.rrin3d.RRIN3D). Grava em out_dir/shell<B>_n<N>_qc/.
#   USE_QUALITY_COND=1 sbatch slurm/04c_train_amt.sh <work_dir> <shell_b> <n_level>
#
# ONLY_VALID=0 (variavel de ambiente, default 1) -- liga --no-only-valid
# (mesma semantica de ONLY_VALID em slurm/04b_train_rrin.sh). Grava em
# out_dir/shell<B>_n<N>_inclinv/.
#   ONLY_VALID=0 sbatch slurm/04c_train_amt.sh <work_dir> <shell_b> <n_level>
#
# ANGULAR_LOSS_WEIGHT=<valor> (variavel de ambiente, default 0.0) -- liga
# --angular-loss-weight (mesma infra/convencao de ANGULAR_LOSS_WEIGHT em
# slurm/04b_train_rrin.sh, utils/sh_angular_loss.py). Grava em
# out_dir/.../_sh/. SH_LOSS_Q_OUT/SH_LOSS_HIGH_ORDER_MIN/SH_LOSS_LMAX_CAP
# tem os mesmos defaults/efeito de la.
#   ANGULAR_LOSS_WEIGHT=0.1 sbatch slurm/04c_train_amt.sh <work_dir> <shell_b> <n_level>
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04c_train_amt.sh <work_dir> [shell_b n_level]}"
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
echo "Treinando AMT3D para shell_b=$SHELL_B, n_level=$N_LEVEL"
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
    echo "USE_QUALITY_COND=1 -- treinando a variante consciente da qualidade da trinca (checkpoint em shell${SHELL_B%.*}_n${N_LEVEL}_qc/)"
fi
ONLY_VALID_FLAG=()
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    ONLY_VALID_FLAG=(--no-only-valid)
    echo "ONLY_VALID=0 -- treinando/validando tambem com trincas invalidas (checkpoint em shell${SHELL_B%.*}_n${N_LEVEL}_inclinv/)"
fi
NUM_FIELDS="${NUM_FIELDS:-3}"
NUM_FIELDS_FLAG=()
if [[ "$NUM_FIELDS" != "3" ]]; then
    NUM_FIELDS_FLAG=(--num-fields "$NUM_FIELDS")
    echo "NUM_FIELDS=$NUM_FIELDS -- treinando com K=$NUM_FIELDS campos candidatos (checkpoint em shell${SHELL_B%.*}_n${N_LEVEL}_k${NUM_FIELDS}/)"
fi
CORR_RADIUS="${CORR_RADIUS:-3}"
CORR_RADIUS_FLAG=()
if [[ "$CORR_RADIUS" != "3" ]]; then
    CORR_RADIUS_FLAG=(--corr-radius "$CORR_RADIUS")
    echo "CORR_RADIUS=$CORR_RADIUS -- treinando com raio de lookup $CORR_RADIUS (checkpoint em .../_r${CORR_RADIUS}/)"
fi
NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (checkpoint em .../_bn/, exige treino do zero)"
fi
ANGULAR_LOSS_WEIGHT="${ANGULAR_LOSS_WEIGHT:-0.0}"
ANGULAR_LOSS_FLAG=()
SH_LOSS_FLAG=()
if [[ "$ANGULAR_LOSS_WEIGHT" != "0.0" && "$ANGULAR_LOSS_WEIGHT" != "0" ]]; then
    ANGULAR_LOSS_FLAG=(--angular-loss-weight "$ANGULAR_LOSS_WEIGHT")
    echo "ANGULAR_LOSS_WEIGHT=$ANGULAR_LOSS_WEIGHT -- treinando com a loss angular/SH (checkpoint em .../_sh/, ver utils/sh_angular_loss.py)"
    if [[ -n "${SH_LOSS_Q_OUT:-}" ]]; then
        SH_LOSS_FLAG+=(--sh-loss-q-out "$SH_LOSS_Q_OUT")
        echo "SH_LOSS_Q_OUT=$SH_LOSS_Q_OUT (default 16)"
    fi
    if [[ -n "${SH_LOSS_HIGH_ORDER_MIN:-}" ]]; then
        SH_LOSS_FLAG+=(--sh-loss-high-order-min "$SH_LOSS_HIGH_ORDER_MIN")
        echo "SH_LOSS_HIGH_ORDER_MIN=$SH_LOSS_HIGH_ORDER_MIN (default 4)"
    fi
    if [[ -n "${SH_LOSS_LMAX_CAP:-}" ]]; then
        SH_LOSS_FLAG+=(--sh-loss-lmax-cap "$SH_LOSS_LMAX_CAP")
        echo "SH_LOSS_LMAX_CAP=$SH_LOSS_LMAX_CAP (default 8)"
    fi
fi
python scripts/04c_train_amt.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/amt_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size 8 --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    "${RESUME_FLAG[@]}" "${QC_FLAG[@]}" "${ONLY_VALID_FLAG[@]}" "${NUM_FIELDS_FLAG[@]}" \
    "${CORR_RADIUS_FLAG[@]}" "${NORM_TYPE_FLAG[@]}" "${ANGULAR_LOSS_FLAG[@]}" "${SH_LOSS_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"