#!/bin/bash
#SBATCH --job-name=hfd3d
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
#SBATCH --error=logs/train_hfd.%A_%a.err
#SBATCH --output=logs/train_hfd.%A_%a.out
#
# Treino da HFD3D (etapa 4d, ver scripts/04d_train_hfd.py, model/hfd3d.py e
# protocolo/addendum 2026-08-27 secao 8) para um (shell_b, n_level)
# especifico -- requer que scripts/02b_build_rrin_triplets.py ja tenha
# rodado pra esse work_dir (HFD3D reusa o MESMO esquema de trincas que
# RRIN3D/AMT3D, so muda o modelo) E QUE UMA AMT3D JA TENHA SIDO TREINADA
# pra ESSE MESMO shell_b/n_level (ver TEACHER_CHECKPOINT abaixo -- a HFD3D
# nao treina "do zero" como RRIN3D/AMT3D, ela aprende a replicar/gerar o
# fluxo que a AMT3D ja aprendeu -- ver "TREINO REQUER UM PROFESSOR" em
# model/hfd3d.py).
#
# CUSTO: HFD3D e MAIS LENTA por batch que RRIN3D/AMT3D (cada batch roda o
# loop DDIM completo, ~NUM_SAMPLE_STEPS+1 forwards do denoiser em vez de 1
# forward do modelo inteiro, ver docstring de scripts/04d_train_hfd.py) --
# os recursos abaixo sao copiados de slurm/04c_train_amt.sh como PONTO DE
# PARTIDA, ajuste --time se o throughput real for bem menor.
#
# Uso:
#   sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
#   (TEACHER_CHECKPOINT e obrigatorio, ver abaixo -- nao ha --array=1-N por
#   experiments.tsv aqui porque cada (shell_b,n_level) precisa de um
#   TEACHER_CHECKPOINT diferente, apontando pra AMT3D daquele mesmo
#   shell_b/n_level -- mais seguro exigir que a usuaria informe
#   explicitamente do que inferir errado por convencao de nome de pasta.)
#
# TEACHER_CHECKPOINT=<caminho> (variavel de ambiente, OBRIGATORIA) --
# checkpoint de uma AMT3D ja treinada (scripts/04c_train_amt.py) pro MESMO
# shell_b/n_level deste treino (nao precisa ser a mesma variante de
# USE_QUALITY_COND/NORM_TYPE do aluno, ver scripts/04d_train_hfd.py). A
# arquitetura da professora e lida de dentro do proprio checkpoint.
#   TEACHER_CHECKPOINT=$WORK_DIR/amt_checkpoints/shell1000_n16/best.pt \
#     sbatch slurm/04d_train_hfd.sh <work_dir> 1000 16
#
# Resume automatico (ver scripts/04d_train_hfd.py) -- mesmo mecanismo do
# RRIN/AMT: RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1 (variaveis de
# ambiente).
#
# LR=<valor> (variavel de ambiente, default 1e-3) -- mesmo default
# "canonico" dos outros metodos. Grava sufixo _lr<valor> quando != 1e-3.
#   LR=2e-3 sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
#
# CORR_RADIUS=<r> (variavel de ambiente, default 3) -- liga --corr-radius r
# (ver model/hfd3d.py:HFD3D). BLOQUEANTE em resume -- grava sufixo _r<r>
# quando != 3.
#   CORR_RADIUS=2 sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
#
# NUM_TIMESTEPS=<T> (variavel de ambiente, default 1000) -- liga
# --num-timesteps T (schedule de difusao usado no treino, ver
# model/hfd3d.py). BLOQUEANTE em resume -- grava sufixo _t<T> quando != 1000.
#   NUM_TIMESTEPS=500 sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
#
# NUM_SAMPLE_STEPS=<K> (variavel de ambiente, default 6) -- liga
# --num-sample-steps K (passos DDIM na amostragem -- so custo/qualidade,
# NAO bloqueante em resume, ver scripts/04d_train_hfd.py). Grava sufixo
# _dstep<K> quando != 6.
#   NUM_SAMPLE_STEPS=10 sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
#
# DIFFUSION_LOSS_WEIGHT=<valor> (variavel de ambiente, default 1.0) -- liga
# --diffusion-loss-weight (peso da loss de difusao somada a fotometrica).
# Grava sufixo _dw<valor> quando != 1.0.
#   DIFFUSION_LOSS_WEIGHT=0.5 sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
#
# NORM_TYPE=batch (variavel de ambiente, default instance) -- liga
# --norm-type batch. Exige treinar do ZERO. Grava sufixo _bn.
#   NORM_TYPE=batch NO_RESUME=1 sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
#
# USE_QUALITY_COND=1 (variavel de ambiente) -- liga --use-quality-cond na
# HFD3D (aluna) -- INDEPENDENTE de a professora ter sido treinada assim ou
# nao (ver scripts/04d_train_hfd.py). Grava sufixo _qc.
#   USE_QUALITY_COND=1 sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
#
# ONLY_VALID=0 (variavel de ambiente, default 1) -- liga --no-only-valid.
# Grava sufixo _inclinv.
#   ONLY_VALID=0 sbatch slurm/04d_train_hfd.sh <work_dir> <shell_b> <n_level>
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04d_train_hfd.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 04d_train_hfd.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 04d_train_hfd.sh <work_dir> <shell_b> <n_level>}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:?TEACHER_CHECKPOINT e obrigatorio -- aponte para um \
checkpoint de AMT3D ja treinado (scripts/04c_train_amt.py) pro MESMO shell_b/n_level, ex.: \
TEACHER_CHECKPOINT=\$WORK_DIR/amt_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}/best.pt}"
if [[ ! -f "$TEACHER_CHECKPOINT" ]]; then
    echo "Erro: TEACHER_CHECKPOINT=$TEACHER_CHECKPOINT nao existe -- treine a AMT3D primeiro "
    echo "(scripts/04c_train_amt.py / slurm/04c_train_amt.sh) para shell_b=$SHELL_B n_level=$N_LEVEL."
    exit 1
fi
echo "Treinando HFD3D para shell_b=$SHELL_B, n_level=$N_LEVEL"
echo "Professora (AMT3D congelada): $TEACHER_CHECKPOINT"
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
CORR_RADIUS="${CORR_RADIUS:-3}"
CORR_RADIUS_FLAG=()
if [[ "$CORR_RADIUS" != "3" ]]; then
    CORR_RADIUS_FLAG=(--corr-radius "$CORR_RADIUS")
    echo "CORR_RADIUS=$CORR_RADIUS -- treinando com raio de lookup $CORR_RADIUS (checkpoint em .../_r${CORR_RADIUS}/)"
fi
NUM_TIMESTEPS="${NUM_TIMESTEPS:-1000}"
NUM_TIMESTEPS_FLAG=()
if [[ "$NUM_TIMESTEPS" != "1000" ]]; then
    NUM_TIMESTEPS_FLAG=(--num-timesteps "$NUM_TIMESTEPS")
    echo "NUM_TIMESTEPS=$NUM_TIMESTEPS -- schedule de difusao com $NUM_TIMESTEPS passos de treino (checkpoint em .../_t${NUM_TIMESTEPS}/)"
fi
NUM_SAMPLE_STEPS="${NUM_SAMPLE_STEPS:-6}"
NUM_SAMPLE_STEPS_FLAG=()
if [[ "$NUM_SAMPLE_STEPS" != "6" ]]; then
    NUM_SAMPLE_STEPS_FLAG=(--num-sample-steps "$NUM_SAMPLE_STEPS")
    echo "NUM_SAMPLE_STEPS=$NUM_SAMPLE_STEPS -- amostragem DDIM com $NUM_SAMPLE_STEPS passos (checkpoint em .../_dstep${NUM_SAMPLE_STEPS}/)"
fi
DIFFUSION_LOSS_WEIGHT="${DIFFUSION_LOSS_WEIGHT:-1.0}"
DIFFUSION_LOSS_WEIGHT_FLAG=()
if [[ "$DIFFUSION_LOSS_WEIGHT" != "1.0" && "$DIFFUSION_LOSS_WEIGHT" != "1" ]]; then
    DIFFUSION_LOSS_WEIGHT_FLAG=(--diffusion-loss-weight "$DIFFUSION_LOSS_WEIGHT")
    echo "DIFFUSION_LOSS_WEIGHT=$DIFFUSION_LOSS_WEIGHT -- peso da loss de difusao (checkpoint em .../_dw${DIFFUSION_LOSS_WEIGHT}/)"
fi
NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (checkpoint em .../_bn/, exige treino do zero)"
fi
python scripts/04d_train_hfd.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --teacher-checkpoint "$TEACHER_CHECKPOINT" \
    --out-dir "$WORK_DIR/hfd_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size 8 --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    "${RESUME_FLAG[@]}" "${QC_FLAG[@]}" "${ONLY_VALID_FLAG[@]}" "${CORR_RADIUS_FLAG[@]}" \
    "${NUM_TIMESTEPS_FLAG[@]}" "${NUM_SAMPLE_STEPS_FLAG[@]}" "${DIFFUSION_LOSS_WEIGHT_FLAG[@]}" \
    "${NORM_TYPE_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"