#!/bin/bash
#SBATCH --job-name=pairflow_ssl
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train_pairflow_ssl.%A_%a.err
#SBATCH --output=logs/train_pairflow_ssl.%A_%a.out
#
# Etapa 4g (linha nova `pairflow_ssl`, Etapa 1/2 -- ver
# model/pairflow_ssl.py e addendum secao 20.15): pre-treino AUTO-
# SUPERVISIONADO do fluxo bidirecional entre pares de direcoes reais
# QUAISQUER, sem trinca/alvo curado -- NAO precisa que
# scripts/02b_build_rrin_triplets.py tenha rodado, so precisa do
# manifesto (ver scripts/01_prepare_data.py).
#
# Uso:
#   sbatch slurm/04g_train_pairflow_ssl.sh <work_dir> <shell_b>
#
# CONSISTENCY_WEIGHT=<valor> (default 0.1) / SMOOTH_WEIGHT=<valor> (default
# 0.0, DESLIGADO de proposito -- ver docstring de
# model.pairflow_ssl.pairflow_ssl_losses: suavidade pode apagar estrutura
# real de cruzamento) -- pesos dos termos de loss auto-supervisionada.
#   SMOOTH_WEIGHT=0.01 sbatch slurm/04g_train_pairflow_ssl.sh <work_dir> 1000
#
# MIN_PAIR_GAP_DEG=<graus> (default 5) / MAX_PAIR_GAP_DEG=<graus> (default
# vazio = sem teto, DE PROPOSITO -- ver docstring de
# utils/pairflow_ssl_dataset.py: o ponto central da ideia e' treinar TAMBEM
# com pares distantes).
#
# NORM_TYPE=batch (default instance) -- mesma semantica/custo (exige treino
# do zero) de NORM_TYPE em slurm/04b_train_rrin.sh.
#
# LR=<valor> (default 1e-3). RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1 --
# mesmo mecanismo de resume automatico dos demais treinos.
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04g_train_pairflow_ssl.sh <work_dir> <shell_b>}"
SHELL_B="${2:?uso: sbatch 04g_train_pairflow_ssl.sh <work_dir> <shell_b>}"

echo "Pre-treinando PairFlowNet3D (auto-supervisionado) para shell_b=$SHELL_B"
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

CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.1}"
SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.0}"
echo "CONSISTENCY_WEIGHT=$CONSISTENCY_WEIGHT SMOOTH_WEIGHT=$SMOOTH_WEIGHT (default 0.1/0.0)"

GAP_FLAGS=()
if [[ -n "${MIN_PAIR_GAP_DEG:-}" ]]; then
    GAP_FLAGS+=(--min-pair-gap-deg "$MIN_PAIR_GAP_DEG")
    echo "MIN_PAIR_GAP_DEG=$MIN_PAIR_GAP_DEG (default 5)"
fi
if [[ -n "${MAX_PAIR_GAP_DEG:-}" ]]; then
    GAP_FLAGS+=(--max-pair-gap-deg "$MAX_PAIR_GAP_DEG")
    echo "MAX_PAIR_GAP_DEG=$MAX_PAIR_GAP_DEG (default vazio = sem teto)"
fi

NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (exige treino do zero)"
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
echo "BATCH_SIZE=$BATCH_SIZE (default 8) -- NAO muda o checkpoint/run_tag: batch size nao "
echo "altera o shape dos pesos, entao e' seguro mudar entre reruns/resumes do MESMO checkpoint "
echo "(diferente de NORM_TYPE, por exemplo)."

# BATCH_LOG_EVERY=<N> (default 10) -- grava so 1 a cada N linhas no
# batch_log.csv (o arquivo cresce rapido em treinos longos -- default
# antigo era logar TODO batch). PRINT_EVERY=<N> (default 20) -- imprime no
# stdout, a cada N batches, wait_s (dataloader/CPU) vs. compute_s (GPU,
# com torch.cuda.synchronize) + memoria de GPU alocada -- diagnostico
# rapido de gargalo sem esperar o resumo de fim de epoca. 0 desliga.
BATCH_LOG_EVERY="${BATCH_LOG_EVERY:-10}"
PRINT_EVERY="${PRINT_EVERY:-20}"
echo "BATCH_LOG_EVERY=$BATCH_LOG_EVERY PRINT_EVERY=$PRINT_EVERY (default 10/20)"

# GAP_HIST_STEP_DEG=<graus> (default 15, 0 desliga) -- imprime, no resumo de
# fim de epoca, um histograma do gap angular dos pares REALMENTE sorteados
# naquela epoca (torna visivel no log a cobertura O(N^2) de pares, ver
# addendum secao 20.15 -- batches/epoca continua fixado pelo numero de
# tiles espaciais, nao pelo numero de pares; o histograma mostra que o
# sorteio esta cobrindo pares proximos E distantes ao longo do treino).
GAP_HIST_STEP_DEG="${GAP_HIST_STEP_DEG:-15}"
echo "GAP_HIST_STEP_DEG=$GAP_HIST_STEP_DEG (default 15, 0 desliga)"

# FREEZE_SUBJECT_ORDER=1 (default 0 -- revertido a pedido explicito da
# usuaria em 2026-09-02: o experimento de congelar a ordem dos sujeitos
# entre epocas nao resolveu o gargalo real de dataloading -- confirmado
# via LOG_WORKER_LOADS que o mesmo sujeito e' recarregado por TODOS os
# workers a cada troca DENTRO da mesma epoca, o que freeze_order nao
# evita -- e coincidiu com uma rodada travada). Comportamento default
# agora e' o ANTIGO (ordem reembaralhada a cada epoca, como sempre foi).
# FREEZE_SUBJECT_ORDER=1 liga de volta o experimento so' pra quem quiser
# retestar deliberadamente.
FREEZE_ORDER_FLAG=()
if [[ "${FREEZE_SUBJECT_ORDER:-0}" == "1" ]]; then
    FREEZE_ORDER_FLAG=(--freeze-subject-order)
    echo "FREEZE_SUBJECT_ORDER=1 -- ordem dos sujeitos CONGELADA entre epocas (experimental, revertido por padrao)"
fi

# LOG_WORKER_LOADS=1 (default 0): imprime worker_id/subject_tag a cada
# carga real de disco -- diagnostico pontual (confirma se o mesmo sujeito
# esta sendo recarregado por workers diferentes), gera muitas linhas, nao
# deixe ligado num treino longo de producao.
LOG_WORKER_LOADS_FLAG=()
if [[ "${LOG_WORKER_LOADS:-0}" == "1" ]]; then
    LOG_WORKER_LOADS_FLAG=(--log-worker-loads)
    echo "LOG_WORKER_LOADS=1 -- logando worker_id/subject_tag a cada carga real de disco (diagnostico pontual)"
fi

python scripts/04g_train_pairflow_ssl.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --out-dir "$WORK_DIR/pairflow_ssl_checkpoints" \
    --shell-b "$SHELL_B" \
    --epochs 150 --batch-size "$BATCH_SIZE" --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    --consistency-weight "$CONSISTENCY_WEIGHT" --smooth-weight "$SMOOTH_WEIGHT" \
    --batch-log-every "$BATCH_LOG_EVERY" --print-every "$PRINT_EVERY" \
    --gap-hist-step-deg "$GAP_HIST_STEP_DEG" \
    "${RESUME_FLAG[@]}" "${GAP_FLAGS[@]}" "${NORM_TYPE_FLAG[@]}" \
    "${FREEZE_ORDER_FLAG[@]}" "${LOG_WORKER_LOADS_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"