#!/bin/bash
#SBATCH --job-name=pairflow_ft
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train_pairflow_ft.%A_%a.err
#SBATCH --output=logs/train_pairflow_ft.%A_%a.out
#
# Etapa 4h (linha nova `pairflow_ssl`, Etapa 2/2 -- ver
# model/pairflow_ssl.py e addendum secao 20.15): fine-tuning
# SUPERVISIONADO da PairFlowInterp3D nas MESMAS trincas curadas de sempre
# -- requer que scripts/02b_build_rrin_triplets.py ja tenha rodado.
#
# Uso (com pre-treino da etapa 4g):
#   INIT_CHECKPOINT=$WORK_DIR/pairflow_ssl_checkpoints/shell1000/best.pt \
#     sbatch slurm/04h_train_pairflow_finetune.sh <work_dir> <shell_b> <n_level>
#
# Uso (controle, sem pre-treino -- treina do zero nas trincas curadas):
#   sbatch slurm/04h_train_pairflow_finetune.sh <work_dir> <shell_b> <n_level>
#
# FREEZE_FLOW=1 (default 0, requer INIT_CHECKPOINT) -- congela o fluxo
# pre-treinado, so treina o blend/refino (ver --freeze-flow e docstring de
# model.pairflow_ssl.PairFlowInterp3D).
#
# ONLY_VALID=0 (default 1) -- mesma semantica de ONLY_VALID em
# slurm/04b_train_rrin.sh.
#
# NORM_TYPE=batch (default instance) -- mesma semantica/custo de NORM_TYPE
# em slurm/04b_train_rrin.sh (precisa bater com o checkpoint da etapa 4g,
# se usado -- scripts/04h_train_pairflow_finetune.py avisa se nao bater).
#
# LR=<valor> (default 1e-4). RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1 --
# mesmo mecanismo de resume automatico dos demais treinos.
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04h_train_pairflow_finetune.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 04h_train_pairflow_finetune.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 04h_train_pairflow_finetune.sh <work_dir> <shell_b> <n_level>}"

echo "Fine-tuning PairFlowInterp3D para shell_b=$SHELL_B, n_level=$N_LEVEL"
source "./00_env_common.sh"

RESUME_FLAG=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_FLAG=(--resume-checkpoint "$RESUME_CHECKPOINT")
    echo "RESUME_CHECKPOINT=$RESUME_CHECKPOINT -- retomando explicitamente deste checkpoint"
elif [[ "${NO_RESUME:-0}" == "1" ]]; then
    RESUME_FLAG=(--no-resume)
    echo "NO_RESUME=1 -- ignorando qualquer last.pt existente, comecando do zero"
fi

LR="${LR:-1e-4}"
echo "LR=$LR (default 1e-4)"

INIT_FLAG=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    INIT_FLAG=(--init-checkpoint "$INIT_CHECKPOINT")
    echo "INIT_CHECKPOINT=$INIT_CHECKPOINT -- inicializando flow_net do pre-treino da etapa 4g (checkpoint em .../_pretrained/)"
else
    echo "INIT_CHECKPOINT nao passado -- treinando PairFlowInterp3D do ZERO (controle, sem sufixo '_pretrained')"
fi

FREEZE_FLAG=()
if [[ "${FREEZE_FLOW:-0}" == "1" ]]; then
    if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
        echo "Erro: FREEZE_FLOW=1 requer INIT_CHECKPOINT (nao faz sentido congelar fluxo do zero)"
        exit 1
    fi
    FREEZE_FLAG=(--freeze-flow)
    echo "FREEZE_FLOW=1 -- congelando flow_net, so treinando refine_net (checkpoint em .../_frozen/)"
fi

ONLY_VALID_FLAG=()
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    ONLY_VALID_FLAG=(--no-only-valid)
    echo "ONLY_VALID=0 -- treinando/validando tambem com trincas invalidas (checkpoint em .../_inclinv/)"
fi

NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (checkpoint em .../_bn/, exige treino do zero)"
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
echo "BATCH_SIZE=$BATCH_SIZE (default 8) -- seguro mudar entre reruns/resumes (nao afeta o "
echo "shape dos pesos); MAS mantenha o MESMO valor entre o run com INIT_CHECKPOINT e o "
echo "run de controle (sem INIT_CHECKPOINT) se quiser comparar os dois de forma justa."

BATCH_LOG_EVERY="${BATCH_LOG_EVERY:-10}"
PRINT_EVERY="${PRINT_EVERY:-20}"
echo "BATCH_LOG_EVERY=$BATCH_LOG_EVERY PRINT_EVERY=$PRINT_EVERY (default 10/20 -- ver "
echo "scripts/04h_train_pairflow_finetune.py --help)"

# FREEZE_SUBJECT_ORDER=1 / LOG_WORKER_LOADS=1 -- mesma flag/racional de
# slurm/04g_train_pairflow_ssl.sh. Revertido a pedido explicito da usuaria
# em 2026-09-02 (o experimento de congelar ordem nao resolveu o gargalo
# real e coincidiu com uma rodada travada) -- comportamento default
# voltou a ser o antigo (ordem reembaralhada a cada epoca).
FREEZE_ORDER_FLAG=()
if [[ "${FREEZE_SUBJECT_ORDER:-0}" == "1" ]]; then
    FREEZE_ORDER_FLAG=(--freeze-subject-order)
    echo "FREEZE_SUBJECT_ORDER=1 -- ordem dos sujeitos CONGELADA entre epocas (experimental, revertido por padrao)"
fi
LOG_WORKER_LOADS_FLAG=()
if [[ "${LOG_WORKER_LOADS:-0}" == "1" ]]; then
    LOG_WORKER_LOADS_FLAG=(--log-worker-loads)
    echo "LOG_WORKER_LOADS=1 -- logando worker_id/subject_tag a cada carga real de disco (diagnostico pontual)"
fi

python scripts/04h_train_pairflow_finetune.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/pairflow_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size "$BATCH_SIZE" --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    --batch-log-every "$BATCH_LOG_EVERY" --print-every "$PRINT_EVERY" \
    "${RESUME_FLAG[@]}" "${INIT_FLAG[@]}" "${FREEZE_FLAG[@]}" "${ONLY_VALID_FLAG[@]}" \
    "${NORM_TYPE_FLAG[@]}" "${FREEZE_ORDER_FLAG[@]}" "${LOG_WORKER_LOADS_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"