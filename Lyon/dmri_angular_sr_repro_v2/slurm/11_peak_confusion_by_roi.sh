#!/bin/bash
#SBATCH --job-name=dmri_peak_confusion
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-06:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/peak_confusion.%A_%a.err
#SBATCH --output=logs/peak_confusion.%A_%a.out
#
# Etapa 11 (downstream, opcional): TP/FP/FN/TN de picos de FOD (CSD) por
# ROI, comparando ground truth vs. cada metodo de reconstrucao -- ver
# scripts/11_peak_confusion_by_roi.py. CSD por sujeito e' lento (mesma
# familia de slurm/crossing_fiber_stratified_eval, se existir) -- time
# generoso e sharding por sujeito disponivel (mesmo padrao de
# slurm/05_evaluate_and_downstream.sh).
#
# EXTRA_METHOD="nome=caminho" (repetivel via virgula) -- mesma convencao
# de slurm/05_evaluate_and_downstream.sh, ex.:
#   EXTRA_METHOD="rrin=$WORK_DIR/rrin_recon,naive_blend=$WORK_DIR/naive_blend_recon"
#
# SUBSAMPLED_ONLY=1 -- inclui o metodo extra 'subsampled_only' (CSD so nas
# direcoes de entrada reais, sem reconstrucao -- requer TRIPLETS_DIR,
# default "$WORK_DIR/subsampling").
#
# ROI_TRACTS="FX,CGC,CGH,UF" -- mesma convencao de
# slurm/05_evaluate_and_downstream.sh.
#
# PEAK_MATCH_THRESHOLD_DEG=25 (default) -- tolerancia angular pra
# considerar dois picos "o mesmo".
#
# SUBJECTS="tag1,tag2" (variavel de ambiente) roda so nesse(s) sujeito(s)
# especifico(s) em vez do split inteiro -- mesma convencao de POC_SUBJECTS
# em slurm/poc_csd_direction_count.sh. Util pra testar rapido num sujeito
# so antes de rodar o dataset inteiro (CSD por sujeito e caro). Ex.:
#   SUBJECTS="20170920171326_616_20170920171326_616" \
#     sbatch slurm/11_peak_confusion_by_roi.sh <work_dir> 1000 16
#
# Uso (1 combo especifico, todos os sujeitos do split num job so):
#   EXTRA_METHOD="rrin=$WORK_DIR/rrin_recon" \
#     sbatch slurm/11_peak_confusion_by_roi.sh <work_dir> 1000 16
#
# Uso (sharding por sujeito via array -- junte depois com
# scripts/merge_shard_csvs.py):
#   sbatch --array=1-20 slurm/11_peak_confusion_by_roi.sh <work_dir> 1000 16

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 11_peak_confusion_by_roi.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 11_peak_confusion_by_roi.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 11_peak_confusion_by_roi.sh <work_dir> <shell_b> <n_level>}"

source "./00_env_common.sh"

SHARD_INDEX=0
SHARD_COUNT=1
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
    SHARD_COUNT="${SHARD_COUNT_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-${SLURM_ARRAY_TASK_MAX:-1}}}"
    echo "[shard] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -> SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT"
fi

BASELINE_FLAG=(--baseline-dir "$WORK_DIR/baseline_recon")

EXTRA_METHOD_FLAGS=()
if [[ -n "${EXTRA_METHOD:-}" ]]; then
    IFS=',' read -ra _EXTRA_SPECS <<< "$EXTRA_METHOD"
    for spec in "${_EXTRA_SPECS[@]}"; do
        EXTRA_METHOD_FLAGS+=(--extra-method "$spec")
    done
    echo "EXTRA_METHOD=$EXTRA_METHOD"
fi

SUBSAMPLED_ONLY_FLAG=()
if [[ "${SUBSAMPLED_ONLY:-0}" == "1" ]]; then
    TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
    SUBSAMPLED_ONLY_FLAG=(--subsampled-only --triplets-dir "$TRIPLETS_DIR")
    echo "SUBSAMPLED_ONLY=1 -- lendo target_idx de $TRIPLETS_DIR"

    # SH_ORDER_SUBSAMPLED_ONLY=<ordem> (variavel de ambiente, opcional --
    # ver --sh-order-subsampled-only em scripts/11_peak_confusion_by_roi.py
    # e addendum, secao 16.1/18): por padrao (sem esta variavel) o
    # subsampled_only usa a ordem SH "honesta" pras n_level direcoes reais
    # (max_order_for_n_directions(n_level), equivalente ao
    # 'bruto_n_ordem_max' do poc_csd_direction_count.py da usuaria --
    # estruturalmente incapaz de representar cruzamento, nao por falta de
    # dado). Forcar aqui a MESMA ordem dos metodos de volume completo
    # (equivalente ao 'bruto_n' do PoC -- ordem alta forcada em poucas
    # direcoes, mal-posto de proposito) serve como diagnostico extra: isola
    # se a diferenca de recall/energia contra os metodos de reconstrucao
    # vem so da ordem menor (artefato de escolha de parametro) ou de fato
    # da falta de informacao angular real (nesse caso, esperado que o
    # ajuste fique instavel/hallucine picos em vez de melhorar). Ex.: pra
    # n16 (max_order_for_n_directions(16)=4), comparar com a mesma ordem 8
    # usada pelo shell completo:
    #   SH_ORDER_SUBSAMPLED_ONLY=8 SUBSAMPLED_ONLY=1 \
    #     sbatch slurm/11_peak_confusion_by_roi.sh <work_dir> 1000 16
    if [[ -n "${SH_ORDER_SUBSAMPLED_ONLY:-}" ]]; then
        SUBSAMPLED_ONLY_FLAG+=(--sh-order-subsampled-only "$SH_ORDER_SUBSAMPLED_ONLY")
        echo "SH_ORDER_SUBSAMPLED_ONLY=$SH_ORDER_SUBSAMPLED_ONLY -- forcando essa ordem no subsampled_only (em vez da ordem 'honesta' automatica)"
    fi
fi

ROI_FLAG=()
if [[ -n "${ROI_TRACTS:-}" ]]; then
    ROI_FLAG=(--roi-tracts "$ROI_TRACTS")
    echo "ROI_TRACTS=$ROI_TRACTS"
fi

SUBJECTS_FLAG=()
if [[ -n "${SUBJECTS:-}" ]]; then
    SUBJECTS_FLAG=(--subjects "$SUBJECTS")
    echo "SUBJECTS=$SUBJECTS -- restringindo a esse(s) sujeito(s)"
fi

PEAK_MATCH_THRESHOLD_DEG="${PEAK_MATCH_THRESHOLD_DEG:-25.0}"

python scripts/11_peak_confusion_by_roi.py \
    --manifest "$WORK_DIR/manifest.csv" \
    "${BASELINE_FLAG[@]}" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
    --peak-match-threshold-deg "$PEAK_MATCH_THRESHOLD_DEG" \
    --out-csv "$WORK_DIR/metrics/peak_confusion_shell${SHELL_B%.*}_n${N_LEVEL}.csv" \
    "${EXTRA_METHOD_FLAGS[@]}" "${SUBSAMPLED_ONLY_FLAG[@]}" "${ROI_FLAG[@]}" "${SUBJECTS_FLAG[@]}"