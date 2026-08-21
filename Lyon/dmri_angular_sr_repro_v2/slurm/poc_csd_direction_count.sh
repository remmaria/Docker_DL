#!/bin/bash
#SBATCH --job-name=dmri_poc_csd
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/poc_csd.%A_%a.err
#SBATCH --output=logs/poc_csd.%A_%a.out
#
# Prova de conceito: CSD em poucas direcoes vs. aquisicao densa vs.
# preenchido por SH (e por RCAE, se ja tiver reconstrucao) -- ver
# scripts/poc_csd_direction_count.py pro racional completo. CSD x3-4
# condicoes x (1 + N ROIs) por sujeito e caro (~10min/sujeito observado) --
# sharding por sujeito via SLURM array e o modo recomendado pra mais de
# uns 3-4 sujeitos (mesmo padrao de slurm/05_evaluate_and_downstream.sh e
# slurm/crossing_fiber_stratified.sh).
#
# Uso SEM array (job unico, sequencial -- ok pra poucos sujeitos):
#   sbatch slurm/poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> [n_subjects]
#
# Uso COM array (sharding por sujeito -- cada task processa so uma fatia
# dos sujeitos amostrados, todas em paralelo):
#   sbatch --array=1-8 slurm/poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> 8
#   python scripts/merge_shard_csvs.py --dir <work_dir>/metrics
# (o N do --array=1-N PRECISA bater com [n_subjects] -- 1 task por sujeito
# amostrado. Se pedir --array=1-8 com n_subjects=20, cada task ainda pega
# 1/8 dos 20 sujeitos amostrados, so nao e "1 task = 1 sujeito" nesse caso.)
#
# Resubmissao parcial de tasks que falharam (preserva os IDs originais --
# ver comentario extenso em slurm/05_evaluate_and_downstream.sh pro
# raciocinio completo de por que SHARD_INDEX usa sempre base 1):
#   SHARD_COUNT_OVERRIDE=<N_original> sbatch --array=<ids_que_faltam> slurm/poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> <n_subjects_original>
#
# POC_SUBJECTS="tag1,tag2" (variavel de ambiente) roda em sujeitos
# ESPECIFICOS em vez de uma amostra aleatoria -- use os MESMOS 'tag' que
# voce passou em RECON_SUBJECTS pra 04_reconstruct_rcae.py, assim a
# condicao preenchido_rcae aparece garantido (nao depende de --seed/
# n_subjects reamostrarem os mesmos sujeitos por coincidencia). Ignora
# [n_subjects] quando usado (mas o sharding via --array ainda funciona
# normalmente sobre a lista de --subjects). Ex.:
#   POC_SUBJECTS="20170920171326_616_20170920171326_616" sbatch slurm/poc_csd_direction_count.sh <work_dir> 1000 10
#
# ROI_TRACTS="FX,CGC,CGH,UF" (variavel de ambiente) TAMBEM calcula as
# metricas restritas a esses tratos JHU-ICBM (alem do cerebro inteiro,
# sempre calculado como roi='whole_mask') -- mesma convencao/arquivo do
# ROI_TRACTS em 05_evaluate_and_downstream.sh. Nao reajusta CSD por ROI
# (barato, so re-mascara o que ja foi ajustado). Ex.:
#   ROI_TRACTS="FX,CGC,CGH,UF" sbatch slurm/poc_csd_direction_count.sh <work_dir> 1000 10

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> [n_subjects]}"
SHELL_B="${2:?uso: sbatch poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> [n_subjects]}"
N_LEVEL="${3:?uso: sbatch poc_csd_direction_count.sh <work_dir> <shell_b> <n_level> [n_subjects]}"
N_SUBJECTS="${4:-8}"

SHARD_INDEX=0
SHARD_COUNT=1

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # ver comentario extenso equivalente em slurm/05_evaluate_and_downstream.sh:
    # SHARD_INDEX usa SEMPRE base 1 (SLURM_ARRAY_TASK_ID - 1), NUNCA
    # SLURM_ARRAY_TASK_MIN da submissao atual -- sobrevive a resubmissoes
    # parciais que preservam os IDs originais das tasks que falharam.
    SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
    if [[ -n "${SHARD_COUNT_OVERRIDE:-}" ]]; then
        SHARD_COUNT="$SHARD_COUNT_OVERRIDE"
    elif [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${SLURM_ARRAY_TASK_MIN:-1}" == "1" ]]; then
        SHARD_COUNT="$SLURM_ARRAY_TASK_COUNT"
    else
        SHARD_COUNT=$((${SLURM_ARRAY_TASK_MAX:-1}))
    fi
    echo "[shard] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID MIN=${SLURM_ARRAY_TASK_MIN:-?} MAX=${SLURM_ARRAY_TASK_MAX:-?} -> SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT (confira: numa resubmissao parcial, SHARD_COUNT PRECISA bater com o total original -- use SHARD_COUNT_OVERRIDE=<N> se nao bater)"
fi

echo "Prova de conceito CSD para shell_b=$SHELL_B, n_level=$N_LEVEL, n_subjects=$N_SUBJECTS -- shard $SHARD_INDEX/$SHARD_COUNT"

source "./00_env_common.sh"

SUBJECTS_FLAG=()
if [[ -n "${POC_SUBJECTS:-}" ]]; then
    SUBJECTS_FLAG=(--subjects "$POC_SUBJECTS")
    echo "POC_SUBJECTS=$POC_SUBJECTS -- restringindo a esse(s) sujeito(s) (ignorando n_subjects=$N_SUBJECTS)"
fi

ROI_FLAG=()
if [[ -n "${ROI_TRACTS:-}" ]]; then
    ROI_FLAG=(--roi-tracts "$ROI_TRACTS")
    echo "ROI_TRACTS=$ROI_TRACTS -- metricas tambem calculadas restritas a esses tratos"
fi

python scripts/poc_csd_direction_count.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --baseline-dir "$WORK_DIR/baseline_recon" \
    --rcae-dir "$WORK_DIR/rcae_recon" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --split test --n-subjects "$N_SUBJECTS" --seed 0 \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
    "${SUBJECTS_FLAG[@]}" "${ROI_FLAG[@]}" \
    --out-csv "$WORK_DIR/metrics/poc_csd_shell${SHELL_B%.*}_n${N_LEVEL}.csv"