#!/bin/bash
#SBATCH --job-name=dmri_check_ensemble
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-00:05:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/check_ensemble.%J.err
#SBATCH --output=logs/check_ensemble.%J.out
#
# Diagnostico rapido do feixe "ensemble em estrela" (ver addendum 2026-08-27
# secao 14 / scripts/07c_check_ensemble_between.py) -- confirma numericamente
# se um par do feixe e' between=True (deveria passar perto do alvo na
# figura de scripts/07_visualize_triplet.py) ou between=False (extrapolacao,
# fallback -- nao deveria passar perto, e' esperado). So numpy, sem
# GPU/torch -- roda em segundos, mas via sbatch porque o login node nao
# permite rodar python direto.
#
# Uso (mesmos argumentos posicionais de slurm/07_visualize_triplet.sh):
#   sbatch slurm/07c_check_ensemble_between.sh <work_dir> <shell_b> <n_level>
#
# SUBJECT=<tag> / EXAMPLE={typical,best,worst} (variaveis de ambiente,
# opcionais, mesmo significado de slurm/07_visualize_triplet.sh) -- usa a
# MESMA trinca que a figura escolheria, pra comparar exatamente o mesmo
# caso.
#   SUBJECT=20170417094841_802780_20170417094841_802780 \
#     sbatch slurm/07c_check_ensemble_between.sh <work_dir> 1000 16

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 07c_check_ensemble_between.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 07c_check_ensemble_between.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 07c_check_ensemble_between.sh <work_dir> <shell_b> <n_level>}"

source "./00_env_common.sh"

SUBJECT_FLAG=()
if [[ -n "${SUBJECT:-}" ]]; then
    SUBJECT_FLAG=(--subject "$SUBJECT")
    echo "SUBJECT=$SUBJECT -- fixando o sujeito"
fi

EXAMPLE="${EXAMPLE:-typical}"
echo "EXAMPLE=$EXAMPLE -- criterio de selecao da trinca (mesmo de 07_visualize_triplet.py)"

python scripts/07c_check_ensemble_between.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --example "$EXAMPLE" \
    "${SUBJECT_FLAG[@]}"