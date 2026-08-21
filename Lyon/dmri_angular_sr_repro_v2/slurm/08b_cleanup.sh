#!/bin/bash
#SBATCH --job-name=dmri_cleanup
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/cleanup.%A_%a.err
#SBATCH --output=logs/cleanup.%A_%a.out
#
# Limpeza manual (etapa 10): apaga recon_target.nii.gz (baseline + RCAE) de
# um combo, depois que TODAS as avaliacoes que voce quer para ele (06, 07 e,
# se usou, 08) ja rodaram. Preferencialmente use CLEANUP_AFTER=1 direto no
# job 05 (mais simples); use este script separado se rodou 08 (tratografia)
# depois do 05 e so agora esta pronto pra liberar espaco. E leve, nao
# precisa de GPU, mas mantive a mesma partition/conta por serem os valores
# que eu sei que funcionam no seu cluster.
#
# Uso:
#   sbatch --array=1-N slurm/08b_cleanup.sh <work_dir>
#   sbatch slurm/08b_cleanup.sh <work_dir> <shell_b> <n_level>   # sem array

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 08b_cleanup.sh <work_dir> [shell_b n_level]}"

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

echo "Limpando reconstrucoes de shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

python scripts/10_cleanup_reconstructions.py \
    --work-dir "$WORK_DIR" --shell-b "$SHELL_B" --n-level "$N_LEVEL"
