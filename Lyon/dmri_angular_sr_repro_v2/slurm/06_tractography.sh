#!/bin/bash
#SBATCH --job-name=dmri_tracto
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/tracto.%A_%a.err
#SBATCH --output=logs/tracto.%A_%a.out
#
# Tratografia opcional (etapa 8) via MRtrix3. So CPU. Requer o modulo do
# MRtrix3 do seu cluster -- ajuste a linha `module load mrtrix3/...` abaixo
# para o nome exato disponivel (rode `module avail mrtrix` no login node
# para descobrir). Se MRtrix3 estiver dentro do conda env em vez de modulo,
# apague essa linha.
#
# Uso: sbatch --array=1-N slurm/06_tractography.sh <work_dir>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 06_tractography.sh <work_dir> [shell_b n_level]}"

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

echo "Tratografia shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

# module load mrtrix3/3.0.4   # <-- ajuste/descomente conforme seu cluster
if ! command -v dwi2response &> /dev/null; then
    echo "Erro: MRtrix3 (dwi2response) nao encontrado no PATH. Carregue o modulo "
    echo "certo (module avail mrtrix) ou instale no env conda antes de rodar."
    exit 1
fi

python scripts/08_downstream_tractography.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --baseline-dir "$WORK_DIR/baseline_recon" \
    --rcae-dir "$WORK_DIR/rcae_recon" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$WORK_DIR/tractography" \
    --n-streamlines 200000
