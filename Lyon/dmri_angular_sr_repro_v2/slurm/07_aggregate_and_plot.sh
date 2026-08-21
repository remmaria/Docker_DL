#!/bin/bash
#SBATCH --job-name=dmri_aggregate
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/aggregate.%J.err
#SBATCH --output=logs/aggregate.%J.out
#
# Etapa final (9): agrega todos os CSVs de metricas em tabelas/figuras.
# Roda depois que TODOS os jobs de avaliacao (05) e, se usados, de
# tratografia (06) tiverem terminado -- sem dependencia de array, e rapido.
#
# Uso: sbatch 07_aggregate_and_plot.sh <work_dir>
#
# Dica: para encadear automaticamente depois dos jobs de avaliacao, submeta
# com dependencia, ex.:
#   sbatch --dependency=afterok:<jobid_avaliacao> slurm/07_aggregate_and_plot.sh <work_dir>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 07_aggregate_and_plot.sh <work_dir>}"

source "./00_env_common.sh"

python scripts/09_aggregate_and_plot.py \
    --metrics-dir "$WORK_DIR/metrics" \
    --downstream-dir "$WORK_DIR/downstream" \
    --tractography-dir "$WORK_DIR/tractography" \
    --out-dir "$WORK_DIR/figures"
