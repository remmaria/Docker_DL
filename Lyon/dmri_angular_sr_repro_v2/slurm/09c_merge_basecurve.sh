#!/bin/bash
#SBATCH --job-name=basecurve_merge
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/basecurve_merge.%J.err
#SBATCH --output=logs/basecurve_merge.%J.out
#
# Junta os CSVs por-shard gerados por `sbatch --array=1-N slurm/09b_build_basecurve.sh`
# (scripts/09c_merge_basecurve.py) num CSV final + tabela resumida. Rode DEPOIS
# que todas as N tasks do array ja tiverem terminado com sucesso.
#
# Uso:
#   sbatch slurm/09c_merge_basecurve.sh <work_dir> <shell_b> <n_shards> [out_csv_override]
# O 4o argumento (opcional) tem que ser o MESMO caminho passado como
# --out-csv=PATH pro 09b_build_basecurve.sh correspondente, senao o merge
# procura os shards no lugar errado.
#
# Encadeando automaticamente com o array (nao precisa esperar manualmente):
#   ARRAY_JOB=$(sbatch --parsable --array=1-8 slurm/09b_build_basecurve.sh <work_dir> 1000)
#   sbatch --dependency=afterok:$ARRAY_JOB slurm/09c_merge_basecurve.sh <work_dir> 1000 8

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 09c_merge_basecurve.sh <work_dir> <shell_b> <n_shards> [out_csv_override]}"
SHELL_B="${2:?uso: sbatch 09c_merge_basecurve.sh <work_dir> <shell_b> <n_shards> [out_csv_override]}"
N_SHARDS="${3:?uso: sbatch 09c_merge_basecurve.sh <work_dir> <shell_b> <n_shards> [out_csv_override]}"
POS_OUT_CSV="${4:-}"

if [[ -n "$POS_OUT_CSV" ]]; then
    OUT_CSV="$POS_OUT_CSV"
    echo "out_csv_override (arg 4) = $OUT_CSV -- procurando shards/gravando resultado nesse caminho"
elif [[ -n "${OUT_CSV_OVERRIDE:-}" ]]; then
    OUT_CSV="$OUT_CSV_OVERRIDE"
    echo "OUT_CSV_OVERRIDE=$OUT_CSV_OVERRIDE -- procurando shards/gravando resultado nesse caminho"
else
    OUT_CSV="$WORK_DIR/basecurve_metrics_shell${SHELL_B%.*}.csv"
fi

source "./00_env_common.sh"

python scripts/09c_merge_basecurve.py \
    --out-csv "$OUT_CSV" \
    --shard-count "$N_SHARDS"