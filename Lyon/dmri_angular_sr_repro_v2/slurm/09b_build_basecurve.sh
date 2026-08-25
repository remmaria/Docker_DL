#!/bin/bash
#SBATCH --job-name=basecurve
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/basecurve.%A.err
#SBATCH --output=logs/basecurve.%A.out
#
# Monta a curva "erro do baseline_sh vs. n_level" (scripts/09b_build_basecurve.py),
# restringindo a amostra a sujeitos com uma contagem FIXA de direcoes na shell
# pedida (default do proprio script: o maximo observado -- ver docstring).
# So CPU.
#
# Por padrao le de <work_dir>/baseline_recon (o baseline CANONICO -- desde que
# a lista de niveis do work_dir seja a mesma passada aqui, o que ja e o caso
# pro work_dir novo criado com a lista fina 6 10 16 20 24 32 48 54). Pra ler
# de uma pasta alternativa (ex.: o work_dir_sh antigo, que usava uma pasta
# separada pra nao mexer no esquema canonico dele), passe
# BASELINE_DIR_OVERRIDE=<caminho absoluto> antes do sbatch.
#
# Uso:
#   sbatch slurm/09b_build_basecurve.sh <work_dir> <shell_b> [level1 level2 ...]
#
# Se nenhum nivel for passado, usa o default abaixo (6 10 16 20 24 32 48 54).
#
# Ex. (work_dir novo, le do baseline_recon canonico):
#   sbatch slurm/09b_build_basecurve.sh /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir 1000
#
# Ex. (work_dir_sh antigo, pasta separada):
#   BASELINE_DIR_OVERRIDE=/ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir_sh/baseline_recon_basecurve \
#     sbatch slurm/09b_build_basecurve.sh /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir_sh 1000 6 10 16 20 24 32 48 54

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 09b_build_basecurve.sh <work_dir> <shell_b> [levels...]}"
SHELL_B="${2:?uso: sbatch 09b_build_basecurve.sh <work_dir> <shell_b> [levels...]}"
shift 2
LEVELS=("$@")
if [[ ${#LEVELS[@]} -eq 0 ]]; then
    LEVELS=(6 10 16 20 24 32 48 54)
    echo "Nenhum nivel passado -- usando default: ${LEVELS[*]}"
fi

BASELINE_DIR="${BASELINE_DIR_OVERRIDE:-$WORK_DIR/baseline_recon}"
echo "Montando basecurve para shell_b=$SHELL_B, niveis=${LEVELS[*]} (lendo de $BASELINE_DIR)"

source "./00_env_common.sh"

python scripts/09b_build_basecurve.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --baseline-dir "$BASELINE_DIR" \
    --shell-b "$SHELL_B" \
    --levels "${LEVELS[@]}" \
    --out-csv "$WORK_DIR/basecurve_metrics_shell${SHELL_B%.*}.csv"