#!/bin/bash
#SBATCH --job-name=dmri_prepare
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/prepare.%J.err
#SBATCH --output=logs/prepare.%J.out
#
# Etapas 1 e 1b: manifesto + relatorio de disponibilidade de shells.
# Nao precisa de GPU (nao pedimos --gres=gpu abaixo), mas mantive a mesma
# partition/account do seu exemplo de treino por serem os unicos valores
# que eu sei que funcionam no seu cluster. Se existir uma partition CPU-only
# mais barata/rapida de agendar, troque --partition aqui.
#
# Uso: sbatch 01_prepare_data.sh /caminho/data_root /caminho/work_dir [name_suffix]
# Ex.: sbatch 01_prepare_data.sh /ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias work_dir_cpu _geomcorr

set -euo pipefail
mkdir -p logs

DATA_ROOT="${1:?uso: sbatch 01_prepare_data.sh <data_root> <work_dir> [name_suffix]}"
WORK_DIR="${2:?uso: sbatch 01_prepare_data.sh <data_root> <work_dir> [name_suffix]}"
NAME_SUFFIX="${3:-_geomcorr}"

source "./00_env_common.sh"

python scripts/01_prepare_data.py --data-root "$DATA_ROOT" --out-dir "$WORK_DIR" \
    --name-suffix "$NAME_SUFFIX"

python scripts/01b_shell_availability_report.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --candidate-bvalues 500 700 750 1000 1500 2000 \
    --out-csv "$WORK_DIR/shell_availability.csv"
