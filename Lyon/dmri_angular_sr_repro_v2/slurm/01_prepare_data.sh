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
# Ex.: sbatch 01_prepare_data.sh /ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias work_dir _geomcorr
#
# DEMO_TSV_NAME=... (variavel de ambiente, default "info.tsv") -- TSV opcional
# de demograficos/acquisicao (colunas SessionID/PatientSex/AcquisitionDate/
# PatientAge/Study). DOIS MODOS -- ver docstring de scripts/01_prepare_data.py:
#   - so' existe UM TSV pra tudo -> passe o CAMINHO COMPLETO, ex.:
#       DEMO_TSV_NAME=/ix1/tibrahim/rmm270/DATA/DWIs/7TBRP/info.tsv sbatch ...
#     (casado por SessionID em todos os sujeitos, independente de subestudo)
#   - um TSV por subestudo -> deixe so' o nome do arquivo (default "info.tsv"),
#     procurado em <data_root>/<estudo>/<DEMO_TSV_NAME> pra cada subestudo.
# Sem TSV nenhum, comportamento identico a antes (campos ficam vazios).
#
# STRATIFY_SEX/STRATIFY_STUDY (variaveis de ambiente, default "1" = ligado):
# o split treino/val/teste estratifica por protocol+sexo+estudo por padrao
# (ver utils.manifest.assign_splits) -- passe STRATIFY_SEX=0 e/ou
# STRATIFY_STUDY=0 pra desligar uma ou ambas as estratificacoes extras
# (mantendo so a estratificacao original por protocol). MIN_STRATUM_SIZE
# (default 8): grupos menores que isso recaem pra uma chave mais grosseira
# (ver docstring de assign_splits) em vez de arredondar val/teste pra 0.

set -euo pipefail
mkdir -p logs

DATA_ROOT="${1:?uso: sbatch 01_prepare_data.sh <data_root> <work_dir> [name_suffix]}"
WORK_DIR="${2:?uso: sbatch 01_prepare_data.sh <data_root> <work_dir> [name_suffix]}"
NAME_SUFFIX="${3:-_geomcorr}"
DEMO_TSV_NAME="${DEMO_TSV_NAME:-info.tsv}"
STRATIFY_SEX="${STRATIFY_SEX:-1}"
STRATIFY_STUDY="${STRATIFY_STUDY:-1}"
MIN_STRATUM_SIZE="${MIN_STRATUM_SIZE:-8}"

source "./00_env_common.sh"

STRATIFY_FLAGS=()
if [[ "$STRATIFY_SEX" == "0" ]]; then
    STRATIFY_FLAGS+=(--no-stratify-sex)
fi
if [[ "$STRATIFY_STUDY" == "0" ]]; then
    STRATIFY_FLAGS+=(--no-stratify-study)
fi
echo "STRATIFY_SEX=$STRATIFY_SEX STRATIFY_STUDY=$STRATIFY_STUDY MIN_STRATUM_SIZE=$MIN_STRATUM_SIZE"

python scripts/01_prepare_data.py --data-root "$DATA_ROOT" --out-dir "$WORK_DIR" \
    --name-suffix "$NAME_SUFFIX" --demo-tsv-name "$DEMO_TSV_NAME" \
    --min-stratum-size "$MIN_STRATUM_SIZE" "${STRATIFY_FLAGS[@]}"

python scripts/01b_shell_availability_report.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --candidate-bvalues 500 700 750 1000 1500 2000 \
    --out-csv "$WORK_DIR/shell_availability.csv"