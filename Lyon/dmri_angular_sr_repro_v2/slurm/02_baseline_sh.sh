#!/bin/bash
#SBATCH --job-name=dmri_scheme
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/scheme.%J.err
#SBATCH --output=logs/scheme.%J.out
#
# So a geracao do esquema de subamostragem (etapa 2) -- gera um .npz leve
# (indices, poucos KB) por sujeito, cobrindo TODOS os niveis de uma vez.
# Isso e barato e roda uma vez so; nao precisa ser por combo.
#
# A reconstrucao do baseline SH (o que gera os recon_target.nii.gz pesados)
# NAO esta mais aqui -- ver 02b_baseline_reconstruct.sh, que roda por
# combo (array), pra nao acumular todos os 30 combos em disco de uma vez.
#
# Uso: sbatch 02_baseline_sh.sh /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02_baseline_sh.sh <work_dir>}"

source "./00_env_common.sh"

python scripts/02_subsample_directions.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --out-dir "$WORK_DIR/subsampling" \
    --levels 6 10 16 20 24 32 48 54
# uniao de todos os niveis usados em qualquer experimento (ver configs/experiments.tsv);
# niveis maiores que as direcoes disponiveis numa shell/sujeito sao pulados
# automaticamente (aviso no log), sem problema.
