#!/bin/bash
#SBATCH --job-name=dmri_scheme
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
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
# Uso (canonico, sem overrides):
#   sbatch 02_baseline_sh.sh <work_dir>
#
# Uso (esquema ALTERNATIVO -- ex.: mais niveis, so pra uma investigacao
# pontual tipo a curva erro-vs-N-direcoes -- sem sobrescrever o esquema
# canonico usado por RCAE/RRIN):
#   sbatch 02_baseline_sh.sh <work_dir> <out_dir_ou_-> [level1 level2 ...]
# O 2o argumento e o diretorio de saida alternativo (caminho ABSOLUTO); passe
# "-" nesse lugar se quiser manter o out-dir canonico mas so mudar os
# niveis. Se nenhum nivel for passado depois dele, usa o default (canonico).
# Ex. (curva mais fina, pasta separada, so shell 1000):
#   sbatch 02_baseline_sh.sh <work_dir> $WORK_DIR/subsampling_basecurve \
#       6 10 12 16 20 24 28 32 36 40 44 48 52 56 60
#
# ARGUMENTOS POSICIONAIS EM VEZ DE VARIAVEIS DE AMBIENTE: em alguns clusters
# SLURM, "VAR=valor sbatch ..." nao propaga a variavel pro ambiente do job de
# verdade (depende de configuracao do site) -- o script cairia no default
# canonico SILENCIOSAMENTE nesse caso. Argumento posicional nunca tem esse
# problema. OUT_DIR_OVERRIDE/LEVELS_OVERRIDE (variaveis de ambiente) ainda
# sao aceitas como fallback, mas os argumentos posicionais acima, se
# passados, tem prioridade.

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02_baseline_sh.sh <work_dir> [out_dir_ou_-] [levels...]}"

POS_OUT_DIR="${2:-}"
if [[ $# -ge 2 ]]; then
    shift 2
else
    shift 1
fi
POS_LEVELS=("$@")

source "./00_env_common.sh"

if [[ -n "$POS_OUT_DIR" && "$POS_OUT_DIR" != "-" ]]; then
    OUT_DIR="$POS_OUT_DIR"
    echo "out_dir (arg 2) = $OUT_DIR -- gravando esquema nesse caminho (nao mexe no canonico)"
elif [[ -n "${OUT_DIR_OVERRIDE:-}" ]]; then
    OUT_DIR="$OUT_DIR_OVERRIDE"
    echo "OUT_DIR_OVERRIDE=$OUT_DIR_OVERRIDE -- gravando esquema nesse caminho (nao mexe no canonico)"
else
    OUT_DIR="$WORK_DIR/subsampling"
fi

if [[ ${#POS_LEVELS[@]} -gt 0 ]]; then
    LEVELS=("${POS_LEVELS[@]}")
    echo "niveis (argumentos posicionais) = ${LEVELS[*]}"
elif [[ -n "${LEVELS_OVERRIDE:-}" ]]; then
    read -ra LEVELS <<< "$LEVELS_OVERRIDE"
    echo "LEVELS_OVERRIDE='$LEVELS_OVERRIDE' -- usando essa lista em vez da canonica"
else
    LEVELS=(6 10 16 20 24 32 48 54)
fi

python scripts/02_subsample_directions.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --out-dir "$OUT_DIR" \
    --levels "${LEVELS[@]}"
# uniao de todos os niveis usados em qualquer experimento (ver configs/experiments.tsv);
# niveis maiores que as direcoes disponiveis numa shell/sujeito sao pulados
# automaticamente (aviso no log), sem problema.