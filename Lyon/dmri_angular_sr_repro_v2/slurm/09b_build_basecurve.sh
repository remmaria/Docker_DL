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
#SBATCH --error=logs/basecurve.%A_%a.err
#SBATCH --output=logs/basecurve.%A_%a.out
#
# Monta a curva "erro do baseline_sh vs. n_level" (scripts/09b_build_basecurve.py),
# restringindo a amostra a sujeitos com uma contagem FIXA de direcoes na shell
# pedida (default do proprio script: o maximo observado -- ver docstring).
# So CPU.
#
# Por padrao le de <work_dir>/baseline_recon (o baseline CANONICO). Pra ler
# de uma pasta alternativa (ex.: <work_dir>/baseline_recon_basecurve, gerado
# pra uma curva com niveis diferentes dos canonicos), passe --baseline-dir=PATH
# como um dos argumentos (ver Uso abaixo) -- pode vir em qualquer posicao,
# junto com os niveis.
#
# --out-csv=PATH -- muda onde o CSV final (e os .shard*.csv/.summary.csv
# derivados dele) sao gravados. Default: <work_dir>/basecurve_metrics_shell<B>.csv.
# Use isso sempre que rodar uma lista de niveis DIFERENTE da ja usada antes
# pro mesmo work_dir/shell, pra nao sobrescrever um resultado anterior.
#
# ARGUMENTOS EM VEZ DE VARIAVEIS DE AMBIENTE: em alguns clusters SLURM,
# "VAR=valor sbatch ..." (mesmo com --export=ALL,VAR=valor) nao propaga a
# variavel pro ambiente do job de verdade (depende de config do site) -- o
# script cairia no default silenciosamente. Os flags --baseline-dir=/--out-csv=
# nunca tem esse problema, por isso sao o jeito recomendado agora.
# BASELINE_DIR_OVERRIDE/OUT_CSV_OVERRIDE (variaveis de ambiente) ainda sao
# aceitas como fallback, mas os flags acima, se passados, tem prioridade.
#
# SHARDING (--array) -- acelera dividindo os sujeitos da amostra fixada
# entre varias tasks em paralelo, em vez de um job sequencial so processando
# todos:
#   sbatch --array=1-8 slurm/09b_build_basecurve.sh <work_dir> <shell_b> [levels...] [--baseline-dir=PATH] [--out-csv=PATH]
# Cada task processa 1/8 dos sujeitos e grava seu proprio CSV parcial
# (<out-csv-stem>.shard<i>.csv). DEPOIS que TODAS as tasks do array
# terminarem, rode o merge pra juntar tudo:
#   sbatch --dependency=afterok:<jobid_do_array> slurm/09c_merge_basecurve.sh <work_dir> <shell_b> 8 [out_csv_override]
# (troque 8 pelo N que voce usou no --array). Ou, se preferir nao usar
# --dependency, so rode o merge manualmente depois de conferir nos logs que
# todas as tasks terminaram com sucesso.
#
# Sem --array (SLURM_ARRAY_TASK_ID vazio), roda tudo num job so, sequencial,
# sem sharding -- igual ao comportamento antigo.
#
# Uso (sem sharding, esquema canonico):
#   sbatch slurm/09b_build_basecurve.sh <work_dir> <shell_b> [level1 level2 ...]
#
# Se nenhum nivel for passado, usa o default abaixo (6 10 16 20 24 32 48 54).
#
# Ex. (work_dir novo, le do baseline_recon canonico, com sharding):
#   sbatch --array=1-8 slurm/09b_build_basecurve.sh /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir 1000
#
# Ex. (curva fina, pasta alternativa, CSV separado, com sharding):
#   sbatch --array=1-8 slurm/09b_build_basecurve.sh /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir 1000 \
#       6 10 12 16 20 24 28 32 36 40 44 48 52 56 60 \
#       --baseline-dir=/ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir/baseline_recon_basecurve \
#       --out-csv=/ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir/basecurve_metrics_shell1000_fine.csv

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 09b_build_basecurve.sh <work_dir> <shell_b> [levels...] [--baseline-dir=PATH] [--out-csv=PATH]}"
SHELL_B="${2:?uso: sbatch 09b_build_basecurve.sh <work_dir> <shell_b> [levels...] [--baseline-dir=PATH] [--out-csv=PATH]}"
shift 2

BASELINE_DIR_ARG=""
OUT_CSV_ARG=""
LEVELS=()
for a in "$@"; do
    case "$a" in
        --baseline-dir=*) BASELINE_DIR_ARG="${a#--baseline-dir=}" ;;
        --out-csv=*) OUT_CSV_ARG="${a#--out-csv=}" ;;
        *) LEVELS+=("$a") ;;
    esac
done
if [[ ${#LEVELS[@]} -eq 0 ]]; then
    LEVELS=(6 10 16 20 24 32 48 54)
    echo "Nenhum nivel passado -- usando default: ${LEVELS[*]}"
fi

SHARD_INDEX=0
SHARD_COUNT=1
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    # base-1 sempre (nao SLURM_ARRAY_TASK_MIN desta submissao) -- ver
    # comentario equivalente em slurm/02b_baseline_reconstruct.sh e
    # slurm/05_evaluate_and_downstream.sh sobre resubmissoes parciais.
    SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
    if [[ -n "${SHARD_COUNT_OVERRIDE:-}" ]]; then
        SHARD_COUNT="$SHARD_COUNT_OVERRIDE"
    elif [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${SLURM_ARRAY_TASK_MIN:-1}" == "1" ]]; then
        SHARD_COUNT="$SLURM_ARRAY_TASK_COUNT"
    else
        SHARD_COUNT=$((${SLURM_ARRAY_TASK_MAX:-1}))
    fi
    echo "[shard] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID MIN=${SLURM_ARRAY_TASK_MIN:-?} MAX=${SLURM_ARRAY_TASK_MAX:-?} -> SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT (numa resubmissao parcial, confira que SHARD_COUNT bate com o total original -- use SHARD_COUNT_OVERRIDE=<N> se nao bater)"
fi

if [[ -n "$BASELINE_DIR_ARG" ]]; then
    BASELINE_DIR="$BASELINE_DIR_ARG"
    echo "--baseline-dir=$BASELINE_DIR (argumento) -- lendo reconstrucao desse caminho"
elif [[ -n "${BASELINE_DIR_OVERRIDE:-}" ]]; then
    BASELINE_DIR="$BASELINE_DIR_OVERRIDE"
    echo "BASELINE_DIR_OVERRIDE=$BASELINE_DIR_OVERRIDE -- lendo reconstrucao desse caminho"
else
    BASELINE_DIR="$WORK_DIR/baseline_recon"
fi

if [[ -n "$OUT_CSV_ARG" ]]; then
    OUT_CSV="$OUT_CSV_ARG"
    echo "--out-csv=$OUT_CSV (argumento) -- gravando resultado nesse caminho"
elif [[ -n "${OUT_CSV_OVERRIDE:-}" ]]; then
    OUT_CSV="$OUT_CSV_OVERRIDE"
    echo "OUT_CSV_OVERRIDE=$OUT_CSV_OVERRIDE -- gravando resultado nesse caminho"
else
    OUT_CSV="$WORK_DIR/basecurve_metrics_shell${SHELL_B%.*}.csv"
fi

echo "Montando basecurve para shell_b=$SHELL_B, niveis=${LEVELS[*]} (lendo de $BASELINE_DIR, gravando em $OUT_CSV) -- shard $SHARD_INDEX/$SHARD_COUNT"

source "./00_env_common.sh"

python scripts/09b_build_basecurve.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --baseline-dir "$BASELINE_DIR" \
    --shell-b "$SHELL_B" \
    --levels "${LEVELS[@]}" \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
    --out-csv "$OUT_CSV"