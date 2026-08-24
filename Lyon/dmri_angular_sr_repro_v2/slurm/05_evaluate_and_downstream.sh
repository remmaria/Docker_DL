#!/bin/bash
#SBATCH --job-name=dmri_eval
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/eval.%A_%a.err
#SBATCH --output=logs/eval.%A_%a.out
#
# Metricas de sinal (etapa 6) + downstream DTI/NODDI (etapa 7). So CPU
# (nao pedimos --gres=gpu). Mesmo esquema de array das etapas anteriores.
# Adicione --run-noddi (variavel RUN_NODDI=1) se tiver o pacote `amico`
# instalado no env -- so tem efeito para sujeitos com >=2 shells.
#
# ROI_TRACTS="FX,CGC,CGH,UF" (variavel de ambiente, lista separada por
# virgula) restringe as metricas de DTI/NODDI da etapa 7 tambem aos tratos
# JHU-ICBM pedidos (alem da mascara inteira, que continua sendo calculada
# sempre) -- ver utils/masking.py:load_roi_masks e scripts/07_downstream_dti_noddi.py.
# Cada trato precisa de um arquivo
# "JHU-ICBM-labels-1mm_warped_s_<TRATO>_<R/L>.nii.gz" (ou sem sufixo de
# lado, ex. FX) na mesma pasta do dwi de cada sujeito. Sem essa variavel,
# comportamento antigo inalterado (so 'whole_mask').
#
# CLEANUP_AFTER=1 apaga os recon_target.nii.gz (baseline + RCAE) desse
# combo logo depois de calcular as metricas -- eles ja nao servem pra mais
# nada nesse ponto (nem treino nem reconstrucao leem de volta esses
# arquivos). So NAO ative se ainda for rodar a etapa 8 (tratografia) pra
# esse combo -- ela tambem precisa do recon_target.nii.gz.
#
# Uso (array = 1 combo shell/nivel por task, ve-lo de configs/experiments.tsv):
#   sbatch --array=1-N slurm/05_evaluate_and_downstream.sh <work_dir>
#   RUN_NODDI=1 sbatch --array=1-N slurm/05_evaluate_and_downstream.sh <work_dir>
#   CLEANUP_AFTER=1 sbatch --array=1-N slurm/05_evaluate_and_downstream.sh <work_dir>
#
# Uso (1 combo especifico, sem array -- roda TODOS os sujeitos do split num
# job so, sequencial):
#   sbatch slurm/05_evaluate_and_downstream.sh <work_dir> <shell_b> <n_level>
#
# Uso (1 combo especifico, MAS paralelizando por SUJEITO via array -- cada
# task processa so uma fatia dos sujeitos do split, todas em paralelo, e no
# final voce junta os CSVs com scripts/merge_shard_csvs.py):
#   sbatch --array=1-20 slurm/05_evaluate_and_downstream.sh <work_dir> <shell_b> <n_level>
#   python scripts/merge_shard_csvs.py --dir <work_dir>/metrics
#   python scripts/merge_shard_csvs.py --dir <work_dir>/downstream
#
# ATENCAO: passar <shell_b> <n_level> junto com --array=1-N SEM esse modo de
# shard fazia cada task do array rodar o combo inteiro (todos os sujeitos)
# de forma DUPLICADA -- N copias identicas competindo pelo mesmo storage e
# escrevendo por cima do mesmo --out-csv. Agora, se voce passar shell_b/
# n_level E estiver dentro de um array, o script entende que o array e
# sharding por sujeito (usa --shard-index/--shard-count nos scripts de
# Python) em vez de repetir o combo inteiro em cada task.
#
# RECON_TAG="algumnome" (variavel de ambiente, MESMO nome usado em
# slurm/04_reconstruct_rcae.sh na hora de reconstruir) -- le de
# "$WORK_DIR/rcae_recon_<tag>" em vez do caminho fixo "$WORK_DIR/rcae_recon",
# e grava as metricas em arquivos/pastas com o mesmo sufixo (para nao
# misturar avaliacoes de checkpoints diferentes reconstruidos com tags
# diferentes no MESMO work_dir). Sem RECON_TAG, comportamento identico a
# antes -- so funciona se voce reconstruiu com o MESMO RECON_TAG antes.
#
# RECON_SUBJECTS="tag1,tag2" (variavel de ambiente, MESMA convencao do
# RECON_SUBJECTS em slurm/04_reconstruct_rcae.sh) restringe a etapa 6+7 a
# esses sujeitos -- use os MESMOS sujeitos que voce reconstruiu com
# RECON_SUBJECTS na etapa anterior, senao o script perde tempo calculando
# DTI/NODDI do baseline_sh/ground_truth em sujeitos que nem tem
# reconstrucao 'rcae' pra comparar (o resultado ainda sai certo sem isso,
# so mais lento). Ex.:
#   RECON_SUBJECTS="20170920171326_616_20170920171326_616" \
#     sbatch slurm/05_evaluate_and_downstream.sh <work_dir> 1000 10

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05_evaluate_and_downstream.sh <work_dir> [shell_b n_level]}"

EXPERIMENTS_TSV="configs/experiments.tsv"

SHARD_INDEX=0
SHARD_COUNT=1

if [[ -n "${2:-}" && -n "${3:-}" ]]; then
    SHELL_B="$2"
    N_LEVEL="$3"
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        # shell_b/n_level explicitos + dentro de um array => array e sharding
        # por sujeito para esse UNICO combo, nao "1 task = 1 combo".
        #
        # ATENCAO retry: SHARD_INDEX usa SEMPRE base 1 (SLURM_ARRAY_TASK_ID -
        # 1), NUNCA SLURM_ARRAY_TASK_MIN da submissao atual -- se voce
        # reenviar so as tasks que falharam com
        # "sbatch --array=28,36,42,48,62,81 ...", o Slurm preserva os
        # IDs originais (28, 36, ...) em SLURM_ARRAY_TASK_ID, entao
        # SHARD_INDEX=TASK_ID-1 continua batendo com o shard certo (27, 35,
        # ...) igual na submissao cheia. Usar SLURM_ARRAY_TASK_MIN/MAX (que
        # nessa resubmissao seriam 28/81, NAO 1/100) geraria
        # SHARD_INDEX/SHARD_COUNT errados e arquivos .shardXofY.csv com um Y
        # diferente do da primeira leva -- o merge nunca acharia esses
        # shards (foi exatamente esse bug que gerou "faltam os shards
        # [28, 36, ...]" mesmo depois de rodarem com sucesso).
        #
        # SHARD_COUNT precisa ser o TOTAL da submissao original (ex.: 100
        # se voce rodou "--array=1-100" da primeira vez) -- passe explicito
        # via env var em qualquer resubmissao parcial:
        #   SHARD_COUNT=100 sbatch --array=28,36,42,48,62,81 slurm/05_evaluate_and_downstream.sh ...
        # Sem isso, o fallback abaixo (MAX-MIN+1 desta submissao) SO esta
        # certo na primeira submissao completa (--array=1-N sem buracos).
        SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
        if [[ -n "${SHARD_COUNT_OVERRIDE:-}" ]]; then
            SHARD_COUNT="$SHARD_COUNT_OVERRIDE"
        elif [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${SLURM_ARRAY_TASK_MIN:-1}" == "1" ]]; then
            SHARD_COUNT="$SLURM_ARRAY_TASK_COUNT"
        else
            SHARD_COUNT=$((${SLURM_ARRAY_TASK_MAX:-1}))
        fi
        echo "[shard] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID MIN=${SLURM_ARRAY_TASK_MIN:-?} MAX=${SLURM_ARRAY_TASK_MAX:-?} -> SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT (confira: numa resubmissao parcial, SHARD_COUNT PRECISA bater com o total original -- use SHARD_COUNT_OVERRIDE=<N> se nao bater)"
    fi
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LINE=$(grep -v '^#' "$EXPERIMENTS_TSV" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    SHELL_B=$(echo "$LINE" | cut -f1)
    N_LEVEL=$(echo "$LINE" | cut -f2)
else
    echo "Erro: informe shell_b/n_level ou submeta com --array=1-N"
    exit 1
fi

if [[ "$SHARD_COUNT" -gt 1 ]]; then
    echo "Avaliando shell_b=$SHELL_B, n_level=$N_LEVEL -- shard $SHARD_INDEX/$SHARD_COUNT (RUN_NODDI=${RUN_NODDI:-0})"
else
    echo "Avaliando shell_b=$SHELL_B, n_level=$N_LEVEL (RUN_NODDI=${RUN_NODDI:-0})"
fi

source "./00_env_common.sh"

RCAE_DIR="$WORK_DIR/rcae_recon"
METRICS_SUFFIX=""
DOWNSTREAM_DIR="$WORK_DIR/downstream"
if [[ -n "${RECON_TAG:-}" ]]; then
    RCAE_DIR="$WORK_DIR/rcae_recon_${RECON_TAG}"
    METRICS_SUFFIX=".${RECON_TAG}"
    DOWNSTREAM_DIR="$WORK_DIR/downstream_${RECON_TAG}"
    echo "RECON_TAG=$RECON_TAG -- lendo de $RCAE_DIR, gravando metricas em $DOWNSTREAM_DIR"
fi

SUBJECTS_FLAG=()
if [[ -n "${RECON_SUBJECTS:-}" ]]; then
    SUBJECTS_FLAG=(--subjects "$RECON_SUBJECTS")
    echo "RECON_SUBJECTS=$RECON_SUBJECTS -- restringindo etapas 6+7 a esse(s) sujeito(s)"
fi

python scripts/06_evaluate_reconstruction.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --baseline-dir "$WORK_DIR/baseline_recon" \
    --rcae-dir "$RCAE_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
    --out-csv "$WORK_DIR/metrics/signal_metrics_shell${SHELL_B%.*}_n${N_LEVEL}${METRICS_SUFFIX}.csv" \
    "${SUBJECTS_FLAG[@]}"

NODDI_FLAG=""
if [[ "${RUN_NODDI:-0}" == "1" ]]; then
    NODDI_FLAG="--run-noddi"
fi

ROI_FLAG=()
if [[ -n "${ROI_TRACTS:-}" ]]; then
    ROI_FLAG=(--roi-tracts "$ROI_TRACTS")
    echo "ROI_TRACTS=$ROI_TRACTS -- metricas da etapa 7 tambem restritas a esses tratos"
fi

python scripts/07_downstream_dti_noddi.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --baseline-dir "$WORK_DIR/baseline_recon" \
    --rcae-dir "$RCAE_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" \
    --out-dir "$DOWNSTREAM_DIR" \
    $NODDI_FLAG "${ROI_FLAG[@]}" "${SUBJECTS_FLAG[@]}"

if [[ "${CLEANUP_AFTER:-0}" == "1" ]]; then
    if [[ -n "${RECON_TAG:-}" ]]; then
        echo "CLEANUP_AFTER=1 ignorado com RECON_TAG definido -- scripts/10_cleanup_reconstructions.py "
        echo "so conhece o caminho fixo (rcae_recon/baseline_recon), nao as pastas rotuladas por tag. "
        echo "Apague $RCAE_DIR manualmente quando nao precisar mais dela."
    elif [[ "$SHARD_COUNT" -gt 1 ]]; then
        echo "CLEANUP_AFTER=1 ignorado neste shard ($SHARD_INDEX/$SHARD_COUNT) -- rode a" \
             "limpeza manualmente DEPOIS de juntar todos os shards com merge_shard_csvs.py," \
             "senao outros shards ainda em andamento perdem os recon_target.nii.gz que precisam."
    else
        echo "CLEANUP_AFTER=1: apagando recon_target.nii.gz deste combo (metricas ja salvas)"
        python scripts/10_cleanup_reconstructions.py \
            --work-dir "$WORK_DIR" --shell-b "$SHELL_B" --n-level "$N_LEVEL"
    fi
fi