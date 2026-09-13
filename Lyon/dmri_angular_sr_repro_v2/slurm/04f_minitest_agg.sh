#!/bin/bash
#SBATCH --job-name=implicit_agg_minitest
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/implicit_agg_minitest.%A_%a.err
#SBATCH --output=logs/implicit_agg_minitest.%A_%a.out
#
# Mini-teste RAPIDO pra comparar --aggregation mean vs attention na linha
# `implicit` (scripts/04f_train_implicit.py, model/implicit_angular.py) --
# pedido explicito da usuaria em 2026-09-03: "dá pra fazer este mini treino
# pro implicit tb? testando mean e attention?". MESMO espirito do
# slurm/04i_minitest_io.sh (mini-manifesto + --freeze-subject-order + poucas
# epocas completas), adaptado ao formato de arquivo da linha `implicit`
# (<tag>_scheme.npz, NAO <tag>_rrin_triplets.npz -- ver
# scripts/99_make_mini_manifest.py --filter-scheme).
#
# LEMBRETE (ver addendum, nota metodologica sobre limites do mini-treino):
# isto e' SO' um teste RAPIDO de direcao/corretude -- serve pra comparar
# mean vs attention de forma RELATIVA em poucos minutos (qual converge
# melhor/mais rapido nas primeiras epocas, com codigo funcionando sem
# erro), NAO substitui a avaliacao real (attention pode precisar de mais
# dados/epocas pra mostrar vantagem -- a cabeca de atencao tem mais
# parametros que a media simples).
#
# O QUE ESTE SCRIPT FAZ (dedicado a este teste -- NAO mexe em nenhum
# wrapper slurm de PRODUCAO do `implicit`):
#   1. Constroi um manifesto MINI (scripts/99_make_mini_manifest.py, com
#      --filter-scheme pra so' pegar sujeitos que ja tem esquema pronto pro
#      (shell_b, n_level) pedido) em $WORK_DIR/manifest_mini_iotest.csv --
#      reconstruido do zero a cada chamada (idempotente, MESMO arquivo do
#      slurm/04i_minitest_io.sh -- pode sobrescrever entre um teste e outro,
#      sem problema, e' sempre reconstruido antes de treinar).
#   2. Roda scripts/04f_train_implicit.py sobre esse manifesto mini, com
#      --aggregation mean OU attention (AGGREGATION, default mean), por
#      poucas epocas completas (EPOCHS, default 6), escrevendo em
#      $WORK_DIR/implicit_checkpoints_iotest{_frozen}/ (pasta de DEBUG
#      separada -- nunca toca em checkpoint de producao). run_tag ja
#      diferencia mean/attention automaticamente (sufixo _attn, ver
#      scripts/04f_train_implicit.py), entao as duas rodadas no MESMO
#      OUT_DIR nunca colidem.
#
# COMO COMPARAR (rode as duas chamadas abaixo, mesmo work_dir/shell_b/
# n_level, e depois compare os val_loss por epoca nos dois logs):
#   AGGREGATION=mean      sbatch slurm/04f_minitest_agg.sh <work_dir> <shell_b> <n_level>
#   AGGREGATION=attention sbatch slurm/04f_minitest_agg.sh <work_dir> <shell_b> <n_level>
#
#   grep -E "^epoch" logs/implicit_agg_minitest.<JOBID_mean>.out
#   grep -E "^epoch" logs/implicit_agg_minitest.<JOBID_attention>.out
#
# SCHEME_DIR (default $WORK_DIR/subsampling) -- pasta com os
# <tag>_scheme.npz (etapa que gera isso: ver scripts/03*_build_scheme* ou
# equivalente na sua pipeline -- NAO e' a mesma coisa de --triplets-dir da
# linha RRIN/PairFlow).
#
# N_TRAIN/N_VAL (default 16/4) -- sujeitos no manifesto mini, por split.
# Recomendado manter N_TRAIN >= 2x NUM_WORKERS (default 8).
#
# EPOCHS (default 6) / NUM_WORKERS (default 8) / BATCH_SIZE (default 4,
# MESMO default de scripts/04f_train_implicit.py -- NAO 8 como na linha
# PairFlow) / LR (default 1e-3, MESMO default de 04f -- NAO 1e-4).
#
# FREEZE_SUBJECT_ORDER=1 (default 0) -- ativa --freeze-subject-order (novo
# nesta linha, ver addendum -- ainda nao medido aqui, so' no PairFlowStar).
#
# AGGREGATION=mean|attention (default mean) -- ver model/implicit_angular.py
# e addendum secao 29.
#
# MAX_CACHED_SUBJECTS (default 6) -- MESMO esquema validado no PairFlowStar
# (ver slurm/04i_minitest_io.sh e addendum 33.8): utils/dataset.py's
# DWIPatchDataset._load_subject usa o MESMO cache LRU por-worker (OrderedDict),
# dominado pelo volume DWI 4D completo por sujeito ("dwi": data.astype
# (np.float32)) -- arquitetura identica a RRINTripletDataset. Se
# N_TRAIN/NUM_WORKERS (sujeitos por worker) exceder MAX_CACHED_SUBJECTS, uma
# fracao dos sujeitos e' despejada do cache antes da proxima epoca mesmo com
# FREEZE_SUBJECT_ORDER=1, forcando releitura de disco e derrubando
# patches/s (visto no PairFlowStar: N_TRAIN=64/NUM_WORKERS=8 exigiu
# MAX_CACHED_SUBJECTS>=8 pra nao degradar -- regra pratica: MAX_CACHED_SUBJECTS
# >= ceil(N_TRAIN/NUM_WORKERS), com folga se RAM permitir). RAM total pro
# cache (por causa de NUM_WORKERS processos separados, cada um com seu
# proprio cache): aprox. NUM_WORKERS * MAX_CACHED_SUBJECTS * bytes_por_sujeito,
# onde bytes_por_sujeito ~= nx*ny*nz*n_volumes_total*4. Pra medir o uso real
# (nao confundir com page cache do SO): seff -M gpu <jobid> ou
# sacct -M gpu -j <jobid> --format=JobID,MaxRSS,ReqMem,Elapsed -- se MaxRSS
# bater exatamente no ReqMem, e' inconclusivo (pode ser so' page cache
# reclamavel); reode com --mem bem maior (ex: 64G) pra ver onde o uso
# realmente estabiliza.
#
# VAL_NUM_WORKERS (default 0) / VAL_MAX_CACHED_SUBJECTS (default 1) -- MESMO
# esquema/CUIDADO documentado em slurm/04i_minitest_io.sh (secao 33.11/33.12
# do addendum): ate' 2026-09-03 vinham HARDCODED (0/1), entao a validacao
# rodava 100% sincrona (sem overlap I/O/compute) INDEPENDENTE do
# MAX_CACHED_SUBJECTS do treino. scripts/04f_train_implicit.py's val_loader
# usa `shuffle=False` SEM SubjectGroupedSampler (esse sampler so' e'
# aplicado ao train_loader) -- ou seja, com VAL_NUM_WORKERS>1 NAO ha
# garantia de particionamento de sujeitos por worker como no treino
# (releitura redundante entre workers e' possivel). Mesmo assim, sair de
# VAL_NUM_WORKERS=0 pra >0 deve ajudar so' pelo overlap I/O/compute --
# vale testar empiricamente. Ponto de partida sugerido: VAL_NUM_WORKERS=2
# com VAL_MAX_CACHED_SUBJECTS>=N_VAL.
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04f_minitest_agg.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 04f_minitest_agg.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 04f_minitest_agg.sh <work_dir> <shell_b> <n_level>}"

N_TRAIN="${N_TRAIN:-16}"
N_VAL="${N_VAL:-4}"
EPOCHS="${EPOCHS:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
AGGREGATION="${AGGREGATION:-mean}"
MAX_CACHED_SUBJECTS="${MAX_CACHED_SUBJECTS:-6}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
VAL_MAX_CACHED_SUBJECTS="${VAL_MAX_CACHED_SUBJECTS:-1}"

echo "Mini-teste (implicit, aggregation=$AGGREGATION) para shell_b=$SHELL_B, n_level=$N_LEVEL"
echo "N_TRAIN=$N_TRAIN N_VAL=$N_VAL EPOCHS=$EPOCHS NUM_WORKERS=$NUM_WORKERS BATCH_SIZE=$BATCH_SIZE LR=$LR MAX_CACHED_SUBJECTS=$MAX_CACHED_SUBJECTS"
echo "VAL_NUM_WORKERS=$VAL_NUM_WORKERS VAL_MAX_CACHED_SUBJECTS=$VAL_MAX_CACHED_SUBJECTS"
if [[ "$VAL_NUM_WORKERS" == "0" ]]; then
    echo "VAL_NUM_WORKERS=0 -- validacao 100% sincrona (sem overlap I/O/compute); suba pra >0 pra testar ganho"
fi
if (( MAX_CACHED_SUBJECTS * NUM_WORKERS < N_TRAIN )); then
    echo "AVISO: MAX_CACHED_SUBJECTS ($MAX_CACHED_SUBJECTS) * NUM_WORKERS ($NUM_WORKERS) < N_TRAIN ($N_TRAIN) -- alguns sujeitos serao despejados do cache a cada epoca, degradando patches/s mesmo com FREEZE_SUBJECT_ORDER=1. Considere aumentar MAX_CACHED_SUBJECTS."
fi
source "./00_env_common.sh"

SCHEME_DIR="${SCHEME_DIR:-$WORK_DIR/subsampling}"
if [[ "$SCHEME_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "SCHEME_DIR=$SCHEME_DIR -- lendo esquema de pasta SEPARADA da producao (subsampling/)"
fi

MINI_MANIFEST="$WORK_DIR/manifest_mini_iotest.csv"
echo "[1/2] construindo manifesto mini em $MINI_MANIFEST ..."
python scripts/99_make_mini_manifest.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --out "$MINI_MANIFEST" \
    --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test 0 \
    --filter-scheme --scheme-dir "$SCHEME_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL"

OUT_DIR="$WORK_DIR/implicit_checkpoints_iotest"
FREEZE_FLAG=()
if [[ "${FREEZE_SUBJECT_ORDER:-0}" == "1" ]]; then
    FREEZE_FLAG=(--freeze-subject-order)
    OUT_DIR="${OUT_DIR}_frozen"
    echo "FREEZE_SUBJECT_ORDER=1 -- ativando --freeze-subject-order (checkpoints de debug em $OUT_DIR)"
else
    echo "FREEZE_SUBJECT_ORDER nao ativado -- rodada BASELINE (ordem reembaralhada a cada epoca, checkpoints de debug em $OUT_DIR)"
fi

# INIT_OUTPUT_BIAS_FROM_DATA=1 / WARMUP_STEPS=<N> -- ADITIVOS (default 0/
# desligado), trazidos de scripts/04f_train_implicit.py em 2026-09-03
# (pedido explicito: "o implicit começa com val loss bem alta... se
# pudesse melhorar alguma coisa"). Ver docstring do script pra detalhes --
# resumo: o primeiro inicializa o bias da ultima camada do decoder com a
# media do sinal-alvo (evita a rede gastar epocas so' aprendendo o nivel
# constante); o segundo aquece a LR linearmente nos primeiros N passos de
# otimizador antes do ReduceLROnPlateau assumir.
INIT_BIAS_FLAG=()
if [[ "${INIT_OUTPUT_BIAS_FROM_DATA:-0}" == "1" ]]; then
    INIT_BIAS_FLAG=(--init-output-bias-from-data)
    echo "INIT_OUTPUT_BIAS_FROM_DATA=1 -- inicializando bias da camada de saida com a media do alvo"
fi
WARMUP_STEPS="${WARMUP_STEPS:-0}"
if [[ "$WARMUP_STEPS" != "0" ]]; then
    echo "WARMUP_STEPS=$WARMUP_STEPS -- aquecendo LR linearmente nos primeiros $WARMUP_STEPS passos"
fi

echo "[2/2] rodando $EPOCHS epoca(s) completas no manifesto mini (aggregation=$AGGREGATION) ..."
python scripts/04f_train_implicit.py \
    --manifest "$MINI_MANIFEST" \
    --scheme-dir "$SCHEME_DIR" \
    --out-dir "$OUT_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --aggregation "$AGGREGATION" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --patch-size 10 \
    --lr "$LR" --num-workers "$NUM_WORKERS" --max-cached-subjects "$MAX_CACHED_SUBJECTS" --patience 999 \
    --val-num-workers "$VAL_NUM_WORKERS" --val-max-cached-subjects "$VAL_MAX_CACHED_SUBJECTS" \
    --warmup-steps "$WARMUP_STEPS" \
    --no-resume \
    "${FREEZE_FLAG[@]}" "${INIT_BIAS_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

echo "[ok] mini-teste concluido -- confira as linhas 'epoch NNN | train ... | val ...' acima (ou no .out deste job) para comparar mean vs attention."