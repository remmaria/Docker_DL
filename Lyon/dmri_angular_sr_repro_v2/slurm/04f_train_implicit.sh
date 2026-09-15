#!/bin/bash
#SBATCH --job-name=implicit_angular
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train_implicit.%A_%a.err
#SBATCH --output=logs/train_implicit.%A_%a.out
#
# Treino do modelo de representacao angular IMPLICITA (NeRF/LIIF-style,
# etapa 4f, ver model/implicit_angular.py e scripts/04f_train_implicit.py --
# addendum secao 20.11) para um (shell_b, n_level) especifico.
#
# DIFERENCA IMPORTANTE em relacao a TODOS os wrappers 04b/04c/04d/04e
# (RRIN/AMT/HFD/estrela): este le DIRETAMENTE de <tag>_scheme.npz (saida da
# etapa 2, scripts/02_subsample_directions.py) -- NAO precisa que
# scripts/02b_build_rrin_triplets.py tenha rodado, e por isso NAO tem
# TRIPLETS_DIR/ENSEMBLE_M aqui (nao ha pareamento nenhum nesta linha). Mesmo
# esquema de scheme-dir que scripts/04_train_rcae.py ja usa (ver
# slurm/03_train_rcae.sh).
#
# Uso:
#   sbatch --array=1-N slurm/04f_train_implicit.sh <work_dir>
#   sbatch slurm/04f_train_implicit.sh <work_dir> <shell_b> <n_level>
#
# RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1 -- mesmo mecanismo de resume
# automatico dos demais treinos (ver scripts/04f_train_implicit.py).
#
# LR=<valor> (default 1e-3).
#
# L_MAX=<inteiro> -- ordem par maxima da base SH usada para codificar
# direcao (ver model/implicit_angular.py:sh_positional_encoding). Default
# vazio = automatico (max_order_for_n_directions(n_level), mesma convencao
# do baseline_sh). BLOQUEANTE para resume (muda o shape dos pesos) -- ver
# checagem em scripts/04f_train_implicit.py.
#
# BASE_CH=<inteiro> (default 16) -- largura dos blocos conv. NORM_TYPE=batch
# (default instance) -- mesma semantica/mesmo custo (exige treino do zero)
# de NORM_TYPE em slurm/04b_train_rrin.sh/04e_train_rrin_star.sh.
#
# AGGREGATION=attention (default mean, ADITIVO -- ver addendum 2026-09-03
# secao 29 e model/implicit_angular.py:AttentionAggregator3D) -- troca a
# media simples entre as n_level direcoes de entrada por uma media
# ponderada aprendida (pesos por voxel), motivada pelo diagnostico de
# scripts/14_diagnose_implicit_pooling.py (secao 28.1: colapso de
# agregacao alto, sensibilidade leave-one-out ainda moderada num
# checkpoint precoce). BLOQUEANTE para resume entre valores diferentes
# (muda o shape dos pesos), mas ganha seu proprio run_tag automaticamente
# (sufixo "_attn"), entao nunca colide com checkpoints antigos de
# AGGREGATION=mean no mesmo <work_dir>/implicit_checkpoints -- rodar com
# AGGREGATION=attention comeca um treino NOVO do zero, em paralelo ao
# treino "mean" existente, sem afetar o checkpoint antigo.
#
# CROSS_DIRECTION_ATTENTION=1 (ADITIVO, default desligado, addendum
# 2026-09-13, item 2 -- ver model/implicit_angular.py:CrossDirectionAttention3D)
# -- insere um bloco de self-attention ENTRE as n_level direcoes de entrada
# logo apos PerDirectionEncoder3D e ANTES de qualquer agregacao (compoe com
# AGGREGATION=mean OU attention, sao etapas independentes). Motivado por
# producao real saturando em val_loss~0,041-0,042 ja na epoca 3 mesmo com
# AGGREGATION=attention. CROSS_ATTN_HEADS=<int> (default 4, so tem efeito
# com CROSS_DIRECTION_ATTENTION=1) -- precisa dividir BASE_CH sem resto.
# Adiciona parametros novos -- treino NOVO do zero (run_tag ganha sufixo
# _xattn, nunca colide com checkpoints sem esta flag).
#   CROSS_DIRECTION_ATTENTION=1 AGGREGATION=attention sbatch slurm/04f_train_implicit.sh <work_dir> <shell_b> <n_level>
#
# SCHEME_DIR=<caminho> -- por padrao usa <work_dir>/subsampling (saida da
# etapa 2), mesma convencao de TRIPLETS_DIR nos outros wrappers.
#
# INIT_OUTPUT_BIAS_FROM_DATA=1 (ADITIVO, default desligado, trazido de
# slurm/04f_minitest_agg.sh em 2026-09-04, ver addendum secao 33.18) --
# inicializa o bias da ultima camada do decoder com a media do alvo,
# amostrada de ate' 4 batches de treino, em vez do bias default do PyTorch.
#
# WARMUP_STEPS=<inteiro> (ADITIVO, default 0 = desligado, mesma origem) --
# aquece a LR linearmente de 0.1*LR ate' LR ao longo dos N primeiros passos
# de otimizador. Ignorado (com aviso no log) se estiver retomando de um
# checkpoint existente (RESUME_CHECKPOINT ou last.pt detectado).
#
# ANGULAR_LOSS_WEIGHT=<valor> (ADITIVO, default 0.0 = desligado, porte pra
# esta linha pedido pela usuaria em 2026-09-09, ver addendum secao 33.32) --
# lambda do termo de loss opcional no dominio angular/SH (mesmo mecanismo
# ja usado em scripts/04_train_rcae.py/04b_train_rrin.py, ver protocolo
# secao 9). Grava em run_tag com sufixo _sh (nao colide com o checkpoint
# sem loss angular). SH_LOSS_HIGH_ORDER_MIN=<int> (default 4) e
# SH_LOSS_LMAX_CAP=<int> (default 8) sao overrides opcionais, so tem
# efeito com ANGULAR_LOSS_WEIGHT>0.
#
# ATENCAO (bug de doc corrigido em 2026-09-09, ver addendum secao 33.33):
# --q-out (default 10, QOUT abaixo) e' o teto de N_out TANTO em treino
# quanto em validacao (utils/dataset.py:_dynamic_split -- a re-amostragem
# em treino so' muda QUAIS direcoes formam o alvo, nao a CONTAGEM maxima),
# entao com o default QOUT=10 este termo fica ZERADO em TODO batch se
# SH_LOSS_HIGH_ORDER_MIN=4 (default, precisa de 15 direcoes-alvo). Pra ter
# o termo com efeito real: baixe SH_LOSS_HIGH_ORDER_MIN=2 (maximo
# sustentavel com QOUT=10), OU suba QOUT (>=15 pra l=4).
#   ANGULAR_LOSS_WEIGHT=0.5 SH_LOSS_HIGH_ORDER_MIN=2 sbatch slurm/04f_train_implicit.sh <work_dir> <shell_b> <n_level>
#   ANGULAR_LOSS_WEIGHT=0.5 QOUT=16 sbatch slurm/04f_train_implicit.sh <work_dir> <shell_b> <n_level>
#
# BATCH_SIZE=<N> (default 8) / MAX_CACHED_SUBJECTS=<N> (default 6) /
# FREEZE_SUBJECT_ORDER=1 (default 0) -- trazidos do mesmo pacote de
# achados de I/O ja aplicado a slurm/04i_train_pairflow_star.sh (ver
# addendum 2026-09-03, secoes 33.1-33.17/27/30). ANTES desta mudanca
# (2026-09-13) este wrapper de PRODUCAO tinha os tres hardcoded
# (--batch-size 8/--num-workers 8/--max-cached-subjects 6 fixos no
# python abaixo) -- ou seja, a sintese da secao 33.17
# (BATCH_SIZE=32/LR=4e-3 recomendado, mini-teste) nunca pode ser rodada
# de verdade em producao ate' agora, so' no wrapper de mini-teste
# (slurm/04f_minitest_agg.sh). MESMA ressalva de 04i: o mini-teste foi
# validado com N_TRAIN pequeno (poucos sujeitos/worker) -- a producao
# real (~600-700 sujeitos de treino sobre os mesmos 8 workers, nao
# configuravel aqui) nao tem a mesma cobertura de cache, entao nao
# espere o mesmo ganho de patches/s visto no mini-teste, so BATCH_SIZE/
# LR escalada e FREEZE_SUBJECT_ORDER=1 devem se comportar de forma
# parecida (nao dependem de cobrir todos os sujeitos do worker).
#   BATCH_SIZE=32 LR=4e-3 FREEZE_SUBJECT_ORDER=1 AGGREGATION=attention \
#     sbatch slurm/04f_train_implicit.sh <work_dir> <shell_b> <n_level>
#
# CKPT_TAG=<string> (ADITIVO, default vazio, ver addendum secao 33.24) --
# ATENCAO: o run_tag calculado por scripts/04f_train_implicit.py so' leva
# em conta shell_b/n_level/l_max/base_ch/norm_type/aggregation -- NAO
# INIT_OUTPUT_BIAS_FROM_DATA/WARMUP_STEPS (nem BATCH_SIZE/
# FREEZE_SUBJECT_ORDER, que agora sao refletidos so' no OUT_DIR deste
# wrapper, nao no run_tag do python). Duas rodadas com a MESMA config de
# aggregation/etc mas INIT_OUTPUT_BIAS_FROM_DATA/WARMUP_STEPS DIFERENTES
# apontam pro MESMO out_dir por padrao -- com NO_RESUME=1 a rodada mais
# recente SOBRESCREVE o checkpoint da anterior assim que salvar, sem
# aviso. Passe CKPT_TAG=<algo curto, ex.: init> pra manter os runs
# separados.
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04f_train_implicit.sh <work_dir> [shell_b n_level]}"
EXPERIMENTS_TSV="configs/experiments.tsv"
if [[ -n "${2:-}" && -n "${3:-}" ]]; then
    SHELL_B="$2"
    N_LEVEL="$3"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LINE=$(grep -v '^#' "$EXPERIMENTS_TSV" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    if [[ -z "$LINE" ]]; then
        echo "Erro: nao ha linha $SLURM_ARRAY_TASK_ID em $EXPERIMENTS_TSV (confira --array=1-N)"
        exit 1
    fi
    SHELL_B=$(echo "$LINE" | cut -f1)
    N_LEVEL=$(echo "$LINE" | cut -f2)
else
    echo "Erro: informe shell_b/n_level como argumentos OU submeta com --array=1-N"
    exit 1
fi
echo "Treinando modelo implicito (angular, estilo NeRF/LIIF) para shell_b=$SHELL_B, n_level=$N_LEVEL"
source "./00_env_common.sh"

RESUME_FLAG=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_FLAG=(--resume-checkpoint "$RESUME_CHECKPOINT")
    echo "RESUME_CHECKPOINT=$RESUME_CHECKPOINT -- retomando explicitamente deste checkpoint"
elif [[ "${NO_RESUME:-0}" == "1" ]]; then
    RESUME_FLAG=(--no-resume)
    echo "NO_RESUME=1 -- ignorando qualquer last.pt existente, comecando do zero"
fi

LR="${LR:-1e-3}"
echo "LR=$LR (default 1e-3)"

LMAX_FLAG=()
if [[ -n "${L_MAX:-}" ]]; then
    LMAX_FLAG=(--l-max "$L_MAX")
    echo "L_MAX=$L_MAX -- fixando ordem SH explicita (default seria automatico por n_level)"
fi

BASE_CH="${BASE_CH:-16}"
BASE_CH_FLAG=()
if [[ "$BASE_CH" != "16" ]]; then
    BASE_CH_FLAG=(--base-ch "$BASE_CH")
    echo "BASE_CH=$BASE_CH (default seria 16)"
fi

NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (exige treino do zero)"
fi

SCHEME_DIR="${SCHEME_DIR:-$WORK_DIR/subsampling}"
if [[ "$SCHEME_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "SCHEME_DIR=$SCHEME_DIR -- lendo esquema de pasta SEPARADA da producao (subsampling/)"
fi

AGGREGATION="${AGGREGATION:-mean}"
AGGREGATION_FLAG=()
if [[ "$AGGREGATION" != "mean" ]]; then
    AGGREGATION_FLAG=(--aggregation "$AGGREGATION")
    echo "AGGREGATION=$AGGREGATION -- treino NOVO com agregacao aprendida (run_tag ganha sufixo _attn, nao colide com o checkpoint 'mean' existente)"
fi

CROSS_ATTN_FLAG=()
if [[ "${CROSS_DIRECTION_ATTENTION:-0}" == "1" ]]; then
    CROSS_ATTN_FLAG=(--cross-direction-attention)
    echo "CROSS_DIRECTION_ATTENTION=1 -- treino NOVO com self-attention entre as n_level direcoes de entrada (run_tag ganha sufixo _xattn)"
fi
CROSS_ATTN_HEADS="${CROSS_ATTN_HEADS:-4}"
CROSS_ATTN_HEADS_FLAG=()
if [[ "$CROSS_ATTN_HEADS" != "4" ]]; then
    CROSS_ATTN_HEADS_FLAG=(--cross-attn-heads "$CROSS_ATTN_HEADS")
    echo "CROSS_ATTN_HEADS=$CROSS_ATTN_HEADS (default 4, so tem efeito com CROSS_DIRECTION_ATTENTION=1) -- precisa dividir BASE_CH sem resto"
fi

INIT_BIAS_FLAG=()
if [[ "${INIT_OUTPUT_BIAS_FROM_DATA:-0}" == "1" ]]; then
    INIT_BIAS_FLAG=(--init-output-bias-from-data)
    echo "INIT_OUTPUT_BIAS_FROM_DATA=1 -- inicializando bias de saida com a media do alvo (amostrado do treino)"
fi

WARMUP_STEPS="${WARMUP_STEPS:-0}"
if [[ "$WARMUP_STEPS" != "0" ]]; then
    echo "WARMUP_STEPS=$WARMUP_STEPS -- LR sobe linearmente de 0.1*LR ate LR ao longo dos primeiros $WARMUP_STEPS passos de otimizador"
fi

ANGULAR_LOSS_WEIGHT="${ANGULAR_LOSS_WEIGHT:-0.0}"
SH_LOSS_HIGH_ORDER_MIN="${SH_LOSS_HIGH_ORDER_MIN:-4}"
SH_LOSS_LMAX_CAP="${SH_LOSS_LMAX_CAP:-8}"
ANGULAR_LOSS_FLAG=()
if [[ "$ANGULAR_LOSS_WEIGHT" != "0.0" && "$ANGULAR_LOSS_WEIGHT" != "0" ]]; then
    ANGULAR_LOSS_FLAG=(--angular-loss-weight "$ANGULAR_LOSS_WEIGHT" \
        --sh-loss-high-order-min "$SH_LOSS_HIGH_ORDER_MIN" --sh-loss-lmax-cap "$SH_LOSS_LMAX_CAP")
    echo "ANGULAR_LOSS_WEIGHT=$ANGULAR_LOSS_WEIGHT (high_order_min=$SH_LOSS_HIGH_ORDER_MIN, lmax_cap=$SH_LOSS_LMAX_CAP) -- treino NOVO com loss angular (run_tag ganha sufixo _sh)"
fi

# QOUT (default 10, igual ao default de --q-out do script python): teto de
# N_out TANTO em treino quanto em validacao (ver utils/dataset.py:_dynamic_split).
# Se ANGULAR_LOSS_WEIGHT>0 e SH_LOSS_HIGH_ORDER_MIN exigir mais direcoes-alvo
# validas do que QOUT sustenta, train_loss_angular/val_loss_angular ficam
# ZERADOS em todo batch (ver cabecalho + addendum 33.33). Default 10 so
# sustenta ate l_max=2; suba pra 15+ (l=4) ou 28+ (l=6) se for usar
# SH_LOSS_HIGH_ORDER_MIN>2.
QOUT="${QOUT:-10}"
QOUT_FLAG=(--q-out "$QOUT")
echo "QOUT=$QOUT -- teto de N_out em treino E validacao"

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_CACHED_SUBJECTS="${MAX_CACHED_SUBJECTS:-6}"
OUT_DIR_BASE="$WORK_DIR/implicit_checkpoints"
if [[ "$BATCH_SIZE" != "8" ]]; then
    OUT_DIR_BASE="${OUT_DIR_BASE}_bs${BATCH_SIZE}"
    echo "BATCH_SIZE=$BATCH_SIZE (default 8) -- checkpoints em $OUT_DIR_BASE"
fi
if [[ "$MAX_CACHED_SUBJECTS" != "6" ]]; then
    echo "MAX_CACHED_SUBJECTS=$MAX_CACHED_SUBJECTS (default 6) -- lembrar que a escala de producao (~75-90 sujeitos/worker) nao e' totalmente coberta por esse cache (ver nota de cabecalho)"
fi

FREEZE_ORDER_FLAG=()
if [[ "${FREEZE_SUBJECT_ORDER:-0}" == "1" ]]; then
    FREEZE_ORDER_FLAG=(--freeze-subject-order)
    OUT_DIR_BASE="${OUT_DIR_BASE}_frozen_order"
    echo "FREEZE_SUBJECT_ORDER=1 -- ativando --freeze-subject-order (checkpoints em $OUT_DIR_BASE)"
fi

if [[ -n "${CKPT_TAG:-}" ]]; then
    OUT_DIR_BASE="${OUT_DIR_BASE}_${CKPT_TAG}"
    echo "CKPT_TAG=$CKPT_TAG -- checkpoints em $OUT_DIR_BASE/<run_tag> (ver nota de cabecalho: o run_tag calculado por scripts/04f_train_implicit.py NAO leva em conta INIT_OUTPUT_BIAS_FROM_DATA/WARMUP_STEPS -- use CKPT_TAG pra evitar colidir/sobrescrever um run anterior com essas flags diferentes na MESMA config de shell_b/n_level/l_max/base_ch/norm_type/aggregation)"
fi

# --- capacidade do decoder (addendum 2026-09-14) ---------------------------
# O ImplicitDecoderHead3D e' a rede que representa a funcao continua sobre a
# esfera -- o proposito do modelo implicito -- e com BASE_CH=16 ela tem
# 13.873 parametros e 2 convolucoes, contra 3.723.497 e 10 convolucoes do
# decoder do RCAE (268x). Alem disso o codigo SH da direcao-alvo (a
# "consulta") entrava so' na primeira camada.
#   DECODER_BASE_CH=<N>    largura so' do decoder      -> sufixo _dbc<N>
#   DECODER_DEPTH=<N>      camadas ocultas (default 1) -> sufixo _dd<N>
#   DECODER_REINJECT_CODE=1  reinjeta o codigo SH em cada camada oculta
#                            (como o RCAE faz com o bvec) -> sufixo _dreinj
DECODER_BASE_CH="${DECODER_BASE_CH:-}"
DECODER_BASE_CH_FLAG=()
if [[ -n "$DECODER_BASE_CH" ]]; then
    DECODER_BASE_CH_FLAG=(--decoder-base-ch "$DECODER_BASE_CH")
    echo "DECODER_BASE_CH=$DECODER_BASE_CH -- largura do decoder desacoplada do resto (run_tag ganha _dbc$DECODER_BASE_CH)"
fi
DECODER_DEPTH="${DECODER_DEPTH:-1}"
DECODER_DEPTH_FLAG=()
if [[ "$DECODER_DEPTH" != "1" ]]; then
    DECODER_DEPTH_FLAG=(--decoder-depth "$DECODER_DEPTH")
    echo "DECODER_DEPTH=$DECODER_DEPTH (default 1) -- camadas ocultas do decoder (run_tag ganha _dd$DECODER_DEPTH)"
fi
DECODER_REINJECT_FLAG=()
if [[ "${DECODER_REINJECT_CODE:-0}" == "1" ]]; then
    DECODER_REINJECT_FLAG=(--decoder-reinject-code)
    echo "DECODER_REINJECT_CODE=1 -- codigo SH da direcao-alvo reinjetado em cada camada oculta do decoder (so tem efeito com DECODER_DEPTH>1; run_tag ganha _dreinj)"
fi

python scripts/04f_train_implicit.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$SCHEME_DIR" \
    --out-dir "$OUT_DIR_BASE" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size "$BATCH_SIZE" --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects "$MAX_CACHED_SUBJECTS" --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    --warmup-steps "$WARMUP_STEPS" \
    "${RESUME_FLAG[@]}" "${LMAX_FLAG[@]}" "${BASE_CH_FLAG[@]}" "${NORM_TYPE_FLAG[@]}" \
    "${AGGREGATION_FLAG[@]}" "${INIT_BIAS_FLAG[@]}" "${ANGULAR_LOSS_FLAG[@]}" "${QOUT_FLAG[@]}" \
    "${FREEZE_ORDER_FLAG[@]}" "${CROSS_ATTN_FLAG[@]}" "${CROSS_ATTN_HEADS_FLAG[@]}" \
    "${DECODER_BASE_CH_FLAG[@]}" "${DECODER_DEPTH_FLAG[@]}" "${DECODER_REINJECT_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"