#!/bin/bash
#SBATCH --job-name=pairflow_star
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train_pairflow_star.%A_%a.err
#SBATCH --output=logs/train_pairflow_star.%A_%a.out
#
# Etapa 4i ("ensemble em estrela" da linha PairFlow, ver
# scripts/04i_train_pairflow_star.py e model/pairflow_star.py -- pedido
# explicito da usuaria em 2026-09-03, "quero o pairflow ensemble") para um
# (shell_b, n_level) especifico -- requer que scripts/02b_build_rrin_
# triplets.py ja tenha rodado COM --ensemble-m>=M pra esse work_dir (mesmo
# requisito de slurm/04e_train_rrin_star.sh, MESMO esquema de feixe).
#
# Uso (com pre-treino da etapa 4g -- ver slurm/04h_train_pairflow_finetune.sh):
#   INIT_CHECKPOINT=$WORK_DIR/pairflow_ssl_checkpoints/shell1000/best.pt \
#     sbatch slurm/04i_train_pairflow_star.sh <work_dir> <shell_b> <n_level>
#
# Uso (controle, sem pre-treino -- treina do zero nas trincas curadas):
#   sbatch slurm/04i_train_pairflow_star.sh <work_dir> <shell_b> <n_level>
#
# ENSEMBLE_M=<M> (default 3) -- mesma semantica de ENSEMBLE_M em
# slurm/04e_train_rrin_star.sh, PRECISA bater (ou ser <=) o M usado ao rodar
# scripts/02b_build_rrin_triplets.py --ensemble-m nesse work_dir.
#
# FREEZE_FLOW=1 (default 0, requer INIT_CHECKPOINT) -- congela o fluxo
# pre-treinado, so' treina refine_net/weight_head (ver --freeze-flow e
# docstring de model.pairflow_star.PairFlowStar).
#
# WEIGHT_QUALITY_COND=1 -- alimenta residual_deg/gap_deg direto na
# PairFlowWeightHead3D (condiciona a fusao entre os M pares). NAO ha
# USE_QUALITY_COND aqui (ao contrario de slurm/04e_train_rrin_star.sh) --
# PairFlowNet3D nao condiciona o fluxo por nenhum sinal de qualidade (ver
# model/pairflow_ssl.py).
#
# ONLY_VALID=0 (default 1) / NORM_TYPE=batch (default instance) -- mesma
# semantica/sufixos automaticos de slurm/04e_train_rrin_star.sh.
#
# LR=<valor> (default 1e-4, MESMO default de slurm/04h_train_pairflow_
# finetune.sh -- nao 1e-3 como o RRIN-star, pra manter consistencia com o
# resto da linha PairFlow). RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1 --
# mesmo mecanismo de resume automatico dos demais treinos.
#
# BATCH_SIZE=<N> (default 8) / FREEZE_SUBJECT_ORDER=1 (default 0) /
# MAX_CACHED_SUBJECTS=<N> (default 6) -- trazidos do mini-teste de I/O
# (slurm/04i_minitest_io.sh, ver addendum 2026-09-03, secoes 33.1-33.17)
# pra quem quiser aplicar aqui os achados validados em escala pequena.
#
# ATENCAO -- ESSES ACHADOS NAO TRANSFEREM DIRETO PRA ESCALA DE PRODUCAO:
# o mini-teste foi validado com N_TRAIN=64-128/NUM_WORKERS=8 (8-16
# sujeitos/worker); a producao real tem ~600-700 sujeitos de treino (ver
# "manifesto de origem" no log) sobre os MESMOS 8 workers (nao
# configuravel aqui) -- ou seja, ~75-90 sujeitos/worker. `MAX_CACHED_
# SUBJECTS=16` (bom o suficiente pra 8-16 sujeitos/worker) NAO cobre
# essa escala -- a maioria dos sujeitos de cada worker ainda vai ser
# despejada do cache a cada epoca, mesmo com `--freeze-subject-order`
# (ver formula de RAM na secao 33.9: NUM_WORKERS x MAX_CACHED_SUBJECTS x
# bytes_por_sujeito). Este wrapper ja pede `--mem=64G` (2x o do mini-
# teste) -- da' pra subir `MAX_CACHED_SUBJECTS` um pouco (ex.: 12-16) em
# relacao ao default de producao (6) sem pedir mais RAM, mas nao espere
# o MESMO salto de patches/s visto no mini-teste (que cobria o worker
# INTEIRO) -- aqui e' so' uma cobertura parcial. `BATCH_SIZE`/`LR`
# escalada (ex.: 32/2e-4) e `FREEZE_SUBJECT_ORDER=1`, por outro lado,
# devem se comportar de forma mais parecida com o mini-teste (nao
# dependem de cobrir todos os sujeitos do worker).
#
# WARMUP_STEPS=<inteiro> (ADITIVO, default 0 = desligado, ver addendum
# secao 33.19) -- aquece a LR linearmente de 0.1*LR ate' LR ao longo dos N
# primeiros passos de otimizador. Ignorado (com aviso no log) se estiver
# retomando de um checkpoint existente (RESUME_CHECKPOINT ou last.pt
# detectado).
#
# ZERO_INIT_REFINE_OUTPUT=1 (ADITIVO, default 0, mesma secao) -- zera a
# ultima camada de RefineNet3D na inicializacao, forcando a predicao
# inicial a coincidir com o blend (interpolacao por fluxo optico) em vez
# de somar um residual aleatorio por cima -- analogo, pra esta arquitetura,
# do --init-output-bias-from-data do implicit.
#
# ANGULAR_LOSS_WEIGHT=<valor> (ADITIVO, default 0.0 = desligado, porte pra
# esta linha pedido pela usuaria em 2026-09-09, ver addendum secao 33.32) --
# lambda do termo de loss opcional no dominio angular/SH (mesmo mecanismo
# ja usado em scripts/04_train_rcae.py/04b_train_rrin.py). Cada direcao do
# feixe SH extra roda pelo modelo como um ensemble DEGENERADO de M=1 par
# real (ver scripts/04i_train_pairflow_star.py:_sh_bundle_forward_star --
# o feixe sh_q_out do dataset ainda nao suporta um M completo por
# direcao). Grava em run_tag com sufixo _sh. SH_LOSS_HIGH_ORDER_MIN=<int>
# (default 4), SH_LOSS_LMAX_CAP=<int> (default 8) e SH_LOSS_Q_OUT=<int>
# (default 16, tamanho do feixe extra amostrado por item) sao overrides
# opcionais, so tem efeito com ANGULAR_LOSS_WEIGHT>0.
#   ANGULAR_LOSS_WEIGHT=0.5 sbatch slurm/04i_train_pairflow_star.sh <work_dir> <shell_b> <n_level>
#
# CKPT_TAG=<string> (ADITIVO, default vazio, ver addendum secao 33.23) --
# ATENCAO: OUT_DIR e' montado so' a partir de BATCH_SIZE/FREEZE_SUBJECT_
# ORDER, NAO leva em conta qual INIT_CHECKPOINT foi passado -- duas
# rodadas com a MESMA config de BATCH_SIZE/FREEZE_SUBJECT_ORDER mas
# INIT_CHECKPOINT DIFERENTE (ex.: retreinar do zero com um checkpoint SSL
# mais maduro) apontam pro MESMO out_dir por padrao, entao com
# NO_RESUME=1 a segunda rodada SOBRESCREVE best.pt/last.pt da primeira
# assim que salvar. Passe CKPT_TAG=<algo curto, ex.: sslep131> pra
# adicionar um sufixo distinto ao OUT_DIR e manter os dois runs
# separados.
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04i_train_pairflow_star.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 04i_train_pairflow_star.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 04i_train_pairflow_star.sh <work_dir> <shell_b> <n_level>}"

ENSEMBLE_M="${ENSEMBLE_M:-3}"
echo "Treinando PairFlowStar (ensemble em estrela, M=$ENSEMBLE_M) para shell_b=$SHELL_B, n_level=$N_LEVEL"
source "./00_env_common.sh"

RESUME_FLAG=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_FLAG=(--resume-checkpoint "$RESUME_CHECKPOINT")
    echo "RESUME_CHECKPOINT=$RESUME_CHECKPOINT -- retomando explicitamente deste checkpoint"
elif [[ "${NO_RESUME:-0}" == "1" ]]; then
    RESUME_FLAG=(--no-resume)
    echo "NO_RESUME=1 -- ignorando qualquer last.pt existente, comecando do zero"
fi

LR="${LR:-1e-4}"
echo "LR=$LR (default 1e-4)"

INIT_FLAG=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    INIT_FLAG=(--init-checkpoint "$INIT_CHECKPOINT")
    echo "INIT_CHECKPOINT=$INIT_CHECKPOINT -- inicializando flow_net do pre-treino da etapa 4g (checkpoint em .../_pretrained/)"
else
    echo "INIT_CHECKPOINT nao passado -- treinando PairFlowStar do ZERO (controle, sem sufixo '_pretrained')"
fi

FREEZE_FLAG=()
if [[ "${FREEZE_FLOW:-0}" == "1" ]]; then
    if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
        echo "Erro: FREEZE_FLOW=1 requer INIT_CHECKPOINT (nao faz sentido congelar fluxo do zero)"
        exit 1
    fi
    FREEZE_FLAG=(--freeze-flow)
    echo "FREEZE_FLOW=1 -- congelando flow_net, so treinando refine_net/weight_head (checkpoint em .../_frozen/)"
fi

WQC_FLAG=()
if [[ "${WEIGHT_QUALITY_COND:-0}" == "1" ]]; then
    WQC_FLAG=(--weight-quality-cond)
    echo "WEIGHT_QUALITY_COND=1 -- alimentando residual_deg/gap_deg direto na PairFlowWeightHead3D (checkpoint em .../_wqc/)"
fi

ONLY_VALID_FLAG=()
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    ONLY_VALID_FLAG=(--no-only-valid)
    echo "ONLY_VALID=0 -- treinando/validando tambem com alvos cujo par-unico e invalido (checkpoint em .../_inclinv/)"
fi

NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (checkpoint em .../_bn/, exige treino do zero)"
fi

TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
if [[ "$TRIPLETS_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "TRIPLETS_DIR=$TRIPLETS_DIR -- lendo trincas de pasta SEPARADA da producao (subsampling/)"
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_CACHED_SUBJECTS="${MAX_CACHED_SUBJECTS:-6}"
OUT_DIR="$WORK_DIR/pairflow_star_checkpoints"
if [[ "$BATCH_SIZE" != "8" ]]; then
    OUT_DIR="${OUT_DIR}_bs${BATCH_SIZE}"
    echo "BATCH_SIZE=$BATCH_SIZE (default 8) -- checkpoints em $OUT_DIR"
fi
if [[ "$MAX_CACHED_SUBJECTS" != "6" ]]; then
    echo "MAX_CACHED_SUBJECTS=$MAX_CACHED_SUBJECTS (default 6) -- lembrar que a escala de producao (~75-90 sujeitos/worker) nao e' totalmente coberta por esse cache (ver nota de cabecalho)"
fi

FREEZE_ORDER_FLAG=()
if [[ "${FREEZE_SUBJECT_ORDER:-0}" == "1" ]]; then
    FREEZE_ORDER_FLAG=(--freeze-subject-order)
    OUT_DIR="${OUT_DIR}_frozen_order"
    echo "FREEZE_SUBJECT_ORDER=1 -- ativando --freeze-subject-order (checkpoints em $OUT_DIR)"
fi

if [[ -n "${CKPT_TAG:-}" ]]; then
    OUT_DIR="${OUT_DIR}_${CKPT_TAG}"
    echo "CKPT_TAG=$CKPT_TAG -- checkpoints em $OUT_DIR (ver nota de cabecalho: OUT_DIR NAO leva em conta qual INIT_CHECKPOINT foi usado, so' BATCH_SIZE/FREEZE_SUBJECT_ORDER -- use CKPT_TAG pra evitar colidir/sobrescrever um run anterior com pre-treino diferente na MESMA config de BATCH_SIZE/FREEZE_SUBJECT_ORDER)"
fi

WARMUP_STEPS="${WARMUP_STEPS:-0}"
if [[ "$WARMUP_STEPS" != "0" ]]; then
    echo "WARMUP_STEPS=$WARMUP_STEPS -- LR sobe linearmente de 0.1*LR ate LR ao longo dos primeiros $WARMUP_STEPS passos de otimizador"
fi

ZERO_INIT_FLAG=()
if [[ "${ZERO_INIT_REFINE_OUTPUT:-0}" == "1" ]]; then
    ZERO_INIT_FLAG=(--zero-init-refine-output)
    echo "ZERO_INIT_REFINE_OUTPUT=1 -- zerando ultima camada de RefineNet3D (predicao inicial = blend exato)"
fi

ANGULAR_LOSS_WEIGHT="${ANGULAR_LOSS_WEIGHT:-0.0}"
SH_LOSS_HIGH_ORDER_MIN="${SH_LOSS_HIGH_ORDER_MIN:-4}"
SH_LOSS_LMAX_CAP="${SH_LOSS_LMAX_CAP:-8}"
SH_LOSS_Q_OUT="${SH_LOSS_Q_OUT:-16}"
ANGULAR_LOSS_FLAG=()
if [[ "$ANGULAR_LOSS_WEIGHT" != "0.0" && "$ANGULAR_LOSS_WEIGHT" != "0" ]]; then
    ANGULAR_LOSS_FLAG=(--angular-loss-weight "$ANGULAR_LOSS_WEIGHT" \
        --sh-loss-high-order-min "$SH_LOSS_HIGH_ORDER_MIN" --sh-loss-lmax-cap "$SH_LOSS_LMAX_CAP" \
        --sh-loss-q-out "$SH_LOSS_Q_OUT")
    echo "ANGULAR_LOSS_WEIGHT=$ANGULAR_LOSS_WEIGHT (high_order_min=$SH_LOSS_HIGH_ORDER_MIN, lmax_cap=$SH_LOSS_LMAX_CAP, sh_q_out=$SH_LOSS_Q_OUT) -- treino NOVO com loss angular (run_tag ganha sufixo _sh)"
fi

python scripts/04i_train_pairflow_star.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR" \
    --out-dir "$OUT_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" --ensemble-m "$ENSEMBLE_M" \
    --epochs 150 --batch-size "$BATCH_SIZE" --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects "$MAX_CACHED_SUBJECTS" --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    --warmup-steps "$WARMUP_STEPS" \
    "${RESUME_FLAG[@]}" "${INIT_FLAG[@]}" "${FREEZE_FLAG[@]}" "${WQC_FLAG[@]}" \
    "${ONLY_VALID_FLAG[@]}" "${NORM_TYPE_FLAG[@]}" "${FREEZE_ORDER_FLAG[@]}" \
    "${ZERO_INIT_FLAG[@]}" "${ANGULAR_LOSS_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"