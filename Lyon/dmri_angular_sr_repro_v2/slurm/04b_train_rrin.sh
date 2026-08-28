#!/bin/bash
#SBATCH --job-name=rrin_ok
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
# SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train_rrin.%A_%a.err
#SBATCH --output=logs/train_rrin.%A_%a.out
#
# Treino da RRIN3D (etapa 4b, ver scripts/04b_train_rrin.py e protocolo
# secao 10.1) para um (shell_b, n_level) especifico -- requer que
# scripts/02b_build_rrin_triplets.py ja tenha rodado pra esse work_dir.
# Mesmo padrao de slurm/03_train_rcae.sh (array de configs/experiments.tsv
# OU shell_b/n_level explicitos).
#
# Uso:
#   sbatch --array=1-N slurm/04b_train_rrin.sh <work_dir>
#   sbatch slurm/04b_train_rrin.sh <work_dir> <shell_b> <n_level>
#
# Resume automatico (ver scripts/04b_train_rrin.py) -- mesmo mecanismo do
# RCAE: RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1 (variaveis de ambiente).
#
# LR=<valor> (variavel de ambiente, default 1e-3 se nao setada) -- default
# ja bate com o --lr 1e-3 usado em slurm/03_train_rcae.sh (o default
# original aqui era 1e-4, 10x menor por um esquecimento -- ver historico do
# protocolo secao 10.2/10.3: LRs diferentes confundiam "a hipotese de fluxo
# optico e mais fraca" com "essa rede so aprendeu mais devagar". Corrigido
# alinhando os dois defaults). Se quiser rodar com outro valor:
#   LR=1e-4 sbatch slurm/04b_train_rrin.sh <work_dir> <shell_b> <n_level>
# Qualquer LR != 1e-3 grava em out_dir/shell<B>_n<N>[_variante]_lr<valor>/
# (sufixo automatico, ver scripts/04b_train_rrin.py) -- NAO colide com o
# checkpoint da variante com LR=1e-3 (ex.: o run "cego" n16 em andamento),
# pode rodar em paralelo (job sbatch independente) sem sobrescrever nada.
# Ex.: pra comparar convergencia com LR maior no MESMO n16 cego:
#   LR=2e-3 sbatch slurm/04b_train_rrin.sh <work_dir> 1000 16
#
# USE_QUALITY_COND=1 (variavel de ambiente) -- liga --use-quality-cond (ver
# protocolo secao 10.1/model/rrin3d.py): condiciona a FlowNet3D em
# residual_deg/gap_deg da trinca, testando se a rede compensa geometria
# ruim quando sabe da qualidade da trinca, em vez de so filtrar trincas
# ruins fora do treino ("teste cego", o default). Grava em
# out_dir/shell<B>_n<N>_qc/ (sufixo automatico, ver scripts/04b_train_rrin.py)
# -- NAO colide com o checkpoint da variante cega (shell<B>_n<N>/), pode
# rodar as duas em paralelo (jobs sbatch independentes) sem risco de uma
# sobrescrever o best.pt/last.pt da outra.
#   USE_QUALITY_COND=1 sbatch slurm/04b_train_rrin.sh <work_dir> <shell_b> <n_level>
#
# ONLY_VALID=0 (variavel de ambiente, default 1) -- liga --no-only-valid (ver
# scripts/04b_train_rrin.py e protocolo secao 10.1): treina/valida tambem com
# trincas INVALIDAS, em vez de so as validas (default). Motivado por
# rrin/rrin_qc produzirem NMSE ~1e9-1e11 (explosao numerica) nos alvos
# invalidos durante a reconstrucao -- a rede nunca viu geometria parecida no
# treino porque only_valid=True (default) filtra essas trincas fora. Grava
# em out_dir/shell<B>_n<N>_inclinv/ (sufixo automatico, nao colide com as
# outras variantes).
#   ONLY_VALID=0 sbatch slurm/04b_train_rrin.sh <work_dir> <shell_b> <n_level>
#
# NUM_LAYERS=<K> (variavel de ambiente, default 1) -- liga --num-layers K (ver
# model/rrin3d.py:RRIN3DLayered e protocolo secao 13, "Toward a layered-flow
# extension for crossing fibers"). K=1 (default) usa a arquitetura ORIGINAL
# (RRIN3D, um unico fluxo bidirecional + 1 mapa de visibilidade). K>=2 usa
# RRIN3DLayered: cada camada tem seu proprio par de fluxo e visibilidade, e
# as K camadas sao combinadas por um softmax POR VOXEL (o K em si e fixo pra
# toda a imagem; o que varia por voxel e o peso aprendido de cada camada).
# Comece SEM nenhuma supervisao auxiliar -- so compare K=1 vs K=2 vs K=3 via
# aggregate_valid/aggregate_invalid (scripts/06_evaluate_reconstruction.py).
# Grava em out_dir/shell<B>_n<N>_k<K>/ (sufixo automatico, nao colide com as
# outras variantes).
#   NUM_LAYERS=2 sbatch slurm/04b_train_rrin.sh <work_dir> <shell_b> <n_level>
#
# NORM_TYPE=batch (variavel de ambiente, default instance) -- liga
# --norm-type batch (ver model/rrin3d.py:_norm3d e protocolo, "artefato de
# patch-tiling/InstanceNorm3d"): troca InstanceNorm3d por BatchNorm3d em
# toda a rede. InstanceNorm3d (default) calcula estatisticas POR PATCH, o
# que causa uma "costura" visivel entre patches na reconstrucao com
# sliding-window (confirmado: STRIDE menor em slurm/05b_reconstruct_rrin.sh
# atenua bastante, mas so dilui o efeito, nao remove a causa). BatchNorm3d
# em eval() usa estatisticas FIXAS (running_mean/running_var acumuladas em
# todo o treino), entao a reconstrucao fica livre desse artefato por
# construcao. Exige treinar do ZERO -- nao da pra retomar um checkpoint
# "instance" com NORM_TYPE=batch (parametros/buffers incompativeis,
# scripts/04b_train_rrin.py levanta erro se tentar). Grava em
# out_dir/shell<B>_n<N>[_qc][_inclinv][_k<K>]_bn/ (sufixo automatico, nao
# colide com as variantes "instance" existentes).
#   NORM_TYPE=batch NO_RESUME=1 sbatch slurm/04b_train_rrin.sh <work_dir> <shell_b> <n_level>
#
# ANGULAR_LOSS_WEIGHT=<valor> (variavel de ambiente, default 0.0 = desligado)
# -- liga --angular-loss-weight (ver utils/sh_angular_loss.py e protocolo
# secao 14.5 item 2): porte pro RRIN da mesma loss angular/SH que ja ajudou
# o RCAE a superar o baseline_sh (ver scripts/04_train_rcae.py e protocolo
# secao 9). Alem da MAE de sinal normal, penaliza tambem o erro nos
# coeficientes SH de ordem alta (l>=SH_LOSS_HIGH_ORDER_MIN) de um FEIXE de
# SH_LOSS_Q_OUT trincas amostradas do MESMO sujeito/patch a cada passo (ver
# utils/rrin_dataset.py:RRINTripletDataset(sh_q_out=...) e
# scripts/04b_train_rrin.py:_sh_bundle_forward) -- a rede continua prevendo
# UMA direcao por chamada (nao muda a arquitetura), so o forward e chamado
# uma vez em lote (B*K) pra montar o ajuste SH por item. Grava em
# out_dir/shell<B>_n<N>[_qc][_inclinv][_k<K>][_bn]_sh/ (sufixo automatico,
# nao colide com as variantes sem a loss angular -- resumir um checkpoint
# so-sinal com ANGULAR_LOSS_WEIGHT>0 funciona normalmente, so muda a loss,
# ao contrario de NORM_TYPE que exige treino do zero).
#   ANGULAR_LOSS_WEIGHT=0.1 sbatch slurm/04b_train_rrin.sh <work_dir> <shell_b> <n_level>
#
# SH_LOSS_Q_OUT=<N> (default 16, so tem efeito com ANGULAR_LOSS_WEIGHT>0) --
# tamanho do feixe de trincas amostradas por passo (ver --sh-loss-q-out).
# SH_LOSS_HIGH_ORDER_MIN=<l> (default 4) / SH_LOSS_LMAX_CAP=<l> (default 8)
# -- ordem minima penalizada e teto de ordem do ajuste SH (ver
# --sh-loss-high-order-min / --sh-loss-lmax-cap e o docstring de
# compute_sh_angular_loss sobre o piso de direcoes por ordem: l=4 precisa
# SH_LOSS_Q_OUT>=15, l=6 precisa >=28, l=8 precisa >=45).
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04b_train_rrin.sh <work_dir> [shell_b n_level]}"
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
echo "Treinando RRIN3D para shell_b=$SHELL_B, n_level=$N_LEVEL"
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
QC_FLAG=()
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    QC_FLAG=(--use-quality-cond)
    echo "USE_QUALITY_COND=1 -- treinando a variante consciente da qualidade da trinca (checkpoint em shell${SHELL_B%.*}_n${N_LEVEL}_qc/)"
fi
ONLY_VALID_FLAG=()
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    ONLY_VALID_FLAG=(--no-only-valid)
    echo "ONLY_VALID=0 -- treinando/validando tambem com trincas invalidas (checkpoint em shell${SHELL_B%.*}_n${N_LEVEL}_inclinv/)"
fi
NUM_LAYERS="${NUM_LAYERS:-1}"
NUM_LAYERS_FLAG=()
if [[ "$NUM_LAYERS" != "1" ]]; then
    NUM_LAYERS_FLAG=(--num-layers "$NUM_LAYERS")
    echo "NUM_LAYERS=$NUM_LAYERS -- treinando a variante em camadas RRIN3DLayered (checkpoint em shell${SHELL_B%.*}_n${N_LEVEL}_k${NUM_LAYERS}/)"
fi
NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (checkpoint em .../_bn/, exige treino do zero)"
fi
ANGULAR_LOSS_WEIGHT="${ANGULAR_LOSS_WEIGHT:-0.0}"
ANGULAR_LOSS_FLAG=()
SH_LOSS_FLAG=()
if [[ "$ANGULAR_LOSS_WEIGHT" != "0.0" && "$ANGULAR_LOSS_WEIGHT" != "0" ]]; then
    ANGULAR_LOSS_FLAG=(--angular-loss-weight "$ANGULAR_LOSS_WEIGHT")
    echo "ANGULAR_LOSS_WEIGHT=$ANGULAR_LOSS_WEIGHT -- treinando com a loss angular/SH (checkpoint em .../_sh/, ver utils/sh_angular_loss.py)"
    if [[ -n "${SH_LOSS_Q_OUT:-}" ]]; then
        SH_LOSS_FLAG+=(--sh-loss-q-out "$SH_LOSS_Q_OUT")
        echo "SH_LOSS_Q_OUT=$SH_LOSS_Q_OUT (default 16)"
    fi
    if [[ -n "${SH_LOSS_HIGH_ORDER_MIN:-}" ]]; then
        SH_LOSS_FLAG+=(--sh-loss-high-order-min "$SH_LOSS_HIGH_ORDER_MIN")
        echo "SH_LOSS_HIGH_ORDER_MIN=$SH_LOSS_HIGH_ORDER_MIN (default 4)"
    fi
    if [[ -n "${SH_LOSS_LMAX_CAP:-}" ]]; then
        SH_LOSS_FLAG+=(--sh-loss-lmax-cap "$SH_LOSS_LMAX_CAP")
        echo "SH_LOSS_LMAX_CAP=$SH_LOSS_LMAX_CAP (default 8)"
    fi
fi
python scripts/04b_train_rrin.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/rrin_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size 8 --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    "${RESUME_FLAG[@]}" "${QC_FLAG[@]}" "${ONLY_VALID_FLAG[@]}" "${NUM_LAYERS_FLAG[@]}" \
    "${NORM_TYPE_FLAG[@]}" "${ANGULAR_LOSS_FLAG[@]}" "${SH_LOSS_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"