#!/bin/bash
#SBATCH --job-name=rrin_debug_layers
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/debug_rrin_layers.%A_%a.err
#SBATCH --output=logs/debug_rrin_layers.%A_%a.out
#
# Diagnostico dos pesos de camada (pi^(k)) de um checkpoint RRIN3DLayered
# (K>=2, ver model/rrin3d.py, scripts/05c_debug_rrin_layers.py e protocolo
# secao 13, "Toward a layered-flow extension for crossing fibers"). NAO faz
# parte do pipeline principal (treino/reconstrucao/avaliacao) -- e so pra
# inspecionar, num UNICO sujeito, se a rede aprendeu a usar mais de uma
# camada em algum lugar do cerebro.
#
# Roda em CPU por padrao (cluster=htc, sem --gres=gpu) -- um unico sujeito,
# nao precisa de GPU pra ser rapido o bastante. Troque pra --cluster=gpu
# --partition=l40s --gres=gpu:1 (mesmo padrao de slurm/05b_reconstruct_rrin.sh)
# se quiser acelerar.
#
# Uso:
#   NUM_LAYERS=3 sbatch slurm/05c_debug_rrin_layers.sh <work_dir> <shell_b> <n_level> <subject>
#
# NUM_LAYERS=<K> (OBRIGATORIO, sem default) -- precisa bater com o K usado
# no treino (ver NUM_LAYERS em slurm/04b_train_rrin.sh), pra achar o
# checkpoint certo (shell<B>_n<N>_k<K>/). Checkpoints K=1 (arquitetura
# original, RRIN3D) nao tem pesos de camada pra inspecionar -- o proprio
# scripts/05c_debug_rrin_layers.py recusa rodar nesse caso.
#
# USE_QUALITY_COND=1 / ONLY_VALID=0 -- mesma convencao de
# slurm/04b_train_rrin.sh e slurm/05b_reconstruct_rrin.sh, pra achar o
# checkpoint certo quando a variante em camadas foi combinada com
# quality-conditioning e/ou --no-only-valid (sufixos _qc/_inclinv no
# diretorio, na mesma ordem que 04b_train_rrin.py monta o run_tag).
#
# CKPT_PATH=<caminho> -- usa esse checkpoint diretamente, ignorando toda a
# logica de resolucao de diretorio acima (mesma convencao de 05b).
# CKPT_JOB_ID=<job_id> -- usa o checkpoint de um run especifico
# (CKPT_DIR/runs/<job_id>/best.pt) em vez do best.pt canonico.
#
# TARGETS="0,10,20" -- passa --targets pro script python (default: ate 6
# alvos espacados uniformemente, ver scripts/05c_debug_rrin_layers.py).
# ALL_TARGETS=1 -- passa --all-targets (roda TODOS os alvos, cuidado com o
# numero de arquivos gerados: n_target * K volumes 3D no total).
# GFA_PATH=<caminho> -- passa --gfa-path (correlacao opcional com um mapa
# de GFA calculado da aquisicao COMPLETA do sujeito, ver docstring do
# script python sobre nao circularizar a evidencia).
# DEBUG_OUT_DIR=<caminho> -- onde salvar (default: $WORK_DIR/rrin_layer_debug).
#
# NORM_TYPE=batch (default instance) -- so ajuda a achar o CKPT_DIR certo
# (sufixo _bn) quando o checkpoint foi treinado com --norm-type batch (ver
# NORM_TYPE em slurm/04b_train_rrin.sh) -- a variante que resolve o
# artefato de costura de patch de vez (BatchNorm3d usa estatisticas fixas
# em eval(), independentes do patch), em vez de so atenuar via STRIDE menor.
#
# ANGULAR_LOSS=1 (default 0) -- so ajuda a achar o CKPT_DIR certo (sufixo
# _sh, ver ANGULAR_LOSS_WEIGHT em slurm/04b_train_rrin.sh e
# utils/sh_angular_loss.py) quando o checkpoint foi treinado com
# --angular-loss-weight > 0 -- a loss angular so afeta o treino, nao muda
# a arquitetura nem os mapas pi^(k) inspecionados aqui.
#
# STRIDE=<N> / PATCH_SIZE=<N> (default 8 / 10, mesma convencao de
# slurm/05b_reconstruct_rrin.sh) -- se voce esta testando a hipotese de que
# o padrao listrado nos mapas pi^(k) e um artefato de costura de patch
# (pouca sobreposicao + InstanceNorm3d por patch, ver protocolo), baixar o
# STRIDE aqui e comparar o mapa de efflayers antes/depois e a forma mais
# direta de checar isso -- ex.:
#   STRIDE=4 NUM_LAYERS=2 sbatch slurm/05c_debug_rrin_layers.sh <work_dir> <shell_b> <n_level> <subject>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05c_debug_rrin_layers.sh <work_dir> <shell_b> <n_level> <subject>}"
SHELL_B="${2:?uso: sbatch 05c_debug_rrin_layers.sh <work_dir> <shell_b> <n_level> <subject>}"
N_LEVEL="${3:?uso: sbatch 05c_debug_rrin_layers.sh <work_dir> <shell_b> <n_level> <subject>}"
SUBJECT="${4:?uso: sbatch 05c_debug_rrin_layers.sh <work_dir> <shell_b> <n_level> <subject>}"

NUM_LAYERS="${NUM_LAYERS:?defina NUM_LAYERS=<K> (K>=2) -- precisa bater com o checkpoint ja treinado, ver NUM_LAYERS em slurm/04b_train_rrin.sh}"
if [[ "$NUM_LAYERS" == "1" ]]; then
    echo "Erro: NUM_LAYERS=1 nao tem pesos de camada pra inspecionar (arquitetura RRIN3D original, sem RRIN3DLayered)."
    exit 1
fi

echo "Depurando camadas (RRIN3DLayered, K=$NUM_LAYERS) para shell_b=$SHELL_B, n_level=$N_LEVEL, sujeito=$SUBJECT"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/rrin_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_qc"
    echo "USE_QUALITY_COND=1 -- lendo checkpoint da variante consciente da qualidade"
fi
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    CKPT_DIR="${CKPT_DIR}_inclinv"
    echo "ONLY_VALID=0 -- lendo checkpoint treinado tambem com trincas invalidas"
fi
CKPT_DIR="${CKPT_DIR}_k${NUM_LAYERS}"
if [[ "${NORM_TYPE:-instance}" == "batch" ]]; then
    CKPT_DIR="${CKPT_DIR}_bn"
    echo "NORM_TYPE=batch -- lendo checkpoint da variante com BatchNorm3d (resolve de vez o "
    echo "artefato de costura entre patches, ver model/rrin3d.py:_norm3d e protocolo)"
fi
if [[ "${ANGULAR_LOSS:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_sh"
    echo "ANGULAR_LOSS=1 -- lendo checkpoint da variante treinada com a loss angular/SH"
fi
echo "Diretorio de checkpoint: $CKPT_DIR"

if [[ -n "${CKPT_PATH:-}" ]]; then
    CKPT="$CKPT_PATH"
    echo "Usando checkpoint explicito (CKPT_PATH): $CKPT"
elif [[ -n "${CKPT_JOB_ID:-}" ]]; then
    CKPT="$CKPT_DIR/runs/$CKPT_JOB_ID/best.pt"
    echo "Usando checkpoint do run job_id=$CKPT_JOB_ID: $CKPT"
else
    CKPT="$CKPT_DIR/best.pt"
    echo "Usando checkpoint canonico (mais recente): $CKPT"
fi
if [[ ! -f "$CKPT" ]]; then
    echo "Erro: checkpoint nao encontrado em $CKPT"
    echo "  Rode o treino primeiro, ex.: NUM_LAYERS=$NUM_LAYERS sbatch slurm/04b_train_rrin.sh $WORK_DIR $SHELL_B $N_LEVEL"
    echo "  Ou confira CKPT_JOB_ID/CKPT_PATH -- runs disponiveis em: $CKPT_DIR/runs/"
    exit 1
fi

TARGETS_FLAG=()
if [[ -n "${TARGETS:-}" ]]; then
    TARGETS_FLAG=(--targets "$TARGETS")
    echo "TARGETS=$TARGETS -- restringindo mapas pi^(k) individuais a esses alvos"
fi
ALL_TARGETS_FLAG=()
if [[ "${ALL_TARGETS:-0}" == "1" ]]; then
    ALL_TARGETS_FLAG=(--all-targets)
    echo "ALL_TARGETS=1 -- processando TODOS os alvos (pode gerar bastante arquivo)"
fi
GFA_FLAG=()
if [[ -n "${GFA_PATH:-}" ]]; then
    GFA_FLAG=(--gfa-path "$GFA_PATH")
    echo "GFA_PATH=$GFA_PATH -- calculando correlacao com efflayers_mean_alltargets"
fi

DEBUG_OUT_DIR="${DEBUG_OUT_DIR:-$WORK_DIR/rrin_layer_debug}"
echo "Gravando diagnostico em: $DEBUG_OUT_DIR"

STRIDE="${STRIDE:-8}"
PATCH_SIZE="${PATCH_SIZE:-10}"
if [[ "$STRIDE" != "8" || "$PATCH_SIZE" != "10" ]]; then
    echo "STRIDE=$STRIDE PATCH_SIZE=$PATCH_SIZE (default seria patch-size=10 stride=8)"
fi

python scripts/05c_debug_rrin_layers.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --subject "$SUBJECT" \
    --out-dir "$DEBUG_OUT_DIR" \
    --patch-size "$PATCH_SIZE" --stride "$STRIDE" \
    "${TARGETS_FLAG[@]}" "${ALL_TARGETS_FLAG[@]}" "${GFA_FLAG[@]}"