#!/bin/bash
#SBATCH --job-name=dmri_rcae_recon
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/recon.%A_%a.err
#SBATCH --output=logs/recon.%A_%a.out
#
# Reconstrucao do conjunto de teste com o RCAE treinado (etapa 5). Mesmo
# esquema de array de slurm/03_train_rcae.sh -- roda depois que o
# checkpoint correspondente ja existe.
#
# Por padrao usa o checkpoint "canonico" (out_dir/best.pt), que e sempre o
# do treino MAIS RECENTE daquele combo (shell_b, n_level) -- rodar
# 03_train_rcae.sh de novo pro mesmo combo sobrescreve esse arquivo. Se
# quiser um checkpoint de um run especifico mais antigo (cada treino
# tambem salva uma copia permanente em rcae_checkpoints/shell*_n*/runs/
# <job_id>/best.pt, nunca sobrescrita), passe o job_id ou um caminho
# explicito via variavel de ambiente:
#
# Uso:
#   sbatch --array=1-N slurm/04_reconstruct_rcae.sh <work_dir>
#   sbatch slurm/04_reconstruct_rcae.sh <work_dir> <shell_b> <n_level>   # sem array
#   CKPT_JOB_ID=10972424_0 sbatch slurm/04_reconstruct_rcae.sh <work_dir> <shell_b> <n_level>
#   CKPT_PATH=/caminho/explicito/best.pt sbatch slurm/04_reconstruct_rcae.sh <work_dir> <shell_b> <n_level>
#
# RECON_SUBJECTS="tag1,tag2" e/ou RECON_LIMIT=1 (variaveis de ambiente)
# restringem a reconstrucao a poucos sujeitos -- util pra testar rapido um
# checkpoint preliminar sem esperar o split de teste inteiro reconstruir.
# 'tag' e subject (ou subject_session, se houver sessao) -- o mesmo que
# aparece nas colunas dos CSVs de metricas/POC. Ex.:
#   RECON_LIMIT=1 sbatch slurm/04_reconstruct_rcae.sh <work_dir> 1000 10
#   RECON_SUBJECTS="20170920171326_616_20170920171326_616" sbatch slurm/04_reconstruct_rcae.sh <work_dir> 1000 10
#
# ATENCAO -- a saida (--out-dir abaixo) e SEMPRE "$WORK_DIR/rcae_recon",
# INDEPENDENTE de qual CKPT_PATH/CKPT_JOB_ID voce passou: reconstruir duas
# vezes pro MESMO shell_b/n_level com checkpoints DIFERENTES sobrescreve a
# reconstrucao anterior, sem deixar rastro de qual .pt gerou o que esta la.
# slurm/05_evaluate_and_downstream.sh so le o que estiver em rcae_recon
# nesse momento -- ele nao sabe (nem tem como saber) qual checkpoint
# reconstruiu aquilo.
#
# RECON_TAG="algumnome" (variavel de ambiente) evita essa sobrescrita:
# grava em "$WORK_DIR/rcae_recon_<tag>" em vez do caminho fixo -- use o
# MESMO RECON_TAG depois em slurm/05_evaluate_and_downstream.sh (mesma
# variavel de ambiente, ver comentario la) pra ele ler dessa pasta rotulada
# em vez da generica. Ex., pra manter varios checkpoints avaliados lado a
# lado no MESMO work_dir sem um apagar o outro:
#   RECON_TAG=job3498743 CKPT_JOB_ID=3498743_0 sbatch slurm/04_reconstruct_rcae.sh <work_dir> 1000 10
#   RECON_TAG=job3498743 sbatch slurm/05_evaluate_and_downstream.sh <work_dir> 1000 10
# Sem RECON_TAG, comportamento identico a antes (rcae_recon fixo).
#
# ANGULAR_LOSS=1 (variavel de ambiente) -- so tem efeito no checkpoint
# CANONICO (quando nem CKPT_PATH nem CKPT_JOB_ID sao passados): aponta pra
# "$WORK_DIR/rcae_checkpoints/shell<B>_n<N>_sh/best.pt" em vez de
# ".../shell<B>_n<N>/best.pt" -- ver scripts/04_train_rcae.py, que agora
# grava a variante com --angular-loss-weight>0 num diretorio SEPARADO
# (sufixo _sh) exatamente pra evitar a colisao de checkpoint entre as duas
# variantes que ja aconteceu uma vez em producao (as duas escrevendo no
# MESMO out_dir/best.pt quando treinadas para o mesmo shell_b/n_level).
#   ANGULAR_LOSS=1 sbatch slurm/04_reconstruct_rcae.sh <work_dir> <shell_b> <n_level>
#
# DECODER_TYPE=sh (variavel de ambiente) -- so tem efeito no checkpoint
# CANONICO, mesma logica de ANGULAR_LOSS acima: aponta pro sufixo _shdec
# (ver DECODER_TYPE em slurm/03_train_rcae.sh e model/rcae.py:Decoder3DSH).
# O proprio scripts/05_reconstruct_rcae.py ja le decoder_type/sh_lmax de
# dentro do checkpoint (nao precisa passar de novo na linha de comando).
#   DECODER_TYPE=sh sbatch slurm/04_reconstruct_rcae.sh <work_dir> <shell_b> <n_level>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04_reconstruct_rcae.sh <work_dir> [shell_b n_level]}"

EXPERIMENTS_TSV="configs/experiments.tsv"

if [[ -n "${2:-}" && -n "${3:-}" ]]; then
    SHELL_B="$2"
    N_LEVEL="$3"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LINE=$(grep -v '^#' "$EXPERIMENTS_TSV" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    SHELL_B=$(echo "$LINE" | cut -f1)
    N_LEVEL=$(echo "$LINE" | cut -f2)
else
    echo "Erro: informe shell_b/n_level ou submeta com --array=1-N"
    exit 1
fi

echo "Reconstruindo (RCAE) para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/rcae_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"
if [[ "${ANGULAR_LOSS:-0}" == "1" ]]; then
    CKPT_DIR="${CKPT_DIR}_sh"
    echo "ANGULAR_LOSS=1 -- lendo checkpoint da variante com loss angular/SH: $CKPT_DIR"
fi
if [[ "${DECODER_TYPE:-direct}" == "sh" ]]; then
    CKPT_DIR="${CKPT_DIR}_shdec"
    echo "DECODER_TYPE=sh -- lendo checkpoint da variante com Decoder3DSH: $CKPT_DIR"
fi
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
    echo "Erro: checkpoint nao encontrado em $CKPT (rode o treino primeiro, ou confira "
    echo "CKPT_JOB_ID/CKPT_PATH -- runs disponiveis em: $CKPT_DIR/runs/)"
    exit 1
fi

SUBJECTS_FLAG=()
if [[ -n "${RECON_SUBJECTS:-}" ]]; then
    SUBJECTS_FLAG=(--subjects "$RECON_SUBJECTS")
    echo "RECON_SUBJECTS=$RECON_SUBJECTS -- restringindo reconstrucao a esse(s) sujeito(s)"
fi
LIMIT_FLAG=()
if [[ -n "${RECON_LIMIT:-}" ]]; then
    LIMIT_FLAG=(--limit "$RECON_LIMIT")
    echo "RECON_LIMIT=$RECON_LIMIT -- restringindo reconstrucao aos primeiros $RECON_LIMIT sujeito(s)"
fi

RECON_OUT_DIR="$WORK_DIR/rcae_recon"
if [[ -n "${RECON_TAG:-}" ]]; then
    RECON_OUT_DIR="$WORK_DIR/rcae_recon_${RECON_TAG}"
    echo "RECON_TAG=$RECON_TAG -- gravando em $RECON_OUT_DIR (nao sobrescreve rcae_recon nem outras tags)"
fi

python scripts/05_reconstruct_rcae.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$RECON_OUT_DIR" \
    --split test --patch-size 24 --stride 16 \
    "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}"