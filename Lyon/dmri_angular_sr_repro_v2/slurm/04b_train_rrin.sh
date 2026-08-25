#!/bin/bash
#SBATCH --job-name=rrincond
#SBATCH --cluster=gpu
#SBATCH --partition=h200
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
python scripts/04b_train_rrin.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/rrin_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size 8 --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    "${RESUME_FLAG[@]}" "${QC_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"