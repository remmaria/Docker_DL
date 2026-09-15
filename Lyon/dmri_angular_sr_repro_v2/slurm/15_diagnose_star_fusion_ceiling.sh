#!/bin/bash
#SBATCH --job-name=star_fusion_ceiling
#SBATCH --cluster=gpu
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/star_fusion_ceiling.%A_%a.err
#SBATCH --output=logs/star_fusion_ceiling.%A_%a.out
#
# Etapa 15 (DIAGNOSTICO, nao treina nada) -- mede o TETO da fusao do ensemble
# em estrela a partir de um checkpoint JA EXISTENTE, para decidir entre duas
# direcoes de trabalho muito diferentes (ver docstring de
# scripts/15_diagnose_star_fusion_ceiling.py):
#   (A) melhorar a CABECA DE FUSAO  -> item 1, CROSS_CANDIDATE_ATTENTION=1,
#       ja implementado e barato de testar;
#   (B) melhorar as PREDICOES CANDIDATAS -> contexto angular global na linha
#       de fluxo (FlowNet3D so ve 2 das n_level direcoes medidas), bem mais
#       caro de implementar.
#
# A fusao e uma combinacao convexa por voxel, entao existe um teto EXATO e
# calculavel: se o alvo cai dentro do intervalo coberto pelos M candidatos,
# alguma combinacao acerta exato; se cai fora, nenhuma alcanca. Esse teto
# ("oraculo convexo") e um limite inferior para QUALQUER cabeca de fusao,
# presente ou futura -- se ele ja estiver acima do alvo de ~0,026 do RCAE, a
# direcao (A) esta matematicamente descartada como forma de fechar o gap.
#
# Job barato (45 min, 1 GPU, sem escrita em nenhum diretorio de checkpoint) --
# seguro de rodar com treinos em andamento.
#
# Uso:
#   CKPT=<caminho/best.pt> sbatch slurm/15_diagnose_star_fusion_ceiling.sh \
#       <work_dir> <shell_b> <n_level>
#
# MODEL=rrin_star (default) ou MODEL=pairflow_star -- qual linha e o
#   checkpoint (as duas tem a mesma assinatura de forward).
# TRIPLETS_DIR=<dir> (default $WORK_DIR/subsampling) -- PRECISA ser o MESMO
#   usado no treino que gerou o checkpoint, senao os candidatos do feixe sao
#   outros e a comparacao nao vale.
# MAX_BATCHES=<N> (default 60), SPLIT=<split> (default val),
# ONLY_VALID=0 (default 1) -- use o MESMO valor do treino comparado.
# OUT_CSV=<arquivo> (opcional) -- medias por batch, pra conferir estabilidade.
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: CKPT=... sbatch 15_diagnose_star_fusion_ceiling.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: CKPT=... sbatch 15_diagnose_star_fusion_ceiling.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: CKPT=... sbatch 15_diagnose_star_fusion_ceiling.sh <work_dir> <shell_b> <n_level>}"
CKPT="${CKPT:?informe CKPT=<caminho do best.pt/last.pt do treino star>}"

MODEL="${MODEL:-rrin_star}"
TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
MAX_BATCHES="${MAX_BATCHES:-60}"
SPLIT="${SPLIT:-val}"

echo "Diagnostico de teto de fusao: modelo=$MODEL shell_b=$SHELL_B n_level=$N_LEVEL"
echo "  checkpoint:   $CKPT"
echo "  triplets_dir: $TRIPLETS_DIR (PRECISA bater com o do treino que gerou o checkpoint)"
source "./00_env_common.sh"

ONLY_VALID_FLAG=()
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    ONLY_VALID_FLAG=(--no-only-valid)
    echo "ONLY_VALID=0 -- incluindo tambem alvos cujo par-unico e invalido"
fi
OUT_CSV_FLAG=()
if [[ -n "${OUT_CSV:-}" ]]; then
    OUT_CSV_FLAG=(--out-csv "$OUT_CSV")
fi

python scripts/15_diagnose_star_fusion_ceiling.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR" \
    --checkpoint "$CKPT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --model "$MODEL" --split "$SPLIT" \
    --patch-size 10 --batch-size 8 --num-workers 4 \
    --max-batches "$MAX_BATCHES" \
    "${ONLY_VALID_FLAG[@]}" "${OUT_CSV_FLAG[@]}"