#!/bin/bash
#SBATCH --job-name=dmri_star_pair_weights
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/star_pair_weights.%J.err
#SBATCH --output=logs/star_pair_weights.%J.out
#
# Etapa 13 (diagnostico): roda scripts/13_inspect_star_pair_weights.py --
# inspeciona os pesos de fusao (`pi`, PairWeightHead3D) de uma RRIN3DStar
# treinada, num UNICO voxel especifico, para todos os alvos held-out do
# feixe de trincas. So um forward num patch pequeno (nao reconstroi o
# volume inteiro) -- CPU basta, roda em minutos.
#
# Motivacao: checar se um voxel de cruzamento "deformado" na figura de
# glifos (scripts/12_visualize_fod_glyphs.py) tem peso de fusao concentrado
# num unico par candidato que discorda dos demais (ver docstring do script
# python para a hipotese completa).
#
# Uso:
#   sbatch slurm/13_inspect_star_pair_weights.sh <work_dir> <checkpoint> \
#     <shell_b> <n_level> <subject> <voxel_x,y,z>
#
# Variaveis opcionais:
#   TRIPLETS_DIR (default: $WORK_DIR/subsampling) -- mesma convencao de
#                 slurm/05f_reconstruct_rrin_star.sh, pra quando as trincas
#                 usadas no treino/avaliacao ficam numa pasta separada.
#   PATCH_SIZE (default: 10, mesmo default de 05f_reconstruct_rrin_star.sh)
#   TOP         -- se setado, imprime so os TOP alvos com maior concentracao
#                  de peso (peso_max), pra focar nos casos mais extremos
#   OUT_FILE    (default: $WORK_DIR/diagnostics/pair_weights_<subject>_vox<x>-<y>-<z>.csv)

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 13_inspect_star_pair_weights.sh <work_dir> <checkpoint> <shell_b> <n_level> <subject> <voxel_x,y,z>}"
CHECKPOINT="${2:?uso: sbatch 13_inspect_star_pair_weights.sh <work_dir> <checkpoint> <shell_b> <n_level> <subject> <voxel_x,y,z>}"
SHELL_B="${3:?uso: sbatch 13_inspect_star_pair_weights.sh <work_dir> <checkpoint> <shell_b> <n_level> <subject> <voxel_x,y,z>}"
N_LEVEL="${4:?uso: sbatch 13_inspect_star_pair_weights.sh <work_dir> <checkpoint> <shell_b> <n_level> <subject> <voxel_x,y,z>}"
SUBJECT="${5:?uso: sbatch 13_inspect_star_pair_weights.sh <work_dir> <checkpoint> <shell_b> <n_level> <subject> <voxel_x,y,z>}"
VOXEL="${6:?uso: sbatch 13_inspect_star_pair_weights.sh <work_dir> <checkpoint> <shell_b> <n_level> <subject> <voxel_x,y,z>}"

source "./00_env_common.sh"

VOXEL_TAG="${VOXEL//,/-}"
OUT_FILE="${OUT_FILE:-$WORK_DIR/diagnostics/pair_weights_${SUBJECT}_vox${VOXEL_TAG}.csv}"
TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
if [[ "$TRIPLETS_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "TRIPLETS_DIR=$TRIPLETS_DIR -- lendo trincas de pasta SEPARADA da producao (subsampling/)"
fi

TOP_FLAG=()
if [[ -n "${TOP:-}" ]]; then
    TOP_FLAG=(--top "$TOP")
    echo "TOP=$TOP -- mostrando so os $TOP alvos mais concentrados"
fi

python scripts/13_inspect_star_pair_weights.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR" \
    --checkpoint "$CHECKPOINT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --subject "$SUBJECT" \
    --voxel "$VOXEL" \
    --patch-size "${PATCH_SIZE:-10}" \
    "${TOP_FLAG[@]}" \
    --out "$OUT_FILE"