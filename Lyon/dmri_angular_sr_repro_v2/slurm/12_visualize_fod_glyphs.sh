#!/bin/bash
#SBATCH --job-name=dmri_fod_glyphs
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/fod_glyphs.%J.err
#SBATCH --output=logs/fod_glyphs.%J.out
#
# Etapa 12 (diagnostico visual): reconstroi via
# scripts/12_visualize_fod_glyphs.py -- ajusta CSD (ConstrainedSphericalDeconvModel/
# auto_response_ssst/peaks_from_model, mesma convencao de
# scripts/11_peak_confusion_by_roi.py) num sub-volume cubico centrado no
# centroide da mascara, ACHA AUTOMATICAMENTE uma regiao de cruzamento de
# fibras genuino ali dentro (janela --patch-size x --patch-size com maior
# fracao de voxels com >=2 picos no ground truth) e desenha, lado a lado,
# o glifo 2D do FOD (perfil de amplitude no plano da fatia, 72 direcoes
# por padrao) do ground truth e de cada metodo de reconstrucao passado
# via --baseline-dir/--rcae-dir/--extra-method.
#
# So numpy/nibabel/dipy/matplotlib, sem GPU -- CSD e' o unico passo caro,
# mas roda so no sub-volume (--search-radius, default 15 -> 31^3), nao no
# cerebro inteiro.
#
# Escopo desta etapa (a pedido da usuaria, 2026-08-31): SO glifos por
# enquanto ("Só glifos primeiro"), tratografia fica pra depois. Regiao
# escolhida automaticamente ("Região de cruzamento genérica"), nao presa a
# nenhum trato JHU especifico -- se quiser comparar no mesmo lugar de
# scripts/11_peak_confusion_by_roi.py, ajuste --search-radius/--slice-axis
# manualmente, mas o script nao aceita ROI por nome de trato.
#
# Uso:
#   EXTRA_METHOD="naive_blend=<dir>,naive_ensemble_blend=<dir>,rrin_n16_star610=<dir>,rcae_n16=<dir>" \
#     sbatch slurm/12_visualize_fod_glyphs.sh <work_dir> <shell_b> <n_level> <baseline_dir> [rcae_dir]
#
# Variaveis opcionais (todas com default sensato):
#   SPLIT (default: test)
#   SUBJECTS            -- tag(s) separadas por virgula (default: primeiro sujeito do split)
#   SEARCH_RADIUS        (default: 15)
#   PATCH_SIZE            (default: 4)
#   SLICE_AXIS           (default: 2 = axial; 0=sagital, 1=coronal)
#   MIN_PEAKS_FOR_CROSSING (default: 2)
#   MIN_MASK_FRAC         (default: 0.5)
#   NORMALIZE            (default: global; ou per_voxel)
#   OUT_FILE             (default: $WORK_DIR/figures/fod_glyphs_shell<b>_n<n>.png)

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 12_visualize_fod_glyphs.sh <work_dir> <shell_b> <n_level> <baseline_dir> [rcae_dir]}"
SHELL_B="${2:?uso: sbatch 12_visualize_fod_glyphs.sh <work_dir> <shell_b> <n_level> <baseline_dir> [rcae_dir]}"
N_LEVEL="${3:?uso: sbatch 12_visualize_fod_glyphs.sh <work_dir> <shell_b> <n_level> <baseline_dir> [rcae_dir]}"
BASELINE_DIR="${4:?uso: sbatch 12_visualize_fod_glyphs.sh <work_dir> <shell_b> <n_level> <baseline_dir> [rcae_dir]}"
RCAE_DIR="${5:-}"

source "./00_env_common.sh"

SPLIT="${SPLIT:-test}"
OUT_FILE="${OUT_FILE:-$WORK_DIR/figures/fod_glyphs_shell${SHELL_B%.*}_n${N_LEVEL}.png}"

SUBJECTS_FLAG=()
if [[ -n "${SUBJECTS:-}" ]]; then
    SUBJECTS_FLAG=(--subjects "$SUBJECTS")
    echo "SUBJECTS=$SUBJECTS -- restringindo a esse(s) sujeito(s)"
fi

RCAE_FLAG=()
if [[ -n "$RCAE_DIR" ]]; then
    RCAE_FLAG=(--rcae-dir "$RCAE_DIR")
fi

EXTRA_FLAGS=()
if [[ -n "${EXTRA_METHOD:-}" ]]; then
    EXTRA_FLAGS=(--extra-method "$EXTRA_METHOD")
    echo "EXTRA_METHOD=$EXTRA_METHOD"
fi

python scripts/12_visualize_fod_glyphs.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --baseline-dir "$BASELINE_DIR" \
    "${RCAE_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --split "$SPLIT" \
    "${SUBJECTS_FLAG[@]}" \
    --search-radius "${SEARCH_RADIUS:-15}" \
    --patch-size "${PATCH_SIZE:-4}" \
    --slice-axis "${SLICE_AXIS:-2}" \
    --min-peaks-for-crossing "${MIN_PEAKS_FOR_CROSSING:-2}" \
    --min-mask-frac "${MIN_MASK_FRAC:-0.5}" \
    --normalize "${NORMALIZE:-global}" \
    --out "$OUT_FILE"