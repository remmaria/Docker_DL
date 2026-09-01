#!/bin/bash
#SBATCH --job-name=dmri_naive_ens_blend
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/naive_ens_blend.%J.err
#SBATCH --output=logs/naive_ens_blend.%J.out
#
# Etapa 5h (baseline "burro" com o MESMO pool geometrico do ensemble em
# estrela, mas SEM rede nenhuma e SEM fusao aprendida): reconstroi via
# scripts/05h_reconstruct_naive_ensemble_blend.py -- media UNIFORME dos M
# blends ingenuos de cada par do feixe `{key}__ens_*`, em vez do par unico
# (isso e' o 05g) ou da fusao aprendida por voxel (isso e' o RRIN3DStar,
# 04e/05f). Serve pra isolar se o ganho do RRIN3DStar vem do POOL
# geometrico rico (mais candidatos diversos pra mediar, efeito tipo
# "bagging") ou da REDE (warp de verdade + fusao aprendida) -- ver
# addendum, secao 19/20. So numpy/nibabel, sem GPU -- roda rapido.
#
# IMPORTANTE: use o MESMO --triplets-dir (mesmo TRIPLETS_DIR) usado pra
# treinar o checkpoint do RRIN3DStar que voce quer comparar (ex.: o `.npz`
# gerado com ENSEMBLE_M=6 ENSEMBLE_MAX_RESIDUAL_DEG=10, pra comparar
# contra o `star610`) -- senao a comparacao reintroduz a confusao entre
# "pool diferente" e "fusao diferente" que este script existe pra evitar.
#
# Depois de rodar, use --extra-method naive_ensemble_blend=<out_dir> em
# 06_evaluate_reconstruction.py / 07_downstream_dti_noddi.py, ao lado de
# naive_blend=<out_dir> (05g) e rrin_n16_star610=<out_dir> (05f), pra
# comparar os tres.
#
# Uso:
#   TRIPLETS_DIR=<pasta com os ens_* do M/residuo que voce quer testar> \
#     sbatch slurm/05h_reconstruct_naive_ensemble_blend.sh <work_dir> <shell_b> <n_level>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 05h_reconstruct_naive_ensemble_blend.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 05h_reconstruct_naive_ensemble_blend.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 05h_reconstruct_naive_ensemble_blend.sh <work_dir> <shell_b> <n_level>}"

source "./00_env_common.sh"

SPLIT="${SPLIT:-test}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/naive_ensemble_blend_recon}"
TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
if [[ "$TRIPLETS_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "TRIPLETS_DIR=$TRIPLETS_DIR -- lendo trincas/feixe de pasta SEPARADA da producao (subsampling/)"
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

python scripts/05h_reconstruct_naive_ensemble_blend.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out-dir "$OUT_DIR" \
    --split "$SPLIT" \
    "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}"