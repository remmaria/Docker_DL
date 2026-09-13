#!/bin/bash
#SBATCH --job-name=dmri_implicit_pooling
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/implicit_pooling.%J.err
#SBATCH --output=logs/implicit_pooling.%J.out
#
# Etapa 14 (diagnostico): roda scripts/14_diagnose_implicit_pooling.py --
# checa, SEM RETREINAR NADA (so forward passes com um checkpoint ja
# existente), se o mean-pooling de ImplicitAngularModel3D.encode esta
# jogando fora informacao real entre as n_level direcoes de entrada, antes
# de decidir se vale a pena implementar uma agregacao aprendida (atencao).
# Poucos patches (--n-patches, default 8), CPU basta -- roda em segundos a
# poucos minutos, nao compete por GPU com os treinos longos em andamento.
#
# Ver docstring de scripts/14_diagnose_implicit_pooling.py para a hipotese
# completa e o criterio de leitura das duas metricas (colapso de agregacao
# e sensibilidade leave-one-out).
#
# Uso:
#   sbatch slurm/14_diagnose_implicit_pooling.sh <work_dir> <checkpoint> \
#     <shell_b> <n_level>
#
# Ex. (checkpoint precoce, epoca 13, discutido no addendum 2026-09-03):
#   sbatch slurm/14_diagnose_implicit_pooling.sh \
#     /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir \
#     $WORK_DIR/implicit_checkpoints/shell1000_n16/best.pt \
#     1000 16
#
# Variaveis opcionais:
#   SCHEME_DIR   (default: $WORK_DIR/subsampling) -- mesma pasta usada por
#                04f_train_implicit.sh (--scheme-dir).
#   N_PATCHES    (default: 8) -- quantos patches de validacao amostrar no
#                total (espacados dentro de cada sujeito escolhido, ver
#                MAX_SUBJECTS). Suba se quiser mais confianca.
#   MAX_SUBJECTS (default: 2) -- a quantos sujeitos DISTINTOS os N_PATCHES
#                sao distribuidos. Cada sujeito novo tocado forca a leitura
#                do volume 4D inteiro do disco (pode levar bastante tempo em
#                storage de cluster) -- manter isto baixo (2-3) evita ficar
#                minutos esperando sem log nenhum antes do primeiro
#                resultado. O script agora imprime progresso por patch
#                (sujeito, tempo de carga, tempo de forward) -- se aparecer
#                so a linha de "amostrando N patches" por muito tempo, o
#                job provavelmente esta preso lendo o PRIMEIRO sujeito do
#                disco (storage lento), nao travado.
#   PATCH_SIZE   (default: 10, mesmo default de 04f_train_implicit.sh)
#   OUT_FILE     (default: $WORK_DIR/diagnostics/implicit_pooling_<shell_b>_n<n_level>.csv)

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 14_diagnose_implicit_pooling.sh <work_dir> <checkpoint> <shell_b> <n_level>}"
CHECKPOINT="${2:?uso: sbatch 14_diagnose_implicit_pooling.sh <work_dir> <checkpoint> <shell_b> <n_level>}"
SHELL_B="${3:?uso: sbatch 14_diagnose_implicit_pooling.sh <work_dir> <checkpoint> <shell_b> <n_level>}"
N_LEVEL="${4:?uso: sbatch 14_diagnose_implicit_pooling.sh <work_dir> <checkpoint> <shell_b> <n_level>}"

source "./00_env_common.sh"

SCHEME_DIR="${SCHEME_DIR:-$WORK_DIR/subsampling}"
OUT_FILE="${OUT_FILE:-$WORK_DIR/diagnostics/implicit_pooling_${SHELL_B}_n${N_LEVEL}.csv}"

python scripts/14_diagnose_implicit_pooling.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$SCHEME_DIR" \
    --checkpoint "$CHECKPOINT" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --patch-size "${PATCH_SIZE:-10}" \
    --n-patches "${N_PATCHES:-8}" \
    --max-subjects "${MAX_SUBJECTS:-2}" \
    --out "$OUT_FILE"