#!/bin/bash
#SBATCH --job-name=dmri_rrin_triplets
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/rrin_trip.%J.err
#SBATCH --output=logs/rrin_trip.%J.out
#
# Etapa 2b: constroi as trincas (par de entrada + alvo) para a linha
# RRIN/VFI-por-triplets, a partir do esquema de subamostragem ja gerado
# pela etapa 2 (scripts/02_subsample_directions.py). Ver
# scripts/02b_build_rrin_triplets.py e protocolo secao 10.1.
#
# Uso:
#   sbatch slurm/02b_build_rrin_triplets.sh <work_dir>
#   MAX_RESIDUAL_DEG=8.0 sbatch slurm/02b_build_rrin_triplets.sh <work_dir>
#
# ENSEMBLE_M=<M> (variavel de ambiente, default 0 = desligado, ADITIVO -- ver
# --ensemble-m em scripts/02b_build_rrin_triplets.py e protocolo secao 14.5
# item 1/addendum 2026-08-27, "ensemble em estrela"): quando > 0, TAMBEM
# grava, para cada alvo, um feixe de ate M pares de entrada DIVERSOS (nao so
# o melhor par -- ver utils/gradients.py:find_star_ensemble_batch), usado
# pelo treino/reconstrucao do ensemble em estrela
# (slurm/04e_train_rrin_star.sh / slurm/05f_reconstruct_rrin_star.sh). Nao
# muda nada do que ja existe no npz (par unico continua identico) -- so
# ACRESCENTA campos novos, entao e seguro rodar de novo com ENSEMBLE_M>0
# em cima de um work_dir ja processado sem essa flag, sem afetar RRIN3D/
# AMT3D/HFD3D (single-pair) ja treinados.
#   ENSEMBLE_M=3 sbatch slurm/02b_build_rrin_triplets.sh <work_dir>
#
# ENSEMBLE_MAX_RESIDUAL_DEG=<graus> (variavel de ambiente, default vazio =
# usa o MESMO valor de MAX_RESIDUAL_DEG, comportamento de sempre -- ver
# --ensemble-max-residual-deg em scripts/02b_build_rrin_triplets.py e
# addendum 2026-08-27 secao 14.1): teto de residuo SEPARADO, so pro pool do
# feixe (ENSEMBLE_M) -- o par unico continua sempre controlado só por
# MAX_RESIDUAL_DEG, mesmo quando este estiver setado.
#
# ENSEMBLE_MAX_GAP_DEG=<graus> (variavel de ambiente, default vazio = sem
# teto de gap, comportamento de sempre -- ver --ensemble-max-gap-deg em
# scripts/02b_build_rrin_triplets.py e addendum 2026-08-27 secao 20.6):
# teto de gap_deg (separacao angular a/b) para PREFERIR, quando ha
# candidatos suficientes, pares com separacao pequena entre os m-1 pares
# ALEM da semente do feixe (a semente ja e' sempre o de menor gap_deg do
# pool, com ou sem este teto). Motivacao: a diversidade geometrica pura
# (FPS por normais) pode escolher varios pares com gap_deg grande so por
# serem dispersos entre si -- um diagnostico de pesos de fusao por voxel
# (addendum secao 20.6) mostrou que, quando NENHUM candidato do feixe tem
# gap pequeno, a fusao aprendida acaba borrando estrutura angular fina numa
# media quase-uniforme entre candidatos mediocres, em vez de confiar num
# par 'facil'. Nunca reduz o numero de pares reais do feixe -- so influencia
# QUAIS sao escolhidos quando ha opcao de sobra; sem candidatos suficientes
# dentro do teto, cai de volta no comportamento sem teto so naquele alvo.
#   ENSEMBLE_MAX_GAP_DEG=25 sbatch slurm/02b_build_rrin_triplets.sh <work_dir>
#
# OUT_DIR=<pasta> (variavel de ambiente, default "$WORK_DIR/subsampling" --
# ATENCAO: esse e' o default de SEMPRE, o mesmo usado por qualquer treino
# ja rodando que le desse work_dir): escreve os <tag>_rrin_triplets.npz
# numa pasta DIFERENTE em vez de sobrescrever a de sempre. Use isso pra
# testar um teto de ensemble diferente (ou qualquer outro parametro do
# 02b) SEM mexer no arquivo que um treino ja em andamento esta lendo --
# --scheme-dir continua apontando pra "$WORK_DIR/subsampling" (so leitura,
# nao e alterado), so o --out-dir muda. Depois, aponte
# 02c_diagnose_rrin_triplets.sh / o treino de teste pra essa mesma pasta
# nova (ver TRIPLETS_DIR em slurm/02c_diagnose_rrin_triplets.sh).
#   OUT_DIR=$WORK_DIR/subsampling_ens_test ENSEMBLE_M=3 \
#     ENSEMBLE_MAX_RESIDUAL_DEG=10 sbatch slurm/02b_build_rrin_triplets.sh <work_dir>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02b_build_rrin_triplets.sh <work_dir>}"

source "./00_env_common.sh"

MAX_RESIDUAL_DEG="${MAX_RESIDUAL_DEG:-5.0}"
echo "MAX_RESIDUAL_DEG=$MAX_RESIDUAL_DEG"

OUT_DIR="${OUT_DIR:-$WORK_DIR/subsampling}"
if [[ "$OUT_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "OUT_DIR=$OUT_DIR -- escrevendo em pasta SEPARADA (nao mexe no subsampling/ de sempre)"
fi

ENSEMBLE_M="${ENSEMBLE_M:-0}"
ENSEMBLE_FLAG=()
if [[ "$ENSEMBLE_M" != "0" ]]; then
    ENSEMBLE_FLAG=(--ensemble-m "$ENSEMBLE_M")
    echo "ENSEMBLE_M=$ENSEMBLE_M -- gravando tambem o feixe 'ensemble em estrela' (campos __ens_*)"
fi

ENSEMBLE_MAX_RESIDUAL_DEG="${ENSEMBLE_MAX_RESIDUAL_DEG:-}"
ENS_MAX_RESIDUAL_FLAG=()
if [[ -n "$ENSEMBLE_MAX_RESIDUAL_DEG" ]]; then
    ENS_MAX_RESIDUAL_FLAG=(--ensemble-max-residual-deg "$ENSEMBLE_MAX_RESIDUAL_DEG")
    echo "ENSEMBLE_MAX_RESIDUAL_DEG=$ENSEMBLE_MAX_RESIDUAL_DEG -- teto separado so pro pool do feixe"
fi

ENSEMBLE_MAX_GAP_DEG="${ENSEMBLE_MAX_GAP_DEG:-}"
ENS_MAX_GAP_FLAG=()
if [[ -n "$ENSEMBLE_MAX_GAP_DEG" ]]; then
    ENS_MAX_GAP_FLAG=(--ensemble-max-gap-deg "$ENSEMBLE_MAX_GAP_DEG")
    echo "ENSEMBLE_MAX_GAP_DEG=$ENSEMBLE_MAX_GAP_DEG -- preferindo pares de gap pequeno no feixe, quando possivel"
fi

python scripts/02b_build_rrin_triplets.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --out-dir "$OUT_DIR" \
    --max-residual-deg "$MAX_RESIDUAL_DEG" \
    "${ENSEMBLE_FLAG[@]}" \
    "${ENS_MAX_RESIDUAL_FLAG[@]}" \
    "${ENS_MAX_GAP_FLAG[@]}"