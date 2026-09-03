#!/bin/bash
#SBATCH --job-name=dmri_tractography
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-12:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/tractography.%A_%a.err
#SBATCH --output=logs/tractography.%A_%a.out
#
# Etapa 8 (downstream, opcional, a mais forte pra tese -- ver addendum
# secao 24): tratografia real via MRtrix3 (dwi2response/dwi2fod/tckgen)
# comparando ground truth, o piso nativo (--subsampled-only, sem
# reconstrucao nenhuma) e cada metodo de reconstrucao, por trato
# (--roi-tracts): contagem de streamlines que atravessam o trato,
# comprimento medio, e Dice de densidade contra o ground truth. Requer
# MRtrix3 no PATH -- carregado automaticamente abaixo via
# `module load mrtrix3/<versao>` (MRTRIX_MODULE, default "mrtrix3/3.0.5",
# a versao marcada "(D)"/default no seu cluster -- confirmado pela usuaria
# via `module avail`: "mrtrix3/3.0.0" e "mrtrix3/3.0.5 (D)" disponiveis).
# Carregado DEPOIS de "./00_env_common.sh" de proposito -- esse script faz
# "module purge" no inicio, que apagaria qualquer modulo carregado antes.
#
# EXTRA_METHOD="nome=caminho" (repetivel via virgula) -- mesma convencao de
# slurm/11_peak_confusion_by_roi.py, ex.:
#   EXTRA_METHOD="rrin_star=$WORK_DIR/rrin_star_recon,naive_blend=$WORK_DIR/naive_blend_recon"
#
# SUBSAMPLED_ONLY=1 -- inclui o "piso nativo" (so as n_level direcoes reais
# de entrada, sem reconstrucao -- requer TRIPLETS_DIR, default
# "$WORK_DIR/subsampling").
#
# ROI_TRACTS="FX,CGC,CGH,UF" (default vazio = so a linha whole_mask, mesma
# convencao de sempre) -- filtra o track whole-brain (tckedit -include) por
# cada trato e reporta contagem de streamlines/comprimento medio/Dice POR
# TRATO, alem da linha whole_mask.
#
# ROI_PAIR="NOME=ROI_A+ROI_B,NOME2=ROI_C+ROI_D" (ADITIVO, ver addendum secao
# 25.2) -- exige que a streamline atravesse AMBAS as ROIs listadas, nao
# qualquer uma isolada (AND via dois `tckedit -include`); filtra fibra de
# estruturas vizinhas que so' "raspam" a borda de uma unica ROI sem
# pertencer genuinamente aquele circuito (ex.: a stria terminalis passa
# perto do forice e pode contaminar a contagem de "FX" sozinho). ROI_A/
# ROI_B usam os mesmos nomes de ROI_TRACTS (tratos JHU ou mascaras de
# segmentacao por sujeito, ex. Hipp_L/Hipp_R) e nao precisam tambem estar
# em ROI_TRACTS (sao carregadas automaticamente so' como insumo do par).
# Ex.:
#   ROI_PAIR="FX_Hipp_R=FX+Hipp_R,FX_Hipp_L=FX+Hipp_L"
#
# ROI_DILATE="NOME=N,NOME2=N2" (ADITIVO, ver addendum secao 26.1) --
# dilata a mascara de ROI "NOME" em N voxels antes de usa-la em
# tckedit -include. Recomendado pra mascaras de SEGMENTACAO tipo
# Hipp_L/Hipp_R usadas em ROI_PAIR (sao substancia cinzenta -- a fimbria
# do fornix e' substancia branca fina colada na borda, um erro de
# registro de 1-2 voxels descarta conexao genuina sem dilatar). NAO usar
# nos tratos LATERALIZADOS do atlas JHU (CGC_L/CGC_R/CGH_L/CGH_R/UF_L/
# UF_R) -- dilatar em direcao a linha media reintroduz vazamento
# comissural. Ex.:
#   ROI_DILATE="Hipp_R=2,Hipp_L=2"
#
# ALGORITHM=SD_STREAM (default -- deterministico, segue o pico dominante da
# FOD) ou ALGORITHM=iFOD2 (probabilistico -- o default do proprio MRtrix se
# esta variavel nao existisse, ver docstring do script python).
#
# N_STREAMLINES=200000 (default) -- numero de streamlines geradas no
# tckgen whole-brain (por metodo/sujeito, antes de filtrar por trato).
#
# GPU NAO ajuda aqui (pergunta da usuaria, addendum secao 24.2) -- MRtrix3
# (dwi2response/dwi2fod/tckgen/tckedit/tckmap) e' CPU-only, sem nenhum
# passo acelerado por GPU. A alavanca real e' multithread de CPU:
# scripts/08_downstream_tractography.py ja usa $SLURM_CPUS_PER_TASK como
# `-nthreads` por default (sem precisar de nada aqui) -- NTHREADS=<N>
# (variavel de ambiente opcional) sobrescreve isso manualmente, se quiser
# testar um valor diferente do que o job alocou.
#
# SUBJECTS="tag1,tag2" (variavel de ambiente) -- roda so nesse(s) sujeito(s)
# especifico(s) em vez do --split inteiro, mesma convencao de
# slurm/11_peak_confusion_by_roi.sh/poc_csd_direction_count.sh. Util pra
# testar rapido num sujeito so antes de rodar o dataset inteiro
# (tratografia e' cara). Ex.:
#   SUBJECTS="20170417094841_802780_20170417094841_802780" SUBSAMPLED_ONLY=1 \
#   ROI_TRACTS="FX,CGC,CGH,UF" \
#     sbatch slurm/08_downstream_tractography.sh <work_dir> 1000 16
#
# Uso (comparacao completa, 1 combo shell/n_level):
#   EXTRA_METHOD="rrin_star=$WORK_DIR/rrin_star_recon" SUBSAMPLED_ONLY=1 \
#   ROI_TRACTS="FX,CGC,CGH,UF" \
#     sbatch slurm/08_downstream_tractography.sh <work_dir> 1000 16
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 08_downstream_tractography.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 08_downstream_tractography.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 08_downstream_tractography.sh <work_dir> <shell_b> <n_level>}"

source "./00_env_common.sh"

MRTRIX_MODULE="${MRTRIX_MODULE:-mrtrix3/3.0.5}"
module load "$MRTRIX_MODULE"
echo "MRTRIX_MODULE=$MRTRIX_MODULE -- tckgen: $(which tckgen)"
tckgen -version | head -1

BASELINE_FLAG=(--baseline-dir "$WORK_DIR/baseline_recon")
RCAE_FLAG=()
if [[ -d "$WORK_DIR/rcae_recon" ]]; then
    RCAE_FLAG=(--rcae-dir "$WORK_DIR/rcae_recon")
fi

EXTRA_METHOD_FLAGS=()
if [[ -n "${EXTRA_METHOD:-}" ]]; then
    IFS=',' read -ra _EXTRA_SPECS <<< "$EXTRA_METHOD"
    for spec in "${_EXTRA_SPECS[@]}"; do
        EXTRA_METHOD_FLAGS+=(--extra-method "$spec")
    done
    echo "EXTRA_METHOD=$EXTRA_METHOD"
fi

SUBSAMPLED_ONLY_FLAG=()
if [[ "${SUBSAMPLED_ONLY:-0}" == "1" ]]; then
    TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
    SUBSAMPLED_ONLY_FLAG=(--subsampled-only --triplets-dir "$TRIPLETS_DIR")
    echo "SUBSAMPLED_ONLY=1 -- piso nativo lendo target_idx de $TRIPLETS_DIR"
fi

ROI_FLAG=()
if [[ -n "${ROI_TRACTS:-}" ]]; then
    ROI_FLAG=(--roi-tracts "$ROI_TRACTS")
    echo "ROI_TRACTS=$ROI_TRACTS"
else
    echo "ROI_TRACTS nao passado -- so a linha whole_mask sera reportada"
fi

ROI_PAIR_FLAGS=()
if [[ -n "${ROI_PAIR:-}" ]]; then
    IFS=',' read -ra _ROI_PAIR_SPECS <<< "$ROI_PAIR"
    for spec in "${_ROI_PAIR_SPECS[@]}"; do
        ROI_PAIR_FLAGS+=(--roi-pair "$spec")
    done
    echo "ROI_PAIR=$ROI_PAIR"
fi

ROI_DILATE_FLAGS=()
if [[ -n "${ROI_DILATE:-}" ]]; then
    IFS=',' read -ra _ROI_DILATE_SPECS <<< "$ROI_DILATE"
    for spec in "${_ROI_DILATE_SPECS[@]}"; do
        ROI_DILATE_FLAGS+=(--roi-dilate "$spec")
    done
    echo "ROI_DILATE=$ROI_DILATE"
fi

ALGORITHM="${ALGORITHM:-SD_STREAM}"
echo "ALGORITHM=$ALGORITHM (default SD_STREAM = deterministico)"

N_STREAMLINES="${N_STREAMLINES:-200000}"

NTHREADS_FLAG=()
if [[ -n "${NTHREADS:-}" ]]; then
    NTHREADS_FLAG=(--nthreads "$NTHREADS")
    echo "NTHREADS=$NTHREADS -- sobrescrevendo o default (\$SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-nao definido})"
fi

SUBJECTS_FLAG=()
if [[ -n "${SUBJECTS:-}" ]]; then
    SUBJECTS_FLAG=(--subjects "$SUBJECTS")
    echo "SUBJECTS=$SUBJECTS -- restringindo a esse(s) sujeito(s)"
fi

python scripts/08_downstream_tractography.py \
    --manifest "$WORK_DIR/manifest.csv" \
    "${BASELINE_FLAG[@]}" "${RCAE_FLAG[@]}" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --algorithm "$ALGORITHM" --n-streamlines "$N_STREAMLINES" \
    --out-dir "$WORK_DIR/tractography" \
    "${EXTRA_METHOD_FLAGS[@]}" "${SUBSAMPLED_ONLY_FLAG[@]}" "${ROI_FLAG[@]}" "${ROI_PAIR_FLAGS[@]}" \
    "${ROI_DILATE_FLAGS[@]}" "${NTHREADS_FLAG[@]}" "${SUBJECTS_FLAG[@]}"