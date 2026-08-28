#!/bin/bash
#SBATCH --job-name=dmri_viz_triplet
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-00:15:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/viz_triplet.%J.err
#SBATCH --output=logs/viz_triplet.%J.out
#
# Diagnostico visual (scripts/07_visualize_triplet.py, ver addendum
# 2026-08-27 secao 11) -- plota a esfera de bvecs (com espelho antipodal,
# destacando o conjunto de entrada de um n_level e uma trinca-exemplo com
# a geodesica a->b) e as DWIs reais dessa trinca (vol_a, vol_b, alvo real,
# blend ingenuo, diferenca absoluta). NAO precisa de GPU/torch (so
# numpy/matplotlib/nibabel) -- roda nesta particao so por conveniencia de
# acesso ao filesystem dos dados, e' bem rapido/leve (--time 15min, sem
# --gres, mesma logica de slurm/02_baseline_sh.sh).
#
# Uso (canonico -- escolhe sujeito e trinca automaticamente):
#   sbatch slurm/07_visualize_triplet.sh <work_dir> <shell_b> <n_level>
#
# SUBJECT=<tag> (variavel de ambiente, opcional) -- fixa o sujeito (default:
# primeiro do split 'train' com dados disponiveis para esse shell/n_level).
#   SUBJECT=sub01 sbatch slurm/07_visualize_triplet.sh <work_dir> 1000 16
#
# EXAMPLE={typical,best,worst} (variavel de ambiente, opcional, default
# 'typical') -- qual trinca escolher automaticamente (ver docstring de
# scripts/07_visualize_triplet.py). Ignorado se TRIPLET_INDEX for passado.
#   EXAMPLE=worst sbatch slurm/07_visualize_triplet.sh <work_dir> 1000 16
#
# TRIPLET_INDEX=<i> (variavel de ambiente, opcional) -- sobrepoe EXAMPLE,
# usa o indice literal dentro do array de trincas desse (shell_b,n_level).
#   TRIPLET_INDEX=3 sbatch slurm/07_visualize_triplet.sh <work_dir> 1000 16
#
# OUT_NAME=<nome> (variavel de ambiente, opcional) -- nome-base do PNG de
# saida (sem sufixo _esfera/_dwis, sem diretorio -- sempre gravado em
# $WORK_DIR/figures/). Default: triplet_shell<shell_b>_n<n_level>[_<sujeito>].
#   OUT_NAME=meu_exemplo sbatch slurm/07_visualize_triplet.sh <work_dir> 1000 16
#
# SLICE_AXIS=<0|1|2> / SLICE_INDEX=<i> / NO_CROP=1 (variaveis de ambiente,
# opcionais) -- ver --slice-axis/--slice-index/--no-crop no proprio script.
#
# ENSEMBLE_M=<M> / NO_ENSEMBLE=1 (variaveis de ambiente, opcionais, ver
# protocolo secao 14.5 item 1/addendum 2026-08-27, "ensemble em estrela") --
# se o npz de trincas tiver sido gerado com --ensemble-m (ver
# slurm/02b_build_rrin_triplets.py ENSEMBLE_M), a figura da esfera TAMBEM
# desenha os pares diversos do feixe daquela trinca. ENSEMBLE_M limita
# quantos pares desenhar (default: todos os gravados); NO_ENSEMBLE=1
# desliga esse desenho mesmo que o npz tenha os campos. Sem efeito se o npz
# nao tiver --ensemble-m (par-unico apenas, comportamento de sempre).
#   ENSEMBLE_M=3 sbatch slurm/07_visualize_triplet.sh <work_dir> 1000 16
#
# ARGUMENTOS POSICIONAIS (nao variaveis de ambiente) para work_dir/shell_b/
# n_level pelo mesmo motivo documentado em slurm/02_baseline_sh.sh: em
# alguns clusters SLURM "VAR=valor sbatch ..." nao propaga a variavel pro
# ambiente do job de verdade -- argumento posicional nunca tem esse problema.

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 07_visualize_triplet.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 07_visualize_triplet.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 07_visualize_triplet.sh <work_dir> <shell_b> <n_level>}"

source "./00_env_common.sh"

FIG_DIR="$WORK_DIR/figures"
mkdir -p "$FIG_DIR"

SUBJECT_FLAG=()
if [[ -n "${SUBJECT:-}" ]]; then
    SUBJECT_FLAG=(--subject "$SUBJECT")
    echo "SUBJECT=$SUBJECT -- fixando o sujeito"
fi

EXAMPLE_FLAG=()
if [[ -n "${EXAMPLE:-}" ]]; then
    EXAMPLE_FLAG=(--example "$EXAMPLE")
    echo "EXAMPLE=$EXAMPLE -- criterio de selecao da trinca-exemplo"
fi

TRIPLET_INDEX_FLAG=()
if [[ -n "${TRIPLET_INDEX:-}" ]]; then
    TRIPLET_INDEX_FLAG=(--triplet-index "$TRIPLET_INDEX")
    echo "TRIPLET_INDEX=$TRIPLET_INDEX -- sobrepoe EXAMPLE, usando este indice literal"
fi

SLICE_AXIS_FLAG=()
if [[ -n "${SLICE_AXIS:-}" ]]; then
    SLICE_AXIS_FLAG=(--slice-axis "$SLICE_AXIS")
fi
SLICE_INDEX_FLAG=()
if [[ -n "${SLICE_INDEX:-}" ]]; then
    SLICE_INDEX_FLAG=(--slice-index "$SLICE_INDEX")
fi
NO_CROP_FLAG=()
if [[ "${NO_CROP:-0}" == "1" ]]; then
    NO_CROP_FLAG=(--no-crop)
fi

ENSEMBLE_M_FLAG=()
if [[ -n "${ENSEMBLE_M:-}" ]]; then
    ENSEMBLE_M_FLAG=(--ensemble-m "$ENSEMBLE_M")
    echo "ENSEMBLE_M=$ENSEMBLE_M -- limitando a esse numero de pares do feixe na figura"
fi
NO_ENSEMBLE_FLAG=()
if [[ "${NO_ENSEMBLE:-0}" == "1" ]]; then
    NO_ENSEMBLE_FLAG=(--no-ensemble)
    echo "NO_ENSEMBLE=1 -- desligando o desenho do feixe 'ensemble em estrela'"
fi

if [[ -n "${OUT_NAME:-}" ]]; then
    OUT_BASENAME="$OUT_NAME"
else
    OUT_BASENAME="triplet_shell${SHELL_B%.*}_n${N_LEVEL}"
    if [[ -n "${SUBJECT:-}" ]]; then
        OUT_BASENAME="${OUT_BASENAME}_${SUBJECT}"
    fi
fi
OUT_PATH="$FIG_DIR/${OUT_BASENAME}.png"

echo "Gerando diagnostico visual para shell_b=$SHELL_B, n_level=$N_LEVEL"
echo "Saida: ${OUT_PATH%.png}_esfera.png e ${OUT_PATH%.png}_dwis.png"

python scripts/07_visualize_triplet.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$WORK_DIR/subsampling" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --out "$OUT_PATH" \
    "${SUBJECT_FLAG[@]}" "${EXAMPLE_FLAG[@]}" "${TRIPLET_INDEX_FLAG[@]}" \
    "${SLICE_AXIS_FLAG[@]}" "${SLICE_INDEX_FLAG[@]}" "${NO_CROP_FLAG[@]}" \
    "${ENSEMBLE_M_FLAG[@]}" "${NO_ENSEMBLE_FLAG[@]}"