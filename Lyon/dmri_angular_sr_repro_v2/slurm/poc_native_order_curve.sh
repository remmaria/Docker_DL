#!/bin/bash
#SBATCH --job-name=poc_nativecurve
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/poc_nativecurve.%A_%a.err
#SBATCH --output=logs/poc_nativecurve.%A_%a.out
#
# Prova de conceito (2026-09-02, ver scripts/poc_native_order_curve.py pro
# racional completo): no MESMO sujeito real usado no evaluate (shell/n_dirs
# alta), mede como a estrutura de cruzamento do ground truth NATIVO (sem
# reconstrucao nenhuma -- CSD ajustado so nas direcoes de entrada reais,
# via os mesmos esquemas de subamostragem/*.npz ja usados pra treinar as
# redes) degrada conforme o numero de direcoes de entrada (n_level) cai,
# com a ordem SH mudando automaticamente junto (max_order_for_n_directions),
# igual a producao real faz. Serve de "piso nativo" pra comparar com
# reconstrucoes de rede no mesmo n_level (11_peak_confusion_by_roi.py).
#
# Job UNICO, sem sharding -- so 1 sujeito, uns poucos n_levels, CSD roda
# sequencial dentro do mesmo job.
#
# Uso (sujeito e' obrigatorio via SUBJECTS -- recomendado passar o mesmo
# sujeito padrao ja usado no evaluate; N_LEVELS default cobre os niveis
# canonicos do resto da pipeline):
#   SUBJECTS=<tag> sbatch slurm/poc_native_order_curve.sh <work_dir> <shell_b> [out_csv]
#
# Ex.:
#   SUBJECTS=20170417094841_802780_20170417094841_802780 \
#     sbatch slurm/poc_native_order_curve.sh work_dir 1000
#
# N_LEVELS="6 10 16 20 24 32 40 48 54" (variavel de ambiente, mesmos
# defaults canonicos do resto da pipeline) -- precisam ja existir no
# esquema de subamostragem do sujeito (TRIPLETS_DIR/<tag>_rrin_triplets.npz,
# gerado por scripts/02b_build_rrin_triplets.py).
#
# TRIPLETS_DIR (default <work_dir>/subsampling, mesma convencao do resto
# da pipeline) -- pasta com os esquemas de subamostragem ja gerados.
#
# ROI_TRACTS="FX,CGC,CGH,UF" (variavel de ambiente) -- mesma convencao do
# resto da pipeline, alem do 'whole_mask' sempre calculado.
#
# SH_ORDER=<N> (default vazio = ordem auto por n_level, RECOMENDADO pra
# esta curva -- ver docstring do script python) -- so passe se quiser
# forcar a MESMA ordem em todos os pontos (variante de controle).
#
# SPLIT (default "test", mesma convencao do resto da pipeline).
# MASK_SUFFIX (default "_mask3d.nii.gz").
#
# METRICAS TIER 1 (2026-09-02, ver docstring do script python): cada linha
# n_level agora TAMBEM ganha, alem de frac_crossing/energy_frac_high_order,
# precision/recall/mean_tp_angle_deg (casamento de picos contra a linha
# 'completa', requer so' o CSD ja ajustado, sem custo extra relevante) e
# FA_r2/FA_mae/FA_bias/FA_resid_std (ajusta DTI tambem, custo adicional mas
# pequeno perto do CSD). Sem flag nenhuma pra ligar -- sempre calculadas.
# PEAK_MATCH_THRESHOLD_DEG (default 25.0, mesmo default/mesma semantica de
# --peak-match-threshold-deg em scripts/11_peak_confusion_by_roi.py) ajusta
# o limiar angular do casamento de picos, se quiser mudar do default.
#
# N_JOBS=<N> (2026-09-02, EXPERIMENTAL, default vazio = --n-jobs 1 = mesmo
# comportamento sequencial de sempre -- ver --help do script python) -- se
# setado >1, tenta paralelizar o ajuste de CSD (peaks_from_model) de
# mascara INTEIRA (o gargalo dominante do script) usando essa quantidade de
# processos. Recomendado testar com um --n-levels pequeno (1-2 valores)
# antes de confiar num job grande -- nao foi possivel validar isso no
# ambiente de desenvolvimento (sem DIPY instalado). Combine com
# #SBATCH --cpus-per-task acima (ja pede 8 por default -- ajuste os dois
# juntos se quiser mudar): N_JOBS=8 usa toda a alocacao.
#   N_JOBS=8 SUBJECTS=<tag> sbatch slurm/poc_native_order_curve.sh work_dir 1000
#
# MAKE_GLYPHS=1 (default 0) -- gera tambem uma figura com um painel de
# glifo por n_level (mais um painel "completo" de ancora), TODOS no MESMO
# patch/voxels fisicos (escolhido uma vez no fit de ordem cheia).
# GLYPH_N_LEVELS (default = N_LEVELS inteiro -- passe um subconjunto pra
# nao gerar uma figura com paineis demais). OUT_FIG (default
# <work_dir>/figures/poc_native_order_curve_glyphs.png). CENTER_VOXEL
# opcional ("X,Y,Z") pra fixar o voxel manualmente em vez da busca
# automatica. SEARCH_RADIUS/PATCH_SIZE/SLICE_AXIS/MIN_MASK_FRAC/
# GLYPH_SCALE/GLYPH_N_ANGLES/NORMALIZE (defaults sensatos, mesma semantica
# de 12_visualize_fod_glyphs.sh -- NORMALIZE default aqui e' "global", ja
# que TODOS os paineis sao do MESMO sujeito/mesma escala fisica de sinal,
# diferente do poc_multiprotocol_gt_reliability.sh).
#   MAKE_GLYPHS=1 GLYPH_N_LEVELS="6 16 32 54" SUBJECTS=<tag> \
#     sbatch slurm/poc_native_order_curve.sh work_dir 1000
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch poc_native_order_curve.sh <work_dir> <shell_b> [out_csv]}"
SHELL_B="${2:?uso: sbatch poc_native_order_curve.sh <work_dir> <shell_b> [out_csv]}"
OUT_CSV="${3:-$WORK_DIR/metrics/poc_native_order_curve_shell${SHELL_B}.csv}"

if [[ -z "${SUBJECTS:-}" ]]; then
    echo "Erro: SUBJECTS obrigatorio (tag do sujeito, ex. o mesmo sujeito padrao ja usado no "
    echo "evaluate) -- ex.: SUBJECTS=<tag> sbatch slurm/poc_native_order_curve.sh $WORK_DIR $SHELL_B"
    exit 1
fi

echo "Curva nativa (sem reconstrucao) de estrutura de cruzamento vs. n_level: sujeito=$SUBJECTS shell_b=$SHELL_B"
source "./00_env_common.sh"

TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
N_LEVELS="${N_LEVELS:-6 10 16 20 24 32 40 48 54 60}"
echo "N_LEVELS=$N_LEVELS (default canonico da pipeline) -- precisam ja existir no esquema em $TRIPLETS_DIR"

SPLIT="${SPLIT:-test}"

ROI_FLAG=()
if [[ -n "${ROI_TRACTS:-}" ]]; then
    ROI_FLAG=(--roi-tracts "$ROI_TRACTS")
    echo "ROI_TRACTS=$ROI_TRACTS -- metricas tambem calculadas restritas a esses tratos"
fi

SH_ORDER_FLAG=()
if [[ -n "${SH_ORDER:-}" ]]; then
    SH_ORDER_FLAG=(--sh-order "$SH_ORDER")
    echo "SH_ORDER=$SH_ORDER -- forcando a MESMA ordem SH em todos os n_levels (variante de controle)"
else
    echo "SH_ORDER nao passado -- ordem auto por n_level (recomendado, mesma logica da producao)"
fi

MASK_SUFFIX="${MASK_SUFFIX:-_mask3d.nii.gz}"
echo "MASK_SUFFIX=$MASK_SUFFIX (default _mask3d.nii.gz)"

PEAK_MATCH_FLAG=()
if [[ -n "${PEAK_MATCH_THRESHOLD_DEG:-}" ]]; then
    PEAK_MATCH_FLAG=(--peak-match-threshold-deg "$PEAK_MATCH_THRESHOLD_DEG")
    echo "PEAK_MATCH_THRESHOLD_DEG=$PEAK_MATCH_THRESHOLD_DEG (default do script python: 25.0)"
fi

N_JOBS_FLAG=()
if [[ -n "${N_JOBS:-}" ]]; then
    N_JOBS_FLAG=(--n-jobs "$N_JOBS")
    echo "N_JOBS=$N_JOBS -- EXPERIMENTAL, paralelizando o ajuste de CSD de mascara inteira " \
         "(nao validado neste ambiente sem DIPY -- teste com --n-levels pequeno primeiro)"
fi

GLYPH_FLAGS=()
if [[ "${MAKE_GLYPHS:-0}" == "1" ]]; then
    OUT_FIG="${OUT_FIG:-$WORK_DIR/figures/poc_native_order_curve_glyphs.png}"
    GLYPH_LEVELS="${GLYPH_N_LEVELS:-$N_LEVELS}"
    GLYPH_FLAGS=(--make-glyphs --out-fig "$OUT_FIG" --glyph-n-levels $GLYPH_LEVELS
                 --search-radius "${SEARCH_RADIUS:-15}"
                 --patch-size "${PATCH_SIZE:-4}"
                 --slice-axis "${SLICE_AXIS:-2}"
                 --min-mask-frac "${MIN_MASK_FRAC:-0.5}"
                 --glyph-scale "${GLYPH_SCALE:-0.45}"
                 --glyph-n-angles "${GLYPH_N_ANGLES:-72}"
                 --normalize "${NORMALIZE:-global}")
    if [[ -n "${CENTER_VOXEL:-}" ]]; then
        GLYPH_FLAGS+=(--center-voxel "$CENTER_VOXEL")
        echo "CENTER_VOXEL=$CENTER_VOXEL -- fixando o voxel manualmente (pula a busca automatica)"
    fi
    echo "MAKE_GLYPHS=1 -- gerando tambem figura de glifos em $OUT_FIG (n_levels: $GLYPH_LEVELS)"
fi

python scripts/poc_native_order_curve.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR" \
    --shell-b "$SHELL_B" --n-levels $N_LEVELS \
    --split "$SPLIT" --subjects "$SUBJECTS" \
    --mask-suffix "$MASK_SUFFIX" \
    "${ROI_FLAG[@]}" "${SH_ORDER_FLAG[@]}" "${PEAK_MATCH_FLAG[@]}" "${N_JOBS_FLAG[@]}" \
    "${GLYPH_FLAGS[@]}" \
    --out-csv "$OUT_CSV"