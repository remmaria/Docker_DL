#!/bin/bash
#SBATCH --job-name=poc_multiproto
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/poc_multiproto.%A_%a.err
#SBATCH --output=logs/poc_multiproto.%A_%a.out
#
# Prova de conceito: testa quao confiavel e' o ground truth de CSD em
# b1000/64dir (o que a pipeline inteira usa como referencia em
# 11_peak_confusion_by_roi.py/12_visualize_fod_glyphs.py), comparando o
# MESMO sujeito adquirido em VARIOS protocolos/b-values/n-direcoes
# diferentes -- ver scripts/poc_multiprotocol_gt_reliability.py pro
# racional completo (3 comparacoes: repetibilidade em b fixo, sensibilidade
# a N em b fixo, sensibilidade ao b-value com N ~fixo).
#
# Job UNICO, sem sharding/array -- so uns poucos protocolos (nao centenas
# de sujeitos como o resto da pipeline), CSD em cada shell nao-zero de
# cada protocolo roda sequencial dentro do mesmo job.
#
# Uso (processa TODOS os protocolos encontrados em <data_root>; <out_csv>
# e' opcional, default work_dir/metrics/poc_multiprotocol_gt_reliability.csv
# relativo a raiz do repo -- passe um caminho proprio se nao tiver/quiser
# usar essa pasta work_dir):
#   sbatch slurm/poc_multiprotocol_gt_reliability.sh <data_root> <name_suffix> [out_csv]
#
# Ex.:
#   sbatch slurm/poc_multiprotocol_gt_reliability.sh /caminho/para/folder_main _geomcorr
#
# PROTOCOLS="SeqA1,SeqA5,SeqA6,SeqB" (variavel de ambiente) restringe a
# so esses protocolos (nomes das subpastas de <data_root>) em vez de
# processar todos -- util pra rodar so um subconjunto primeiro (ex.: so a
# repetibilidade em b1000) antes do lote inteiro. Ex.:
#   PROTOCOLS="SeqA1,SeqA5,SeqA6" sbatch slurm/poc_multiprotocol_gt_reliability.sh /caminho/para/folder_main _geomcorr
#
# ROI_TRACTS="FX,CGC,CGH,UF" (variavel de ambiente) TAMBEM calcula as
# metricas restritas a esses tratos JHU-ICBM, alem do cerebro inteiro
# (sempre calculado como roi='whole_mask') -- mesma convencao/mesmo nome
# de variavel de 11_peak_confusion_by_roi.sh/poc_csd_direction_count.sh.
#   ROI_TRACTS="FX,CGC,CGH,UF" sbatch slurm/poc_multiprotocol_gt_reliability.sh /caminho/para/folder_main _geomcorr
#
# SH_ORDER=<N> (variavel de ambiente, default vazio = ordem auto por
# shell, mesma convencao do baseline_sh/pipeline principal) -- forca a
# MESMA ordem SH em toda comparacao, pra isolar o efeito do b-value/N sem
# deixar a ordem tambem mudar junto (ver --sh-order no --help do script
# python).
#
# MASK_SUFFIX=<sufixo> (default "_mask3d.nii.gz") -- mesma convencao de
# --mask-suffix usado em todo o resto da pipeline (utils/masking.py).
#
# MAKE_GLYPHS=1 (2026-09-02, default 0) -- gera tambem uma figura com um
# painel de glifo FOD por (protocolo, shell), na melhor regiao de
# cruzamento de CADA um (nao refaz CSD, reaproveita o fit ja feito pras
# metricas -- ver docstring de scripts/poc_multiprotocol_gt_reliability.py).
# OUT_FIG (default: work_dir/figures/poc_multiprotocol_glyphs.png).
# SEARCH_RADIUS/PATCH_SIZE/SLICE_AXIS/MIN_MASK_FRAC/GLYPH_SCALE/
# GLYPH_N_ANGLES/NORMALIZE (defaults sensatos, mesma semantica de
# 12_visualize_fod_glyphs.sh -- NORMALIZE default aqui e' "per_voxel", nao
# "global", ja que nao ha' um painel "ground truth" comum entre
# protocolos).
#   MAKE_GLYPHS=1 sbatch slurm/poc_multiprotocol_gt_reliability.sh /caminho/para/folder_main _geomcorr
#
# MSMT=1 (2026-09-02, default 0) -- TAMBEM ajusta MSMT-CSD (multi-shell
# multi-tecido) usando TODAS as shells nao-zero de cada protocolo com >=2
# shells de uma vez (ex.: SeqB/SeqC/SeqD) -- ancora extra alem das
# comparacoes single-shell de sempre (ver docstring de
# scripts/poc_multiprotocol_gt_reliability.py, item 4). Mais lento e
# usa uma parte do DIPY (dipy.reconst.mcsd) nao testada em nenhum outro
# lugar desta pipeline -- se falhar com erro de assinatura de funcao,
# confira a versao do dipy instalada.
# MSMT_WM_FA_THR/MSMT_GM_FA_THR/MSMT_CSF_FA_THR/MSMT_GM_MD_THR/
# MSMT_CSF_MD_THR (defaults = defaults do proprio DIPY) -- limiares da
# segmentacao automatica de tecido, so' usados com MSMT=1; ajuste se a
# segmentacao automatica falhar/ficar visualmente ruim pra este sujeito.
#   MSMT=1 sbatch slurm/poc_multiprotocol_gt_reliability.sh /caminho/para/folder_main _geomcorr
#
# FIXED_PATCH_REFERENCE=<nome_protocolo> (2026-09-02, so' com MAKE_GLYPHS=1)
# -- em vez de cada painel buscar seu proprio melhor patch de cruzamento
# (default), localiza o patch UMA VEZ no protocolo indicado (ex. "SeqA1")
# e reaproveita o MESMO voxel fisico em TODOS os outros paineis -- so'
# funciona pra protocolos com o MESMO shape espacial do de referencia (o
# script pula, com aviso, qualquer painel cujo shape difira -- comparacao
# voxel-a-voxel sem registro previo nao e' valida). FIXED_PATCH_SHELL
# (opcional, float, so' com FIXED_PATCH_REFERENCE) escolhe qual shell do
# protocolo de referencia usar pra achar o patch, se ele tiver mais de
# uma shell nao-zero -- default usa a menor. Com patch fixo, considere
# tambem NORMALIZE=global (em vez do default per_voxel) -- com o MESMO
# voxel em todos os paineis, comparar magnitude relativa entre protocolos
# passa a fazer sentido (ainda sujeita a diferenca de contraste por
# b-value em si, ver --help do script python).
#   MAKE_GLYPHS=1 FIXED_PATCH_REFERENCE=SeqA1 NORMALIZE=global \
#     sbatch slurm/poc_multiprotocol_gt_reliability.sh /caminho/para/folder_main _geomcorr
set -euo pipefail
mkdir -p logs
DATA_ROOT="${1:?uso: sbatch poc_multiprotocol_gt_reliability.sh <data_root> <name_suffix> [out_csv]}"
NAME_SUFFIX="${2:?uso: sbatch poc_multiprotocol_gt_reliability.sh <data_root> <name_suffix> [out_csv]}"
OUT_CSV="${3:-work_dir/metrics/poc_multiprotocol_gt_reliability.csv}"

echo "Prova de conceito de confiabilidade do GT multi-protocolo: data_root=$DATA_ROOT name_suffix=$NAME_SUFFIX"
source "./00_env_common.sh"

PROTOCOLS_FLAG=()
if [[ -n "${PROTOCOLS:-}" ]]; then
    PROTOCOLS_FLAG=(--protocols "$PROTOCOLS")
    echo "PROTOCOLS=$PROTOCOLS -- restringindo a esse(s) protocolo(s)"
fi

ROI_FLAG=()
if [[ -n "${ROI_TRACTS:-}" ]]; then
    ROI_FLAG=(--roi-tracts "$ROI_TRACTS")
    echo "ROI_TRACTS=$ROI_TRACTS -- metricas tambem calculadas restritas a esses tratos"
fi

SH_ORDER_FLAG=()
if [[ -n "${SH_ORDER:-}" ]]; then
    SH_ORDER_FLAG=(--sh-order "$SH_ORDER")
    echo "SH_ORDER=$SH_ORDER -- forcando a MESMA ordem SH em todos os protocolos/shells"
fi

MASK_SUFFIX="${MASK_SUFFIX:-_mask3d.nii.gz}"
echo "MASK_SUFFIX=$MASK_SUFFIX (default _mask3d.nii.gz)"

MSMT_FLAGS=()
if [[ "${MSMT:-0}" == "1" ]]; then
    MSMT_FLAGS=(--msmt
                --msmt-wm-fa-thr "${MSMT_WM_FA_THR:-0.7}"
                --msmt-gm-fa-thr "${MSMT_GM_FA_THR:-0.3}"
                --msmt-csf-fa-thr "${MSMT_CSF_FA_THR:-0.15}"
                --msmt-gm-md-thr "${MSMT_GM_MD_THR:-0.001}"
                --msmt-csf-md-thr "${MSMT_CSF_MD_THR:-0.0032}")
    echo "MSMT=1 -- tambem ajustando MSMT-CSD (multi-shell) pros protocolos com >=2 shells"
fi

GLYPH_FLAGS=()
if [[ "${MAKE_GLYPHS:-0}" == "1" ]]; then
    OUT_FIG="${OUT_FIG:-work_dir/figures/poc_multiprotocol_glyphs.png}"
    GLYPH_FLAGS=(--make-glyphs --out-fig "$OUT_FIG"
                 --search-radius "${SEARCH_RADIUS:-15}"
                 --patch-size "${PATCH_SIZE:-4}"
                 --slice-axis "${SLICE_AXIS:-2}"
                 --min-mask-frac "${MIN_MASK_FRAC:-0.5}"
                 --glyph-scale "${GLYPH_SCALE:-0.45}"
                 --glyph-n-angles "${GLYPH_N_ANGLES:-72}"
                 --normalize "${NORMALIZE:-per_voxel}")
    echo "MAKE_GLYPHS=1 -- gerando tambem figura de glifos em $OUT_FIG"
    if [[ -n "${FIXED_PATCH_REFERENCE:-}" ]]; then
        GLYPH_FLAGS+=(--fixed-patch-reference "$FIXED_PATCH_REFERENCE")
        echo "FIXED_PATCH_REFERENCE=$FIXED_PATCH_REFERENCE -- reaproveitando o MESMO patch " \
             "fisico em todos os protocolos com o mesmo shape espacial"
        if [[ -n "${FIXED_PATCH_SHELL:-}" ]]; then
            GLYPH_FLAGS+=(--fixed-patch-shell "$FIXED_PATCH_SHELL")
            echo "FIXED_PATCH_SHELL=$FIXED_PATCH_SHELL -- usando essa shell do protocolo de " \
                 "referencia pra localizar o patch"
        fi
    fi
fi

python scripts/poc_multiprotocol_gt_reliability.py \
    --data-root "$DATA_ROOT" --name-suffix "$NAME_SUFFIX" \
    --mask-suffix "$MASK_SUFFIX" \
    "${PROTOCOLS_FLAG[@]}" "${ROI_FLAG[@]}" "${SH_ORDER_FLAG[@]}" "${MSMT_FLAGS[@]}" \
    "${GLYPH_FLAGS[@]}" \
    --out-csv "$OUT_CSV"