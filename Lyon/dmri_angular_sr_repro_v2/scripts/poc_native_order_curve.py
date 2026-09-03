#!/usr/bin/env python3
"""
Prova de conceito (2026-09-02, a pedido da usuaria -- "se pegarmos o
sujeito que sempre uso pra fazer o evaluate... da pra reduzir as direcoes
e ir vendo como as ordens e os glifos vao sendo afetados?"): mede, num
UNICO sujeito real de shell_b/n_dirs alta (o mesmo sujeito padrao usado em
06_evaluate_reconstruction.py/11_peak_confusion_by_roi.py/etc.), como a
estrutura de cruzamento do "ground truth nativo" (SEM reconstrucao
nenhuma) degrada conforme o numero de direcoes de entrada (n_level) cai --
usando a MESMA logica ja implementada do modo --subsampled-only de
scripts/11_peak_confusion_by_roi.py/scripts/12_visualize_fod_glyphs.py:
CSD ajustado so nas direcoes REAIS de entrada (exclui as direcoes-alvo do
esquema de subamostragem via exclude_idx), com sh_order automatico
max_order_for_n_directions(n_level) -- a MESMA regra que a producao usa
pra decidir a ordem do fit, sem nenhum valor forcado.

DIFERENCA em relacao a scripts/poc_multiprotocol_gt_reliability.py: aquele
POC compara PROTOCOLOS DE AQUISICAO DIFERENTES (b-value/N/n_b0 variando
JUNTOS, confundidos entre si, ver addendum secao 21) -- aqui e' o MESMO
sujeito, MESMA aquisicao b1000/64dir (ou o que for --shell-b), variando SO
o numero de direcoes de ENTRADA usadas no fit (via os mesmos esquemas de
subamostragem/*.npz ja gerados por scripts/02b_build_rrin_triplets.py pra
treinar as redes) -- sem confundir b-value/n_b0 nenhum, isola
especificamente "quanto da estrutura de cruzamento eu credito por ter mais
ou menos direcoes medidas, na MESMA aquisicao que a rede tambem ve".

PARA QUE SERVE NA PRATICA: depois de rodar isso, quando voce reconstruir
com uma rede (RRIN-star, RCAE, etc.) num n_level especifico e' calcular
frac_crossing/energy_frac_high_order/mean_n_peaks da reconstrucao (colunas
ja existentes em scripts/11_peak_confusion_by_roi.py) e comparar com o
PONTO DESTA CURVA no mesmo n_level: se a reconstrucao fica bem ACIMA do
que aquele N sozinho sustenta nativamente, e' evidencia de que a rede
esta genuinamente recuperando estrutura angular que nao estava naquelas
direcoes; se fica perto ou abaixo, a rede nao esta indo muito alem do que
a propria contagem de direcoes ja garantiria de qualquer jeito.

Uso (curva numerica, sem glifos):
    python scripts/poc_native_order_curve.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 \
        --n-levels 6 10 16 20 24 32 40 48 54 \
        --subjects 20170417094841_802780_20170417094841_802780 \
        --roi-tracts FX,CGC,CGH,UF \
        --out-csv work_dir/metrics/poc_native_order_curve_shell1000.csv

--MAKE-GLYPHS (ADITIVO): gera tambem UMA figura com um painel de glifo por
n_level (mesmo patch de cruzamento, MESMOS voxels fisicos em todos os
paineis -- escolhido UMA vez a partir do fit de ordem cheia, diferente do
--make-glyphs de poc_multiprotocol_gt_reliability.py, que escolhe um patch
por protocolo pois os protocolos podem nao estar no mesmo grid; aqui e' o
MESMO volume, entao faz sentido e e' mais informativo usar sempre o mesmo
voxel), permitindo ver visualmente a forma do FOD se degradando conforme
N cai. Painel extra "completo (nXX)" (sem exclusao nenhuma) sempre
incluido como ancora à esquerda.

Uso (com glifos, --glyph-n-levels default = --n-levels inteiro):
    python scripts/poc_native_order_curve.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-levels 6 10 16 20 24 32 40 48 54 \
        --subjects 20170417094841_802780_20170417094841_802780 \
        --make-glyphs --glyph-n-levels 6 16 32 54 \
        --out-fig work_dir/figures/poc_native_order_curve_glyphs.png \
        --out-csv work_dir/metrics/poc_native_order_curve_shell1000.csv

METRICAS TIER 1 na propria curva nativa (2026-09-02, a pedido da usuaria
depois da revisao "quais metricas realmente importam" -- ver addendum):
alem de frac_crossing/energy_frac_high_order (que a discussao da secao 22.1
mostrou nao serem totalmente confiaveis pra RANKING entre metodos avaliados
em ordem SH cheia/descasada -- aqui, como e' sempre a MESMA ordem
auto-consistente por n_level, o problema nao se aplica do mesmo jeito, mas
mesmo assim as metricas abaixo sao as mais diretas), cada linha n_level
(nao a linha is_full, que e' a propria referencia) tambem ganha:

- precision/recall/mean_tp_angle_deg (+ TP/FP/FN/TN_voxels): casamento de
  picos (mesma logica de match_peaks_voxel/confusion_for_roi de
  scripts/11_peak_confusion_by_roi.py, duplicada aqui) entre os picos desta
  linha (fit com exclude_idx=target_idx, so' as n_level direcoes reais) e os
  picos da linha 'completa' (is_full=True), tratada como groun truth pra
  este proposito -- mesmo raciocinio de --subsampled-only em
  11_peak_confusion_by_roi.py, so' que aqui e' sempre contra o MESMO sujeito/
  aquisicao, nunca contra reconstrucao nenhuma.
- FA_r2 (+ FA_mae/FA_bias/FA_resid_std): ajusta DTI (mesma logica de
  fit_dti/_roi_r2 de scripts/07_downstream_dti_noddi.py, duplicada aqui) com
  exclude_idx=target_idx (so' as n_level direcoes reais, SEM reconstrucao
  nenhuma) e compara o mapa de FA resultante, por ROI, contra o FA ajustado
  na linha 'completa' -- responde "so' com N direcoes reais (sem nenhuma
  rede), o quanto do FA verdadeiro (heterogeneidade espacial real) eu ja
  recupero?", o mesmo piso conceitual que --subsampled-only mede em
  07_downstream_dti_noddi.py, so' que sem depender de nenhuma reconstrucao
  rodada previamente.

Ambas as familias ficam NaN na linha is_full=True (comparar a referencia
contra ela mesma seria trivial/precision=recall=1/FA_r2=1, sem informacao) e
ficam NaN tambem em qualquer n_level cujo fit de CSD ou de DTI falhe (ver
failed_rows/try-except em main()) -- uma familia pode falhar
independentemente da outra (ex.: DTI converge mas CSD nao, ou vice-versa).

FIX 2026-09-02 (--make-glyphs, achado a partir de inspecao visual da
usuaria -- "o gt e o n54 sao muito diferentes... os glifos estao sendo
gerados da forma correta?"): os paineis de glifo por n_level reajustam CSD
num SUB-VOLUME PEQUENO recortado ao redor do patch (--search-radius), por
velocidade -- MAS `auto_response_ssst` (a estimativa do response function
de fibra unica usado pelo CSD) rodava DENTRO desse recorte pequeno, em vez
do cerebro inteiro como o painel 'completo' usa. Como o recorte fica
propositalmente numa regiao de CRUZAMENTO (o patch foi escolhido por isso),
ele pode nao conter nenhum bom voxel de fibra unica, entao o response
function estimado ali podia ser bem diferente/pior do que o do painel
'completo' -- inflando artificialmente a diferenca visual entre paineis por
causa de um KERNEL DE DECONVOLUCAO diferente, nao pelo efeito genuino de
n_level caindo que a figura pretende mostrar. Corrigido: `fit_peaks` agora
aceita um `response` ja calculado (skip de auto_response_ssst) e SEMPRE
devolve o response realmente usado; main() estima o response UMA VEZ no
fit 'completo' de corpo inteiro e reusa o MESMO em todos os paineis de
glifo por n_level. Isso NAO muda os numeros do CSV (a curva numerica sempre
ajustou CSD no mask/data de CORPO INTEIRO em cada n_level, nunca no
sub-volume recortado -- so' os paineis de glifo tinham esse problema).

Requer DIPY (ConstrainedSphericalDeconvModel/auto_response_ssst/
peaks_from_model/dipy.reconst.dti, + matplotlib se --make-glyphs) -- nao
executado neste ambiente de desenvolvimento (mesma limitacao ja documentada
em scripts/11_peak_confusion_by_roi.py/scripts/12_visualize_fod_glyphs.py/
scripts/poc_multiprotocol_gt_reliability.py/scripts/07_downstream_dti_noddi.py).
Verificado por python3 -m py_compile e testes numericos isolados das
funcoes puramente numpy (bounding_box/find_best_crossing_patch/
glyph_polygon_xy/in_plane_directions/sh_energy_by_order/
_energy_frac_high_order/match_peaks_voxel/confusion_for_roi/_roi_r2), mesma
tecnica (extracao via AST) ja usada nos scripts irmaos.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, load_dwi, split_shells
from utils.masking import load_or_build_mask, load_roi_masks, JHU_TRACT_LABELS
from utils.sh_basis import max_order_for_n_directions
from utils.metrics import signal_bias, residual_std, r2_score_per_voxel


def _resolve_shell_key(shells: dict, shell_b: float, tol: float) -> float:
    """Duplicada de scripts/11_peak_confusion_by_roi.py/
    scripts/12_visualize_fod_glyphs.py (sem import cruzado entre scripts de
    etapas, mesmo padrao ja usado nos dois)."""
    best_key, best_diff = None, None
    for k in shells:
        if k == 0:
            continue
        diff = abs(k - shell_b)
        if best_diff is None or diff < best_diff:
            best_key, best_diff = k, diff
    if best_key is None or best_diff > tol:
        raise RuntimeError(f"shell {shell_b} nao encontrada (tol={tol})")
    return best_key


def fit_peaks(data, bvals, bvecs, shell_b, mask, shell_tol, sh_order,
              relative_peak_threshold, min_separation_angle, npeaks,
              exclude_idx=None, response=None, n_jobs=1):
    """CSD single-shell single-tissue (Tournier07) -- devolve
    (n_peaks_map, peak_dirs, peak_values, shm_coeff, response). Duplicada de
    scripts/11_peak_confusion_by_roi.py:fit_peaks (mesma assinatura/mesma
    convencao de exclude_idx). ATUALIZADO 2026-09-02: agora tambem devolve
    peak_dirs/peak_values (antes omitidos, so' usados nas metricas agregadas
    de contagem/energia) -- necessarios pro casamento de picos contra a
    referencia 'completa' via match_peaks_voxel/confusion_for_roi (ver
    rows_for_fit abaixo).

    `response` (ADITIVO 2026-09-02, default None = comportamento de sempre:
    estima via auto_response_ssst dentro do PROPRIO `mask`/`data` passados):
    se fornecido, PULA auto_response_ssst e usa esse response function
    diretamente no CSD -- necessario pra secao de glifos (ver main()), que
    reajusta CSD num SUB-VOLUME pequeno recortado ao redor do patch
    (--search-radius) so' pra velocidade; auto_response_ssst rodando DENTRO
    desse recorte pequeno (em vez do cerebro inteiro, como faz o fit de
    referencia 'completo') pode achar um voxel de fibra-unica bem pior (ou
    nenhum bom) so' por causa do recorte, produzindo FODs bem diferentes por
    um response function DIFERENTE/pior -- um artefato de implementacao, nao
    o efeito genuino de n_level caindo que a figura pretende mostrar. Sempre
    devolve o response REALMENTE usado (recebido ou recem-estimado), pra
    quem chamar poder reaproveitar no proximo fit.

    `n_jobs` (ADITIVO 2026-09-02, EXPERIMENTAL -- ver --n-jobs em main() e
    a nota de velocidade no docstring do modulo; default 1 = comportamento
    de sempre, `parallel=False`, identico a antes desta flag existir): se
    >1, roda `peaks_from_model` com `parallel=True, num_processes=n_jobs`
    -- o resto da pipeline (11_peak_confusion_by_roi.py/
    12_visualize_fod_glyphs.py/etc.) sempre usou `parallel=False` sem
    nenhuma nota explicando o motivo, entao NAO mudamos o default aqui nem
    nos outros scripts -- so' expomos a opcao pra quem quiser testar."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere

    shells = split_shells(bvals, tol=shell_tol)
    shell_key = _resolve_shell_key(shells, shell_b, shell_tol)
    idx = np.concatenate([shells[0], shells[shell_key]])
    idx.sort()
    if exclude_idx is not None:
        idx = np.setdiff1d(idx, np.asarray(exclude_idx), assume_unique=False)

    gtab = gradient_table(bvals[idx], bvecs[idx])
    vol = data[..., idx]

    if response is None:
        response, _ratio = auto_response_ssst(gtab, vol, roi_radii=10, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)

    sphere = get_sphere("repulsion724")
    parallel_kwargs = {"parallel": False}
    if n_jobs and n_jobs > 1:
        parallel_kwargs = {"parallel": True, "num_processes": n_jobs}
    peaks = peaks_from_model(
        model=csd_model, data=vol, sphere=sphere, mask=mask,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle, npeaks=npeaks,
        normalize_peaks=False,
        return_sh=True, sh_order=sh_order, sh_basis_type="descoteaux07",
        **parallel_kwargs,
    )
    n_peaks_map = (peaks.peak_values > 0).sum(axis=-1).astype(np.int32)
    n_peaks_map[~mask.astype(bool)] = -1
    return n_peaks_map, peaks.peak_dirs, peaks.peak_values, peaks.shm_coeff, response


def sh_energy_by_order(shm_coeff, sh_order, mask):
    """Duplicada de scripts/11_peak_confusion_by_roi.py:sh_energy_by_order
    (ver docstring la pro racional completo). Devolve
    dict {l: (energia_media_por_voxel, fracao_da_energia_total)}."""
    mask_bool = mask.astype(bool)
    coeffs_masked = shm_coeff[mask_bool]
    if coeffs_masked.shape[0] == 0:
        return {}
    energy_total_per_voxel = np.sum(coeffs_masked ** 2, axis=-1)
    energy_total_per_voxel_safe = np.where(energy_total_per_voxel > 0, energy_total_per_voxel, np.nan)
    out = {}
    start = 0
    for l in range(0, sh_order + 1, 2):
        block_size = 2 * l + 1
        end = start + block_size
        block = coeffs_masked[:, start:end]
        energy_l_per_voxel = np.sum(block ** 2, axis=-1)
        frac_l_per_voxel = energy_l_per_voxel / energy_total_per_voxel_safe
        out[l] = (float(np.nanmean(energy_l_per_voxel)), float(np.nanmean(frac_l_per_voxel)))
        start = end
    return out


def _energy_frac_high_order(energy_by_l):
    if not energy_by_l:
        return float("nan")
    return float(sum(frac for l, (_e, frac) in energy_by_l.items() if l >= 4))


def fit_dti(data, bvals, bvecs, shell_key, mask, shell_tol=100.0, exclude_idx=None):
    """Duplicada de scripts/07_downstream_dti_noddi.py:fit_dti (mesma
    convencao de exclude_idx -- ver docstring la pro racional completo).
    ADAPTADA: recebe `shell_key` ja RESOLVIDA (via _resolve_shell_key acima,
    mesma convencao ja usada em fit_peaks neste arquivo) em vez do
    `shell_b` bruto -- o original indexa `shells[shell_b]` diretamente, o
    que exige bater exatamente com uma chave de split_shells; aqui evitamos
    esse risco reaproveitando a chave ja resolvida com --shell-tol em
    main()."""
    import dipy.reconst.dti as dti
    from dipy.core.gradients import gradient_table

    shells = split_shells(bvals, tol=shell_tol)
    idx = np.concatenate([shells[0], shells[shell_key]])
    idx.sort()
    if exclude_idx is not None:
        idx = np.setdiff1d(idx, np.asarray(exclude_idx), assume_unique=False)
    gtab = gradient_table(bvals[idx], bvecs[idx])
    model = dti.TensorModel(gtab)
    fit = model.fit(data[..., idx], mask=mask)
    return {"FA": fit.fa, "MD": fit.md, "AD": fit.ad, "RD": fit.rd}


def _roi_r2(pred_1d, target_1d):
    """Duplicada de scripts/07_downstream_dti_noddi.py:_roi_r2 (ver
    docstring la pro racional completo -- R^2 da variancia ESPACIAL do mapa
    escalar, dentro da ROI, nao da predicao por direcao)."""
    if pred_1d.size < 2:
        return float("nan")
    return float(r2_score_per_voxel(pred_1d[np.newaxis, :], target_1d[np.newaxis, :])[0])


def match_peaks_voxel(true_dirs, true_vals, pred_dirs, pred_vals, threshold_deg):
    """Duplicada de scripts/11_peak_confusion_by_roi.py:match_peaks_voxel
    (ver docstring la pro racional completo -- casamento greedy mais-
    proximo-primeiro, direcoes com simetria antipodal). Devolve
    (n_tp, n_fp, n_fn, sum_ang_tp)."""
    true_list = [true_dirs[k] for k in range(true_dirs.shape[0]) if true_vals[k] > 0]
    pred_list = [pred_dirs[k] for k in range(pred_dirs.shape[0]) if pred_vals[k] > 0]
    n_true, n_pred = len(true_list), len(pred_list)
    if n_true == 0 and n_pred == 0:
        return 0, 0, 0, 0.0

    pairs = []
    for i, tv in enumerate(true_list):
        for j, pv in enumerate(pred_list):
            cos = abs(float(np.dot(tv, pv)))
            cos = min(1.0, max(-1.0, cos))
            ang = np.degrees(np.arccos(cos))
            pairs.append((ang, i, j))
    pairs.sort(key=lambda x: x[0])

    matched_true, matched_pred = set(), set()
    n_tp = 0
    sum_ang_tp = 0.0
    for ang, i, j in pairs:
        if ang > threshold_deg:
            break
        if i in matched_true or j in matched_pred:
            continue
        matched_true.add(i)
        matched_pred.add(j)
        n_tp += 1
        sum_ang_tp += float(ang)

    n_fn = n_true - len(matched_true)
    n_fp = n_pred - len(matched_pred)
    return n_tp, n_fp, n_fn, sum_ang_tp


def confusion_for_roi(gt_n_peaks, gt_dirs, gt_vals, pred_n_peaks, pred_dirs, pred_vals,
                       roi_mask, threshold_deg):
    """Duplicada de scripts/11_peak_confusion_by_roi.py:confusion_for_roi
    (ver docstring la -- aqui SEM a estratificacao simple/crossing daquele
    script, so' o agregado da ROI inteira, mesmo espirito minimalista do
    resto deste arquivo). Devolve dict TP/FP/FN/TN_voxels/sum_tp_angle_deg."""
    idxs = np.argwhere(roi_mask)
    tp = fp = fn = tn_voxels = 0
    sum_tp_angle_deg = 0.0
    for (x, y, z) in idxs:
        gt_n = gt_n_peaks[x, y, z]
        pr_n = pred_n_peaks[x, y, z]
        if gt_n < 0 or pr_n < 0:
            continue
        if gt_n == 0 and pr_n == 0:
            tn_voxels += 1
            continue
        vtp, vfp, vfn, vsum_ang = match_peaks_voxel(
            gt_dirs[x, y, z], gt_vals[x, y, z], pred_dirs[x, y, z], pred_vals[x, y, z],
            threshold_deg)
        tp += vtp
        fp += vfp
        fn += vfn
        sum_tp_angle_deg += vsum_ang
    return {"TP": tp, "FP": fp, "FN": fn, "TN_voxels": tn_voxels,
            "sum_tp_angle_deg": sum_tp_angle_deg}


def bounding_box(mask, center, radius):
    """Duplicada de scripts/12_visualize_fod_glyphs.py:bounding_box."""
    shape = mask.shape
    lo = [max(0, c - radius) for c in center]
    hi = [min(shape[d], center[d] + radius + 1) for d in range(3)]
    slices = tuple(slice(lo[d], hi[d]) for d in range(3))
    return slices, tuple(lo)


def find_best_crossing_patch(n_peaks_map, mask, patch_size, slice_axis,
                              min_peaks_for_crossing=2, min_mask_frac=0.5):
    """Duplicada de scripts/12_visualize_fod_glyphs.py:find_best_crossing_patch
    (ver docstring la pro racional completo)."""
    shape = n_peaks_map.shape
    out_axes = [a for a in range(3) if a != slice_axis]
    mask_bool = mask.astype(bool)
    crossing_bool = mask_bool & (n_peaks_map >= min_peaks_for_crossing)

    best = None
    for s in range(shape[slice_axis]):
        idx3 = [slice(None)] * 3
        idx3[slice_axis] = s
        idx3 = tuple(idx3)
        mask_slice = mask_bool[idx3]
        cross_slice = crossing_bool[idx3]
        n0, n1 = mask_slice.shape
        if n0 < patch_size or n1 < patch_size:
            continue
        for o0 in range(0, n0 - patch_size + 1):
            for o1 in range(0, n1 - patch_size + 1):
                patch_mask = mask_slice[o0:o0 + patch_size, o1:o1 + patch_size]
                n_masked = int(patch_mask.sum())
                if n_masked < min_mask_frac * patch_size * patch_size:
                    continue
                patch_cross = cross_slice[o0:o0 + patch_size, o1:o1 + patch_size]
                frac = float(patch_cross.sum()) / n_masked
                if best is None or frac > best[0]:
                    best = (frac, o0, o1, s)
    if best is None:
        return None
    frac, o0, o1, s = best
    return (o0, o1), s, frac


def in_plane_directions(slice_axis, n_angles=72):
    """Duplicada de scripts/12_visualize_fod_glyphs.py:in_plane_directions."""
    theta = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    zeros = np.zeros_like(theta)
    if slice_axis == 2:
        dirs = np.stack([cos_t, sin_t, zeros], axis=-1)
    elif slice_axis == 1:
        dirs = np.stack([cos_t, zeros, sin_t], axis=-1)
    else:
        dirs = np.stack([zeros, cos_t, sin_t], axis=-1)
    return dirs.astype(np.float64)


def glyph_polygon_xy(amplitudes, center_xy, glyph_scale, clip_negative=True):
    """Duplicada de scripts/12_visualize_fod_glyphs.py:glyph_polygon_xy."""
    amp = np.asarray(amplitudes, dtype=np.float64)
    if clip_negative:
        amp = np.clip(amp, 0.0, None)
    n_angles = amp.shape[0]
    theta = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    r = amp * glyph_scale
    x = center_xy[0] + r * np.cos(theta)
    y = center_xy[1] + r * np.sin(theta)
    return np.stack([x, y], axis=-1)


def render_glyph_field(ax, shm_patch, directions, sh_order, glyph_scale,
                        amplitude_ref=None, cmap_name="viridis"):
    """Duplicada de scripts/12_visualize_fod_glyphs.py:render_glyph_field
    (ver docstring la pro racional completo). Retorna o maior valor de
    amplitude visto (util pra normalizacao 'global')."""
    import matplotlib.pyplot as plt
    from dipy.reconst.shm import sh_to_sf
    from dipy.core.sphere import Sphere

    sphere = Sphere(xyz=directions)
    P, Q, _n_coef = shm_patch.shape
    sf = sh_to_sf(shm_patch.reshape(P * Q, -1), sphere, sh_order=sh_order,
                  basis_type="descoteaux07")
    sf = sf.reshape(P, Q, -1)
    sf_max = float(np.clip(sf, 0.0, None).max()) if sf.size else 0.0
    ref = amplitude_ref if amplitude_ref is not None else sf_max
    ref = ref if ref > 0 else 1.0
    cmap = plt.get_cmap(cmap_name)

    for i in range(P):
        for j in range(Q):
            amp = sf[i, j]
            amp_norm = amp / ref
            poly = glyph_polygon_xy(amp_norm, center_xy=(j, -i), glyph_scale=glyph_scale)
            peak_frac = float(np.clip(amp_norm, 0.0, 1.0).max())
            ax.fill(poly[:, 0], poly[:, 1], color=cmap(peak_frac), edgecolor="black",
                     linewidth=0.4)
    ax.set_xlim(-0.6, Q - 1 + 0.6)
    ax.set_ylim(-(P - 1) - 0.6, 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return sf_max


def rows_for_fit(tag, shell_b, n_level, is_full, sh_order, n_peaks_map, shm_coeff,
                  peak_dirs, peak_values, rois, min_peaks_for_crossing,
                  ref_n_peaks_map=None, ref_peak_dirs=None, ref_peak_values=None,
                  peak_match_threshold_deg=25.0, fa_map=None, ref_fa_map=None):
    """Uma linha por ROI, mesmo schema-base de colunas de
    scripts/poc_multiprotocol_gt_reliability.py:process_protocol_shell
    (frac_0peaks/frac_1peak/frac_crossing/mean_n_peaks/
    energy_frac_high_order), trocando 'protocol' por 'n_level' -- so' o eixo
    de comparacao muda (mesma aquisicao, N de entrada diferente, em vez de
    protocolos de aquisicao diferentes).

    ADICIONADO 2026-09-02 (metricas Tier 1, ver docstring do modulo):
    `ref_n_peaks_map`/`ref_peak_dirs`/`ref_peak_values` (picos da linha
    'completa') e `fa_map`/`ref_fa_map` (FA desta linha e da linha
    'completa') sao OPCIONAIS (default None) -- quando fornecidos E
    `is_full` for False, a linha tambem ganha TP/FP/FN/TN_voxels/precision/
    recall/mean_tp_angle_deg (casamento de picos contra a referencia) e
    FA_mae/FA_bias/FA_resid_std/FA_r2 (FA desta linha vs. FA da referencia).
    Na linha is_full=True (a propria referencia) essas colunas ficam
    sempre NaN -- comparar a referencia contra ela mesma nao carrega
    informacao (seria trivialmente precision=recall=1/FA_r2=1)."""
    rows = []
    for roi_name, roi_bool in rois.items():
        roi_mask = roi_bool
        n_voxels = int(roi_mask.sum())
        if n_voxels == 0:
            continue
        peaks_in_roi = n_peaks_map[roi_mask]
        energy_by_l = sh_energy_by_order(shm_coeff, sh_order, roi_mask)
        row = {
            "subject": tag, "roi": roi_name, "n_level": n_level, "is_full": is_full,
            "shell_b": shell_b, "sh_order": sh_order, "n_voxels": n_voxels,
            "frac_0peaks": float(np.mean(peaks_in_roi == 0)),
            "frac_1peak": float(np.mean(peaks_in_roi == 1)),
            "frac_crossing": float(np.mean(peaks_in_roi >= min_peaks_for_crossing)),
            "mean_n_peaks": float(np.mean(np.clip(peaks_in_roi, 0, None))),
            "energy_frac_high_order": _energy_frac_high_order(energy_by_l),
            "fit_failed": False,
        }
        if ref_n_peaks_map is not None and not is_full:
            conf = confusion_for_roi(ref_n_peaks_map, ref_peak_dirs, ref_peak_values,
                                      n_peaks_map, peak_dirs, peak_values, roi_mask,
                                      peak_match_threshold_deg)
            tp, fp, fn = conf["TP"], conf["FP"], conf["FN"]
            row["TP"] = tp
            row["FP"] = fp
            row["FN"] = fn
            row["TN_voxels"] = conf["TN_voxels"]
            row["precision"] = (tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
            row["recall"] = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
            row["mean_tp_angle_deg"] = (conf["sum_tp_angle_deg"] / tp) if tp > 0 else float("nan")
        else:
            row["TP"] = row["FP"] = row["FN"] = row["TN_voxels"] = np.nan
            row["precision"] = row["recall"] = row["mean_tp_angle_deg"] = np.nan
        if fa_map is not None and ref_fa_map is not None and not is_full:
            pred_fa, ref_fa = fa_map[roi_mask], ref_fa_map[roi_mask]
            diff = pred_fa - ref_fa
            row["FA_mae"] = float(np.nanmean(np.abs(diff)))
            row["FA_bias"] = signal_bias(pred_fa, ref_fa)
            row["FA_resid_std"] = residual_std(pred_fa, ref_fa)
            row["FA_r2"] = _roi_r2(pred_fa, ref_fa)
        else:
            row["FA_mae"] = row["FA_bias"] = row["FA_resid_std"] = row["FA_r2"] = np.nan
        rows.append(row)
    return rows


def failed_rows(tag, shell_b, n_level, is_full, sh_order, rois):
    """Mesmo espirito de _failed_rows em 11_peak_confusion_by_roi.py -- um
    fit de CSD (ou de DTI, ver main()) que falha (ex.: sistema severamente
    sub-determinado em n_level bem baixo) grava uma linha fit_failed=True
    por ROI em vez de derrubar a rodada inteira. Inclui todas as colunas
    Tier 1 adicionadas em 2026-09-02, todas NaN."""
    return [{"subject": tag, "roi": roi_name, "n_level": n_level, "is_full": is_full,
              "shell_b": shell_b, "sh_order": sh_order, "n_voxels": np.nan,
              "frac_0peaks": np.nan, "frac_1peak": np.nan, "frac_crossing": np.nan,
              "mean_n_peaks": np.nan, "energy_frac_high_order": np.nan,
              "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN_voxels": np.nan,
              "precision": np.nan, "recall": np.nan, "mean_tp_angle_deg": np.nan,
              "FA_mae": np.nan, "FA_bias": np.nan, "FA_resid_std": np.nan, "FA_r2": np.nan,
              "fit_failed": True}
            for roi_name in rois]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="pasta com '<tag>_rrin_triplets.npz' (ex.: work_dir/subsampling) -- "
                          "usa o MESMO esquema de subamostragem ja usado pra treinar as redes, "
                          "so' pra saber quais direcoes sao 'alvo' (excluidas do fit) em cada "
                          "n_level -- nenhuma reconstrucao e' usada, so a geometria do esquema.")
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-levels", type=int, nargs="+", required=True,
                     help="niveis de subamostragem ja presentes no esquema do sujeito escolhido "
                          "(mesmos --levels usados em scripts/02b_build_rrin_triplets.py)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--subjects", default=None,
                     help="tag(s) separadas por virgula -- se omitido, usa o primeiro sujeito "
                          "do split com a shell/n_levels pedidos disponiveis (recomendado passar "
                          "explicitamente o sujeito padrao ja usado no evaluate)")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--sh-order", type=int, default=None,
                     help="forca a MESMA ordem SH em todo mundo (inclusive na linha de "
                          "referencia 'completa'), em vez do default (None) que usa a ordem "
                          "auto max_order_for_n_directions(n_level) de cada ponto -- mesma "
                          "convencao/mesmo racional do --sh-order de "
                          "poc_multiprotocol_gt_reliability.py. Deixe em None (default) se o "
                          "objetivo e' ver a ordem mudando junto com N, como a producao faz de "
                          "verdade.")
    ap.add_argument("--min-peaks-for-crossing", type=int, default=2)
    ap.add_argument("--npeaks", type=int, default=3)
    ap.add_argument("--relative-peak-threshold", type=float, default=0.5)
    ap.add_argument("--min-separation-angle", type=float, default=25.0)
    ap.add_argument("--n-jobs", type=int, default=1,
                     help="EXPERIMENTAL (2026-09-02, nao testado neste ambiente de "
                          "desenvolvimento sem DIPY -- ver docstring de fit_peaks): se >1, "
                          "roda peaks_from_model com parallel=True/num_processes=N-JOBS nos "
                          "fits de CSD de mascara INTEIRA (referencia 'completo' + cada "
                          "n_level) -- o custo dominante do script. Default 1 = sequencial, "
                          "identico ao comportamento de sempre (mesmo default de todo o resto "
                          "da pipeline, parallel=False). NAO se aplica aos fits pequenos da "
                          "secao de glifos (sub-volume ja e' rapido, overhead de "
                          "multiprocessing nao compensa). Combine com --cpus-per-task no "
                          "sbatch (ver N_JOBS no slurm/poc_native_order_curve.sh) -- se a API "
                          "de multiprocessing do DIPY instalado no cluster nao aceitar o kwarg "
                          "'num_processes' (versoes mais antigas usam 'nbr_processes'), o erro "
                          "vai aparecer na primeira chamada de fit_peaks; me avise o erro exato "
                          "que eu ajusto.")
    ap.add_argument("--peak-match-threshold-deg", type=float, default=25.0,
                     help="limiar (graus) do casamento de picos de cada n_level contra a "
                          "referencia 'completa' (ver TP/FP/FN/precision/recall/"
                          "mean_tp_angle_deg, adicionado 2026-09-02) -- mesmo default/mesmo "
                          "racional de --peak-match-threshold-deg em "
                          "scripts/11_peak_confusion_by_roi.py.")
    ap.add_argument("--roi-tracts", default=None,
                     help=f"lista separada por virgula de tratos JHU (opcoes: "
                          f"{','.join(JHU_TRACT_LABELS)}) -- default so 'whole_mask'")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--make-glyphs", action="store_true",
                     help="gera tambem uma figura com um painel de glifo por n_level (mais um "
                          "painel 'completo' de ancora), todos no MESMO patch/voxels fisicos -- "
                          "ver docstring do modulo.")
    ap.add_argument("--glyph-n-levels", type=int, nargs="+", default=None,
                     help="subconjunto de --n-levels pra desenhar glifo (default: todos os "
                          "--n-levels) -- util pra nao gerar uma figura com paineis demais.")
    ap.add_argument("--out-fig", default=None,
                     help="caminho do PNG da figura de glifos -- necessario com --make-glyphs")
    ap.add_argument("--search-radius", type=int, default=15)
    ap.add_argument("--patch-size", type=int, default=4)
    ap.add_argument("--slice-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--min-mask-frac", type=float, default=0.5)
    ap.add_argument("--center-voxel", default=None,
                     help="'X,Y,Z' em coordenadas GLOBAIS -- pula a busca automatica de "
                          "cruzamento e centra o patch de glifos nesse voxel especifico (mesmo "
                          "sentido de --center-voxel em scripts/12_visualize_fod_glyphs.py).")
    ap.add_argument("--glyph-scale", type=float, default=0.45)
    ap.add_argument("--glyph-n-angles", type=int, default=72)
    ap.add_argument("--normalize", choices=["global", "per_voxel"], default="global",
                     help="'global' (default): todos os paineis normalizados pelo pico do "
                          "painel 'completo' (referencia), preservando diferenca de MAGNITUDE "
                          "conforme N cai -- recomendado pra esta curva, ja que a magnitude "
                          "caindo (nao so a forma) e' parte do fenomeno. 'per_voxel': cada "
                          "painel normalizado pelo seu proprio pico (so mostra forma).")
    args = ap.parse_args()

    if args.make_glyphs and args.out_fig is None:
        sys.exit("--make-glyphs precisa de --out-fig")

    roi_tracts = [t.strip() for t in args.roi_tracts.split(",")] if args.roi_tracts else None
    glyph_levels = args.glyph_n_levels if args.glyph_n_levels is not None else list(args.n_levels)
    missing_glyph_levels = sorted(set(glyph_levels) - set(args.n_levels))
    if missing_glyph_levels:
        sys.exit(f"--glyph-n-levels {missing_glyph_levels} nao esta em --n-levels -- inclua "
                  f"esses niveis em --n-levels tambem (ou remova de --glyph-n-levels).")

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split] \
        if args.split != "all" else load_manifest(args.manifest)

    def _tag_of(e):
        return e.subject if not e.session else f"{e.subject}_{e.session}"

    if args.subjects:
        wanted = {t.strip() for t in args.subjects.split(",") if t.strip()}
        entries = [e for e in entries if _tag_of(e) in wanted]
        if not entries:
            sys.exit(f"Nenhum dos sujeitos pedidos em --subjects foi encontrado no split "
                      f"{args.split!r}.")
    if not entries:
        sys.exit(f"Nenhum sujeito no split {args.split!r}.")
    e = entries[0]
    tag = _tag_of(e)
    print(f"Sujeito: {tag}", flush=True)

    bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
    data, affine, header = load_dwi(e.dwi_path)
    shells = split_shells(bvals, tol=args.shell_tol)
    b0_idx = shells.get(0, np.array([], dtype=int))
    if b0_idx.size == 0:
        sys.exit(f"{tag}: nenhum volume b0 encontrado.")
    b0_mean = data[..., b0_idx].mean(axis=-1)
    mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)
    mask_bool = mask.astype(bool)

    rois = {"whole_mask": mask_bool}
    if roi_tracts:
        rois.update({k: (mask_bool & v) for k, v in
                     load_roi_masks(e.dwi_path, roi_tracts, base_mask=mask).items()})

    shell_key = _resolve_shell_key(shells, args.shell_b, args.shell_tol)
    n_dirs_full = int(shells[0].size + shells[shell_key].size)
    sh_order_full = args.sh_order or max_order_for_n_directions(n_dirs_full)

    trip_path = Path(args.triplets_dir) / f"{tag}_rrin_triplets.npz"
    if not trip_path.exists():
        sys.exit(f"{trip_path} nao existe -- confira --triplets-dir/--subjects (precisa do "
                  f"esquema de subamostragem ja gerado por scripts/02b_build_rrin_triplets.py "
                  f"pra este sujeito).")
    trip = np.load(trip_path)

    all_rows = []

    print(f"[completo] n_dirs_full={n_dirs_full}, sh_order={sh_order_full} -- ajustando CSD "
          f"(referencia, sem exclusao nenhuma)...", flush=True)
    try:
        full_n_peaks_map, full_peak_dirs, full_peak_values, full_shm_coeff, full_response = \
            fit_peaks(data, bvals, bvecs, args.shell_b, mask, args.shell_tol, sh_order_full,
                      args.relative_peak_threshold, args.min_separation_angle, args.npeaks,
                      n_jobs=args.n_jobs)
        all_rows.extend(rows_for_fit(tag, args.shell_b, n_dirs_full, True, sh_order_full,
                                      full_n_peaks_map, full_shm_coeff,
                                      full_peak_dirs, full_peak_values, rois,
                                      args.min_peaks_for_crossing))
    except Exception as exc:
        print(f"[erro] {tag}: CSD falhou pra referencia 'completo' "
              f"({type(exc).__name__}: {exc}) -- sem essa linha de ancora, seguindo pros "
              f"n_levels mesmo assim (TP/FP/FN/precision/recall/mean_tp_angle_deg ficarao "
              f"NaN em todas as linhas, sem referencia pra comparar os picos).", flush=True)
        full_n_peaks_map, full_peak_dirs, full_peak_values, full_shm_coeff, full_response = \
            None, None, None, None, None

    print(f"[completo] ajustando DTI (referencia, sem exclusao nenhuma)...", flush=True)
    try:
        full_fa_map = fit_dti(data, bvals, bvecs, shell_key, mask,
                               shell_tol=args.shell_tol)["FA"]
    except Exception as exc:
        print(f"[erro] {tag}: DTI falhou pra referencia 'completo' "
              f"({type(exc).__name__}: {exc}) -- FA_r2/FA_mae/FA_bias/FA_resid_std ficarao "
              f"NaN em todas as linhas.", flush=True)
        full_fa_map = None

    for n_level in args.n_levels:
        trip_key = f"{args.shell_b}__{n_level}__target"
        if trip_key not in trip.files:
            print(f"[aviso] {tag}: sem esquema pra n_level={n_level} (chave {trip_key!r} "
                  f"ausente em {trip_path}) -- pulando este ponto da curva.", flush=True)
            continue
        target_idx = trip[trip_key]
        sh_order = args.sh_order or max_order_for_n_directions(n_level)
        print(f"[n_level={n_level}] sh_order={sh_order} (auto via "
              f"max_order_for_n_directions({n_level}))"
              if args.sh_order is None else
              f"[n_level={n_level}] sh_order={sh_order} (forcado via --sh-order)", flush=True)
        try:
            n_peaks_map, peak_dirs, peak_values, shm_coeff, _resp = fit_peaks(
                data, bvals, bvecs, args.shell_b, mask, args.shell_tol, sh_order,
                args.relative_peak_threshold, args.min_separation_angle, args.npeaks,
                exclude_idx=target_idx, n_jobs=args.n_jobs)
            fit_ok = True
        except Exception as exc:
            print(f"[aviso] {tag}: CSD falhou pra n_level={n_level} "
                  f"({type(exc).__name__}: {exc}) -- gravando fit_failed e seguindo.",
                  flush=True)
            all_rows.extend(failed_rows(tag, args.shell_b, n_level, False, sh_order, rois))
            fit_ok = False
        if not fit_ok:
            continue
        try:
            fa_map = fit_dti(data, bvals, bvecs, shell_key, mask, shell_tol=args.shell_tol,
                              exclude_idx=target_idx)["FA"]
        except Exception as exc:
            print(f"[aviso] {tag}: DTI falhou pra n_level={n_level} "
                  f"({type(exc).__name__}: {exc}) -- FA_r2/FA_mae/FA_bias/FA_resid_std "
                  f"ficarao NaN nesta linha (CSD ja tinha convergido, entao as demais "
                  f"colunas ainda sao gravadas).", flush=True)
            fa_map = None
        all_rows.extend(rows_for_fit(
            tag, args.shell_b, n_level, False, sh_order, n_peaks_map, shm_coeff,
            peak_dirs, peak_values, rois, args.min_peaks_for_crossing,
            ref_n_peaks_map=full_n_peaks_map, ref_peak_dirs=full_peak_dirs,
            ref_peak_values=full_peak_values,
            peak_match_threshold_deg=args.peak_match_threshold_deg,
            fa_map=fa_map, ref_fa_map=full_fa_map))

    if not all_rows:
        sys.exit("Nenhum resultado -- confira --triplets-dir/--n-levels/--shell-b.")

    df = pd.DataFrame(all_rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nMetricas salvas em {out_csv}")

    cols = ["roi", "n_level", "is_full", "sh_order", "n_voxels", "frac_0peaks", "frac_1peak",
            "frac_crossing", "mean_n_peaks", "energy_frac_high_order",
            "precision", "recall", "mean_tp_angle_deg", "FA_r2", "fit_failed"]
    ok = df[~df["fit_failed"]]
    print("\nCurva nativa (sem reconstrucao nenhuma) -- estrutura de cruzamento vs. n_level, "
          "por ROI (ordenado por ROI, depois n_level):")
    print(ok[cols].sort_values(["roi", "n_level"]).to_string(index=False))

    print("\nLeitura (frac_crossing/energy_frac_high_order): compare com a mesma metrica de uma "
          "reconstrucao real (mesmo n_level, ver scripts/11_peak_confusion_by_roi.py) contra o "
          "PONTO desta curva no mesmo n_level -- se a reconstrucao fica bem acima do que N "
          "sozinho sustenta nativamente, e' evidencia de estrutura genuinamente recuperada; se "
          "fica perto ou abaixo, a rede nao esta indo muito alem do piso imposto pela propria "
          "contagem de direcoes. CUIDADO (ver addendum secao 22.1): essa comparacao so' e' "
          "confiavel quando os dois lados usam sh_order comparavel -- energy_frac_high_order "
          "avaliado em ordem cheia/descasada pode inflar artificialmente por instabilidade do "
          "ajuste, nao por informacao angular genuina.")
    print("\nLeitura (precision/recall/mean_tp_angle_deg/FA_r2, adicionado 2026-09-02): estas "
          "sao as metricas 'Tier 1' (mais diretas/menos sujeitas ao artefato de ordem acima) -- "
          "compare da MESMA forma, ponto a ponto no mesmo n_level, contra a reconstrucao real "
          "(precision/recall/mean_tp_angle_deg de scripts/11_peak_confusion_by_roi.py, FA_r2 de "
          "scripts/07_downstream_dti_noddi.py). Aqui na curva nativa, ambas comparam contra a "
          "linha 'completa' (is_full=True) deste mesmo sujeito/aquisicao -- fica NaN nessa "
          "propria linha de ancora.")

    if not args.make_glyphs:
        return

    if full_n_peaks_map is None:
        sys.exit("--make-glyphs pedido mas o fit 'completo' falhou (ver erro acima) -- sem ele "
                  "nao ha' como escolher o patch de cruzamento de referencia.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manual_center_voxel = None
    if args.center_voxel:
        parts = [p.strip() for p in args.center_voxel.split(",")]
        if len(parts) != 3:
            sys.exit(f"--center-voxel precisa ser 'X,Y,Z', recebi {args.center_voxel!r}.")
        manual_center_voxel = tuple(int(p) for p in parts)
        centroid = manual_center_voxel
    else:
        centroid = tuple(int(round(c)) for c in np.argwhere(mask_bool).mean(axis=0))
    slices, origin = bounding_box(mask_bool, centroid, args.search_radius)
    print(f"\n[glifos] {'Voxel central (fixado manualmente)' if manual_center_voxel else 'Centroide da mascara'}: "
          f"{centroid}; sub-volume: {slices} (origem {origin})", flush=True)

    sub_mask = mask_bool[slices]
    sub_data = data[slices]
    sub_full_n_peaks = full_n_peaks_map[slices]

    if manual_center_voxel is not None:
        local_center = tuple(manual_center_voxel[d] - origin[d] for d in range(3))
        out_axes = [a for a in range(3) if a != args.slice_axis]
        if sub_mask.shape[out_axes[0]] < args.patch_size or sub_mask.shape[out_axes[1]] < args.patch_size:
            sys.exit("--center-voxel/--patch-size nao cabem no sub-volume recortado -- aumente "
                      "--search-radius.")
        slice_idx = local_center[args.slice_axis]
        o0 = max(0, min(local_center[out_axes[0]] - args.patch_size // 2,
                         sub_mask.shape[out_axes[0]] - args.patch_size))
        o1 = max(0, min(local_center[out_axes[1]] - args.patch_size // 2,
                         sub_mask.shape[out_axes[1]] - args.patch_size))
    else:
        found = find_best_crossing_patch(
            sub_full_n_peaks, sub_mask, args.patch_size, args.slice_axis,
            min_peaks_for_crossing=args.min_peaks_for_crossing,
            min_mask_frac=args.min_mask_frac)
        if found is None:
            sys.exit("Nenhum patch candidato atingiu --min-mask-frac -- tente aumentar "
                      "--search-radius ou diminuir --min-mask-frac/--patch-size.")
        (o0, o1), slice_idx, crossing_frac = found
        print(f"[glifos] Melhor patch (fit completo): fracao de cruzamento = "
              f"{crossing_frac:.1%} (slice_axis={args.slice_axis}, indice={slice_idx}, "
              f"origem={(o0, o1)}) -- MESMOS voxels usados em todos os paineis.", flush=True)

    def _patch_slices(o0, o1, s, axis, size):
        idx3 = [None, None, None]
        out_axes = [a for a in range(3) if a != axis]
        idx3[out_axes[0]] = slice(o0, o0 + size)
        idx3[out_axes[1]] = slice(o1, o1 + size)
        idx3[axis] = slice(s, s + 1)
        return tuple(idx3)

    patch_slices_sub = _patch_slices(o0, o1, slice_idx, args.slice_axis, args.patch_size)
    directions = in_plane_directions(args.slice_axis, n_angles=args.glyph_n_angles)

    panels = []  # (label, shm_patch, sh_order_desse_painel)
    full_shm_patch = full_shm_coeff[slices][patch_slices_sub].reshape(
        args.patch_size, args.patch_size, -1)
    panels.append((f"completo (n{n_dirs_full})", full_shm_patch, sh_order_full))

    for n_level in glyph_levels:
        trip_key = f"{args.shell_b}__{n_level}__target"
        if trip_key not in trip.files:
            print(f"[aviso] sem esquema pra n_level={n_level}, pulando painel de glifo.",
                  flush=True)
            continue
        target_idx = trip[trip_key]
        sh_order = args.sh_order or max_order_for_n_directions(n_level)
        try:
            # response=full_response (FIX 2026-09-02, ver docstring de fit_peaks): reusa o
            # MESMO response function estimado uma vez no fit 'completo' de CORPO INTEIRO
            # (full_response, capturado la em cima), em vez de deixar fit_peaks reestimar via
            # auto_response_ssst dentro do sub-volume PEQUENO recortado ao redor do patch --
            # sem isso, um recorte proximo de um voxel de cruzamento (escolhido de proposito
            # por ter cruzamento) pode nao conter um bom voxel de fibra unica, estimando um
            # response function bem diferente/pior do que o usado no painel 'completo' e
            # inflando artificialmente a diferenca visual entre paineis -- um artefato de
            # kernel de deconvolucao diferente, nao o efeito genuino de n_level caindo que a
            # figura pretende mostrar.
            _n_peaks, _dirs, _vals, shm, _resp = fit_peaks(
                sub_data, bvals, bvecs, args.shell_b, sub_mask, args.shell_tol, sh_order,
                args.relative_peak_threshold, args.min_separation_angle, args.npeaks,
                exclude_idx=target_idx, response=full_response)
        except Exception as exc:
            print(f"[aviso] CSD falhou pro painel n_level={n_level} "
                  f"({type(exc).__name__}: {exc}), pulando.", flush=True)
            continue
        shm_patch = shm[patch_slices_sub].reshape(args.patch_size, args.patch_size, -1)
        panels.append((f"n{n_level}", shm_patch, sh_order))

    if len(panels) < 2:
        sys.exit("Menos de 2 paineis de glifo disponiveis (completo + pelo menos 1 n_level) -- "
                  "confira --glyph-n-levels/--triplets-dir.")

    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.6))
    amplitude_ref = None
    for i, (ax, (label, shm_patch, panel_sh_order)) in enumerate(zip(axes, panels)):
        ref = amplitude_ref if args.normalize == "global" else None
        peak = render_glyph_field(ax, shm_patch, directions, panel_sh_order, args.glyph_scale,
                                   amplitude_ref=ref)
        if i == 0 and args.normalize == "global":
            amplitude_ref = peak
        ax.set_title(f"{label}\n(sh_order={panel_sh_order})", fontsize=8)

    fig.suptitle(f"{tag} -- shell{int(args.shell_b)} -- FOD nativo (sem reconstrucao) vs. "
                 f"n_level, MESMO voxel em todos os paineis -- normalizacao={args.normalize}",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    print(f"\nFigura de glifos salva em {out_fig}", flush=True)


if __name__ == "__main__":
    main()