#!/usr/bin/env python3
"""
Prova de conceito (independente do RCAE/RRIN/etc., nao precisa de nenhuma
etapa anterior ter rodado -- so precisa dos dados brutos): testa QUAO
CONFIAVEL e' o ground truth de CSD que a pipeline inteira usa como
referencia (`gt_n_peaks`/`ref_energy_frac_high_order` em
`scripts/11_peak_confusion_by_roi.py`/`scripts/12_visualize_fod_glyphs.py`),
comparando o MESMO sujeito adquirido em VARIOS protocolos diferentes
(b-value e/ou numero de direcoes diferentes), em vez de assumir que o fit
de CSD em b1000/64dir e' uma verdade absoluta.

Motivacao (ver addendum, discussao 2026-09-02): CSD depende de contraste
angular real entre populacoes de fibra pra separar picos, e esse
contraste cresce com o b-value -- a literatura classica de CSD (Tournier
2007, Descoteaux 2009 e comparacoes de protocolo posteriores) costuma
recomendar b~2000-3000 s/mm^2 pra cruzamentos bem resolvidos, sobretudo
em angulos mais fechados. Em b=1000 (o que a pipeline usa hoje como GT
"oficial"), o CSD ainda funciona mas com MENOS contraste angular -- entao
o proprio GT usado em toda a avaliacao (nao so mais um metodo comparado)
pode estar subestimando levemente cruzamentos mais sutis. Este script
mede isso diretamente em vez de so argumentar sobre isso.

DESENHO DO TESTE (dado o layout descrito pela usuaria -- 1 pessoa, varios
protocolos, cada um numa subpasta de --data-root, mesmo --name-suffix,
MESMA resolucao entre protocolos):
  1) REPETIBILIDADE em b FIXO: protocolos com o MESMO b-value e MESMO (ou
     quase) numero de direcoes, so variando o numero de b0 (ex.: SeqA1/
     SeqA5/SeqA6, todos b1000/64dir) -- mede quanto o GT varia entre duas
     aquisicoes quase identicas, ou seja, o "piso de ruido" do proprio
     fit de CSD, antes mesmo de comparar b-values diferentes.
  2) SENSIBILIDADE A N DE DIRECOES, b FIXO: mesmo b-value, N diferente
     (ex.: SeqA1 b1000/64dir vs. SeqA4 b1000/46dir).
  3) SENSIBILIDADE AO B-VALUE, N ~fixo: b crescente com numero de
     direcoes parecido (ex. ~64dir: b1000 via SeqA1/A5/A6, b1500 via
     SeqD ou o subconjunto b1500 de SeqC, b2000 via SeqB [77dir, um
     pouco mais rico]) -- o teste mais direto da hipotese acima.
  4) --MSMT (2026-09-02, a pedido da usuaria -- "nao seria melhor fazer
     multishell em vez de separar?"): ANCORA extra, so' pra protocolos
     multi-shell (SeqB/SeqC/SeqD) -- ajusta MSMT-CSD (multi-shell
     multi-tecido, Jeurissen et al. NeuroImage 2014) usando TODAS as
     shells nao-zero daquele protocolo DE UMA VEZ, em vez de uma shell por
     vez. Isso NAO substitui as comparacoes 1-3 acima (que respondem uma
     pergunta diferente e continuam validas: "o GT single-shell que a
     pipeline usa de fato, b1000/64dir, e' confiavel comparado a outras
     opcoes single-shell?" -- a pipeline inteira, RRIN/RCAE/estrela, so'
     interpola DENTRO de uma shell, entao essa e' a comparacao
     estruturalmente relevante pro que a tese realmente faz) -- e' um
     ponto de referencia ADICIONAL, tipicamente mais preciso/robusto por
     usar mais contraste de difusao de uma vez (separa resposta de
     materia branca/cinzenta/liquor), pra medir o quao longe o GT
     single-shell fica do "melhor palpite possivel" pra esse sujeito.

Nao exige que os protocolos estejam no mesmo grid/afim exatamente (nao faz
comparacao voxel-a-voxel) -- cada protocolo/shell tem sua propria mascara
de cerebro (mesma logica de `utils.masking.load_or_build_mask` usada em
todo o resto da pipeline) e as metricas sao agregadas por mascara inteira
(ou por ROI JHU, se --roi-tracts for passado). Se algum dia quiser
comparacao voxel-a-voxel, precisa antes registrar os protocolos entre si
-- fora do escopo deste script (so avisa se shape/affine diferirem entre
protocolos, pra usuaria saber se um registro seria necessario).

Como os dados sao descobertos: reaproveita `utils.manifest.discover_dwi_files`
(mesma varredura recursiva de `scripts/01_prepare_data.py`) -- com o layout
"<data_root>/<pasta_do_protocolo>/<nome><name_suffix>.{nii,bval,bvec}"
(2 niveis, sem sub-pasta de sessao), cada pasta de protocolo vira uma
entrada com `subject` = nome da propria pasta (ex. "SeqA1", "SeqB").

Requer DIPY.

Uso:
    python scripts/poc_multiprotocol_gt_reliability.py \
        --data-root /caminho/para/folder_main \
        --name-suffix _geomcorr \
        --out-csv work_dir/metrics/poc_multiprotocol_gt_reliability.csv

    # pra restringir a alguns protocolos so (nomes das subpastas):
    python scripts/poc_multiprotocol_gt_reliability.py \
        --data-root /caminho/para/folder_main --name-suffix _geomcorr \
        --protocols SeqA1,SeqA4,SeqA5,SeqA6,SeqB,SeqD \
        --out-csv work_dir/metrics/poc_multiprotocol_gt_reliability.csv

    # com ROIs de trato (mesma convencao de --roi-tracts do resto da pipeline):
    python scripts/poc_multiprotocol_gt_reliability.py \
        --data-root /caminho/para/folder_main --name-suffix _geomcorr \
        --roi-tracts FX,CGC,CGH,UF \
        --out-csv work_dir/metrics/poc_multiprotocol_gt_reliability.csv

--MAKE-GLYPHS (2026-09-02, a pedido da usuaria -- "tb queria glifo neste
poc"): ADITIVO. Gera UMA figura com um painel de glifo por (protocolo,
shell) -- cada painel mostra o FOD na regiao de cruzamento mais nitido
DAQUELE protocolo/shell especifico (mesma logica de selecao automatica de
scripts/12_visualize_fod_glyphs.py: recorta um sub-volume cubico centrado
no centroide da mascara, desliza uma janela --patch-size x --patch-size
procurando a posicao com maior fracao de voxels com cruzamento). NAO
refaz o ajuste de CSD -- reaproveita o mesmo n_peaks_map/shm_coeff
whole-brain que process_protocol_shell() ja calcula pras metricas, so'
recorta (fatiamento numpy, nao um novo fit). Como os protocolos podem nao
estar no mesmo grid exatamente, cada painel mostra a MELHOR regiao de
cruzamento DAQUELE protocolo (nao necessariamente o mesmo voxel fisico
entre paineis) -- serve pra comparar visualmente "tipicamente, que forma
de FOD cada protocolo consegue recuperar num cruzamento", nao pra
comparar voxel-a-voxel.

Uso (gera work_dir/figures/poc_multiprotocol_glyphs.png alem do CSV):
    python scripts/poc_multiprotocol_gt_reliability.py \
        --data-root /caminho/para/folder_main --name-suffix _geomcorr \
        --make-glyphs --out-fig work_dir/figures/poc_multiprotocol_glyphs.png \
        --out-csv work_dir/metrics/poc_multiprotocol_gt_reliability.csv

Requer, alem de DIPY: nibabel (ja requerido) + matplotlib (so' se
--make-glyphs).

ATENCAO sobre --msmt: `dipy.reconst.mcsd` (MSMT-CSD) nao e' usado em
NENHUM outro lugar desta pipeline ate agora (todo o resto usa CSD
single-shell single-tissio via `ConstrainedSphericalDeconvModel`/
`auto_response_ssst`, uma API bem mais estavel entre versoes do DIPY) --
a implementacao de `fit_msmt_csd_peaks` abaixo segue o tutorial oficial
do DIPY (`dipy.reconst.mcsd`: `mask_for_response_msmt`/
`response_from_mask_msmt`/`multi_shell_fiber_response`/
`MultiShellDeconvModel`), mas NAO foi executada neste ambiente de
desenvolvimento (sem dipy) -- se a assinatura de alguma dessas funcoes
reclamar, confira `python -c "import dipy; print(dipy.__version__)"`
contra a documentacao de `dipy.reconst.mcsd` da versao instalada no
cluster.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import discover_dwi_files
from utils.gradients import load_bval_bvec, load_dwi, split_shells
from utils.masking import load_or_build_mask, load_roi_masks, JHU_TRACT_LABELS


def max_order_for_n_directions(n_dirs: int) -> int:
    """Maior ordem par l_max tal que o numero de coeficientes SH,
    (l_max+1)(l_max+2)/2, seja <= n_dirs -- mesma formula usada em
    utils/sh_basis.py e reimplementada em scripts/poc_csd_direction_count.py
    (duplicada de proposito aqui tambem, mesmo espirito de "sem import
    cruzado entre scripts de prova de conceito" ja documentado no
    addendum)."""
    l_max = 0
    while True:
        next_l = l_max + 2
        n_coef = (next_l + 1) * (next_l + 2) // 2
        if n_coef > n_dirs:
            break
        l_max = next_l
    return l_max


def fit_csd_peaks(vol, bvals_sub, bvecs_sub, mask, sh_order, npeaks,
                   relative_peak_threshold, min_separation_angle):
    """Ajusta CSD single-shell single-tissue no subconjunto de volumes dado
    (`vol`, ja incluindo os b0) e devolve (n_peaks_map, shm_coeff) dentro
    da mascara -- mesma logica/mesmos parametros de
    scripts/poc_csd_direction_count.py:fit_csd_peaks (duplicada aqui, nao
    importada, mesmo padrao ja usado entre scripts desta linha)."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere

    gtab = gradient_table(bvals_sub, bvecs_sub)
    response, _ratio = auto_response_ssst(gtab, vol, roi_radii=10, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)

    sphere = get_sphere("repulsion724")
    peaks = peaks_from_model(
        model=csd_model, data=vol, sphere=sphere, mask=mask,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle, npeaks=npeaks,
        parallel=False, normalize_peaks=False,
        return_sh=True, sh_order=sh_order, sh_basis_type="descoteaux07",
    )
    n_peaks_map = (peaks.peak_values > 0).sum(axis=-1).astype(np.int32)
    n_peaks_map[~mask.astype(bool)] = -1
    return n_peaks_map, peaks.shm_coeff


def sh_energy_by_order(shm_coeff, sh_order, mask):
    """Decompoe a energia dos coeficientes SH por ordem l (0,2,4,...,sh_order)
    -- mesma logica de scripts/poc_csd_direction_count.py:sh_energy_by_order
    (duplicada aqui, ver docstring la pro racional completo). Devolve dict
    {l: (energia_media_por_voxel, fracao_da_energia_total)}."""
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


def bounding_box(mask, center, radius):
    """Sub-volume cubico (2*radius+1)^3 centrado em `center`, recortado
    (clip) pra caber dentro de `mask.shape` -- duplicada de
    scripts/12_visualize_fod_glyphs.py:bounding_box (mesmo padrao de "sem
    import cruzado entre scripts", ver docstring do modulo). Retorna
    (slices, origin)."""
    shape = mask.shape
    lo = [max(0, c - radius) for c in center]
    hi = [min(shape[d], center[d] + radius + 1) for d in range(3)]
    slices = tuple(slice(lo[d], hi[d]) for d in range(3))
    return slices, tuple(lo)


def find_best_crossing_patch(n_peaks_map, mask, patch_size, slice_axis,
                              min_peaks_for_crossing=2, min_mask_frac=0.5):
    """Duplicada de scripts/12_visualize_fod_glyphs.py:find_best_crossing_patch
    (mesma logica/mesma assinatura -- ver docstring la pro racional
    completo). Retorna (origin_in_out_axes, slice_index, crossing_frac) ou
    None se nenhum patch candidato atingir min_mask_frac."""
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
    (mesma logica -- ver docstring la pro racional completo). Retorna o
    maior valor de amplitude visto (util pra normalizacao 'global')."""
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


def locate_patch_absolute(mask, n_peaks_map, args):
    """Mesma busca de sempre (recorta sub-volume centrado no centroide da
    mascara, desliza janela --patch-size procurando o melhor cruzamento),
    mas devolve o patch em COORDENADAS ABSOLUTAS do volume inteiro (idx3,
    uma tupla de 3 slices indexavel direto em qualquer array com o MESMO
    shape) em vez de um array ja recortado -- isso permite reaproveitar a
    MESMA localizacao em outro protocolo/shell (2026-09-02, --fixed-patch-
    reference), desde que os dois tenham exatamente o mesmo shape espacial
    (mesmo grid/afim -- ver aviso em main() sobre isso NAO valer sem
    registro previo se os shapes diferirem). Retorna (idx3, crossing_frac)
    ou None nas mesmas condicoes de falha de antes (mascara vazia,
    sub-volume menor que --patch-size, ou nenhum patch atingindo
    --min-mask-frac)."""
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return None
    centroid = tuple(int(round(c)) for c in np.argwhere(mask_bool).mean(axis=0))
    slices, origin = bounding_box(mask_bool, centroid, args.search_radius)
    sub_mask = mask_bool[slices]
    sub_n_peaks = n_peaks_map[slices]

    found = find_best_crossing_patch(
        sub_n_peaks, sub_mask, args.patch_size, args.slice_axis,
        min_peaks_for_crossing=args.min_peaks_for_crossing,
        min_mask_frac=args.min_mask_frac)
    if found is None:
        return None
    (o0, o1), slice_idx, crossing_frac = found

    idx3 = [None, None, None]
    out_axes = [a for a in range(3) if a != args.slice_axis]
    idx3[out_axes[0]] = slice(origin[out_axes[0]] + o0, origin[out_axes[0]] + o0 + args.patch_size)
    idx3[out_axes[1]] = slice(origin[out_axes[1]] + o1, origin[out_axes[1]] + o1 + args.patch_size)
    idx3[args.slice_axis] = slice(origin[args.slice_axis] + slice_idx,
                                   origin[args.slice_axis] + slice_idx + 1)
    return tuple(idx3), crossing_frac


def glyph_patch_for_protocol_shell(mask, n_peaks_map, shm_coeff, args, fixed_idx3=None):
    """Recorta o patch de glifo pra um (protocolo, shell). Comportamento
    default (fixed_idx3=None, de sempre): busca o MELHOR cruzamento
    proprio deste protocolo (via locate_patch_absolute). NOVO
    (2026-09-02): se `fixed_idx3` for passado (uma tupla de 3 slices em
    coordenadas absolutas, tipicamente vinda de um protocolo de
    referencia via --fixed-patch-reference), usa ESSE patch direto, sem
    busca nenhuma -- garante o MESMO voxel fisico entre paineis, desde
    que este array (`mask`/`n_peaks_map`/`shm_coeff`) tenha o MESMO shape
    espacial de onde `fixed_idx3` foi calculado (responsabilidade do
    chamador conferir isso ANTES de chamar com fixed_idx3 -- ver main()).
    `crossing_frac` nesse modo reflete a fracao de cruzamento DESTE
    protocolo especificamente NESSE patch (pode ser bem diferente do
    melhor patch que ele teria escolhido sozinho -- e' esperado, e' o que
    se quer medir). Retorna (shm_patch, crossing_frac) ou None (mascara
    vazia/sub-volume pequeno demais/nenhum patch valido, so no modo
    default; no modo fixed_idx3 so retorna None se n_masked==0 no patch
    fixo, caso patologico de mascara vazia bem naquele local)."""
    if fixed_idx3 is not None:
        mask_bool = mask.astype(bool)
        patch_mask = mask_bool[fixed_idx3]
        n_masked = int(patch_mask.sum())
        if n_masked == 0:
            return None
        crossing_bool = mask_bool & (n_peaks_map >= args.min_peaks_for_crossing)
        patch_crossing = crossing_bool[fixed_idx3]
        crossing_frac = float(patch_crossing.sum()) / n_masked
        shm_patch = shm_coeff[fixed_idx3].reshape(args.patch_size, args.patch_size, -1)
        return shm_patch, crossing_frac

    located = locate_patch_absolute(mask, n_peaks_map, args)
    if located is None:
        return None
    idx3, crossing_frac = located
    shm_patch = shm_coeff[idx3].reshape(args.patch_size, args.patch_size, -1)
    return shm_patch, crossing_frac


def process_protocol_shell(entry, shell_key, shell_idx, b0_idx, bvals, bvecs, data,
                            mask, sh_order, args, roi_tracts):
    """Ajusta CSD pra UM (protocolo, shell) e devolve (rows, n_peaks_map,
    shm_coeff): `rows` e' uma lista de linhas (uma por ROI, sempre
    incluindo 'whole_mask') com n_peaks/energia SH; `n_peaks_map`/
    `shm_coeff` sao devolvidos pra reaproveitar no glifo (--make-glyphs),
    sem precisar refazer o ajuste de CSD."""
    idx = np.concatenate([b0_idx, shell_idx])
    idx.sort()
    vol = data[..., idx]
    bvals_sub = bvals[idx]
    bvecs_sub = bvecs[idx]

    n_peaks_map, shm_coeff = fit_csd_peaks(
        vol, bvals_sub, bvecs_sub, mask, sh_order, args.npeaks,
        args.relative_peak_threshold, args.min_separation_angle)

    brain_mask = mask.astype(bool)
    rois = {"whole_mask": brain_mask}
    if roi_tracts:
        tract_masks = load_roi_masks(entry["dwi_path"], roi_tracts, base_mask=brain_mask)
        rois.update(tract_masks)

    rows = []
    for roi_name, roi_bool in rois.items():
        roi_mask = brain_mask & roi_bool
        n_voxels = int(roi_mask.sum())
        if n_voxels == 0:
            continue
        peaks_in_roi = n_peaks_map[roi_mask]
        energy_by_l = sh_energy_by_order(shm_coeff, sh_order, roi_mask)
        energy_frac_high_order = float(sum(
            frac for l, (_e, frac) in energy_by_l.items() if l >= 4)) if energy_by_l else float("nan")
        rows.append({
            "protocol": entry["subject"], "roi": roi_name,
            "fit_type": "single_shell", "shells_used": str(int(shell_key)),
            "b_value": shell_key, "n_dirs": int(len(shell_idx)), "n_b0": int(len(b0_idx)),
            "sh_order": sh_order, "n_voxels": n_voxels,
            "frac_0peaks": float(np.mean(peaks_in_roi == 0)),
            "frac_1peak": float(np.mean(peaks_in_roi == 1)),
            "frac_crossing": float(np.mean(peaks_in_roi >= args.min_peaks_for_crossing)),
            "mean_n_peaks": float(np.mean(np.clip(peaks_in_roi, 0, None))),
            "energy_frac_high_order": energy_frac_high_order,
        })
    return rows, n_peaks_map, shm_coeff


def fit_msmt_csd_peaks(vol, bvals_sub, bvecs_sub, mask, sh_order, npeaks,
                        relative_peak_threshold, min_separation_angle, shell_tol,
                        wm_fa_thr, gm_fa_thr, csf_fa_thr, gm_md_thr, csf_md_thr):
    """MSMT-CSD (multi-shell multi-tecido, Jeurissen et al. NeuroImage 2014)
    -- PRECISA de >=2 shells nao-zero (alem do b0). Estima resposta de
    tecido (materia branca/cinzenta/liquor) automaticamente via limiares
    de FA/MD sobre um fit de tensor interno (`mask_for_response_msmt`/
    `response_from_mask_msmt`, mesmos nomes/defaults do tutorial oficial
    do DIPY), depois ajusta `MultiShellDeconvModel` e extrai picos/
    coeficientes SH da FOD de materia branca via `peaks_from_model(...,
    sh_basis_type='descoteaux07', return_sh=True)` -- a MESMA chamada
    generica usada em `fit_csd_peaks` pro CSD single-shell. Importante:
    `peaks_from_model` reprojeta `fit.odf(sphere)` na base/ordem pedida
    (via `sf_to_sh` internamente) INDEPENDENTE da convencao SH interna do
    modelo -- por isso devolve exatamente o mesmo formato (n_peaks_map,
    shm_coeff), diretamente comparavel ao CSD single-shell, sem precisar
    entender a convencao de coeficientes interna do `MultiShellDeconvModel`
    (que mistura fracoes de volume isotropicas de GM/CSF com a FOD SH de
    WM -- so' a FOD de WM reprojetada e' o que sai daqui).

    NAO EXECUTADO/VERIFICADO neste ambiente de desenvolvimento (sem dipy)
    -- ver aviso no docstring do modulo."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.mcsd import (mask_for_response_msmt, response_from_mask_msmt,
                                    multi_shell_fiber_response, MultiShellDeconvModel)
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere

    gtab = gradient_table(bvals_sub, bvecs_sub)
    mask_wm, mask_gm, mask_csf = mask_for_response_msmt(
        gtab, vol, roi_radii=10, wm_fa_thr=wm_fa_thr, gm_fa_thr=gm_fa_thr,
        csf_fa_thr=csf_fa_thr, gm_md_thr=gm_md_thr, csf_md_thr=csf_md_thr)
    response_wm, response_gm, response_csf = response_from_mask_msmt(
        gtab, vol, mask_wm, mask_gm, mask_csf)
    # bvals nominais por shell (nao o array bruto por volume, que tem ruido
    # de medida) -- mesma fonte/convencao (split_shells) ja usada no resto
    # do arquivo pra "qual e' o b-value canonico desta shell".
    shell_bvals = sorted(split_shells(bvals_sub, tol=shell_tol).keys())
    response_mcsd = multi_shell_fiber_response(
        sh_order=sh_order, bvals=shell_bvals,
        wm_rf=response_wm, gm_rf=response_gm, csf_rf=response_csf)
    mcsd_model = MultiShellDeconvModel(gtab, response_mcsd, sh_order=sh_order)

    sphere = get_sphere("repulsion724")
    peaks = peaks_from_model(
        model=mcsd_model, data=vol, sphere=sphere, mask=mask,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle, npeaks=npeaks,
        parallel=False, normalize_peaks=False,
        return_sh=True, sh_order=sh_order, sh_basis_type="descoteaux07",
    )
    n_peaks_map = (peaks.peak_values > 0).sum(axis=-1).astype(np.int32)
    n_peaks_map[~mask.astype(bool)] = -1
    return n_peaks_map, peaks.shm_coeff


def process_protocol_msmt(entry, bvals, bvecs, data, mask, sh_order, args, roi_tracts):
    """Ajusta MSMT-CSD usando TODAS as shells nao-zero do protocolo DE UMA
    VEZ (ao contrario de process_protocol_shell, que ajusta uma shell por
    vez) -- mesmo formato de retorno (rows, n_peaks_map, shm_coeff) que
    process_protocol_shell, com `fit_type='multishell_msmt'`,
    `b_value=NaN` e `shells_used` listando as shells combinadas (ex.
    '700+2000')."""
    shells = split_shells(bvals, tol=args.shell_tol)
    b0_idx = shells.get(0, np.array([], dtype=int))
    nonzero_keys = sorted(k for k in shells if k != 0)
    idx = np.concatenate([b0_idx] + [np.asarray(shells[k], dtype=int) for k in nonzero_keys])
    idx.sort()
    vol = data[..., idx]
    bvals_sub = bvals[idx]
    bvecs_sub = bvecs[idx]

    n_peaks_map, shm_coeff = fit_msmt_csd_peaks(
        vol, bvals_sub, bvecs_sub, mask, sh_order, args.npeaks,
        args.relative_peak_threshold, args.min_separation_angle, args.shell_tol,
        args.msmt_wm_fa_thr, args.msmt_gm_fa_thr, args.msmt_csf_fa_thr,
        args.msmt_gm_md_thr, args.msmt_csf_md_thr)

    brain_mask = mask.astype(bool)
    rois = {"whole_mask": brain_mask}
    if roi_tracts:
        tract_masks = load_roi_masks(entry["dwi_path"], roi_tracts, base_mask=brain_mask)
        rois.update(tract_masks)

    n_dirs_total = int(sum(len(shells[k]) for k in nonzero_keys))
    shells_str = "+".join(str(int(k)) for k in nonzero_keys)

    rows = []
    for roi_name, roi_bool in rois.items():
        roi_mask = brain_mask & roi_bool
        n_voxels = int(roi_mask.sum())
        if n_voxels == 0:
            continue
        peaks_in_roi = n_peaks_map[roi_mask]
        energy_by_l = sh_energy_by_order(shm_coeff, sh_order, roi_mask)
        energy_frac_high_order = float(sum(
            frac for l, (_e, frac) in energy_by_l.items() if l >= 4)) if energy_by_l else float("nan")
        rows.append({
            "protocol": entry["subject"], "roi": roi_name,
            "fit_type": "multishell_msmt", "shells_used": shells_str,
            "b_value": float("nan"), "n_dirs": n_dirs_total, "n_b0": int(len(b0_idx)),
            "sh_order": sh_order, "n_voxels": n_voxels,
            "frac_0peaks": float(np.mean(peaks_in_roi == 0)),
            "frac_1peak": float(np.mean(peaks_in_roi == 1)),
            "frac_crossing": float(np.mean(peaks_in_roi >= args.min_peaks_for_crossing)),
            "mean_n_peaks": float(np.mean(np.clip(peaks_in_roi, 0, None))),
            "energy_frac_high_order": energy_frac_high_order,
        })
    return rows, n_peaks_map, shm_coeff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                     help="pasta com uma subpasta por protocolo (ex.: folder_main/SeqA1/...)")
    ap.add_argument("--name-suffix", default="_geomcorr",
                     help="mesmo sentido de scripts/01_prepare_data.py --name-suffix")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--protocols", default=None,
                     help="lista separada por virgula de nomes de protocolo (nome da subpasta) "
                          "pra restringir -- default processa todos os encontrados")
    ap.add_argument("--sh-order", type=int, default=None,
                     help="forca a MESMA ordem SH em todo mundo (pra isolar o efeito do "
                          "b-value/N sem deixar a ordem tambem mudar junto) -- default (None) "
                          "usa a ordem auto max_order_for_n_directions(n_dirs) de CADA shell, "
                          "mesma convencao do baseline_sh/pipeline principal")
    ap.add_argument("--min-peaks-for-crossing", type=int, default=2,
                     help="mesmo sentido de scripts/11_peak_confusion_by_roi.py")
    ap.add_argument("--npeaks", type=int, default=3)
    ap.add_argument("--relative-peak-threshold", type=float, default=0.5)
    ap.add_argument("--min-separation-angle", type=float, default=25.0)
    ap.add_argument("--roi-tracts", default=None,
                     help=f"lista separada por virgula de tratos JHU (opcoes: "
                          f"{','.join(JHU_TRACT_LABELS)}) -- default so 'whole_mask'")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--msmt", action="store_true",
                     help="(2026-09-02) TAMBEM ajusta MSMT-CSD (multi-shell multi-tecido) "
                          "usando TODAS as shells nao-zero de uma vez, pra protocolos com >=2 "
                          "shells (SeqB/SeqC/SeqD no exemplo da usuaria) -- uma linha extra por "
                          "ROI com fit_type='multishell_msmt', ANCORA adicional alem das "
                          "comparacoes single-shell (ver docstring do modulo). Mais lento e "
                          "menos testado que o CSD single-shell do resto da pipeline.")
    ap.add_argument("--msmt-wm-fa-thr", type=float, default=0.7,
                     help="limiar de FA pra mascara de materia branca em mask_for_response_msmt "
                          "(default = mesmo default do DIPY; so' usado com --msmt)")
    ap.add_argument("--msmt-gm-fa-thr", type=float, default=0.3,
                     help="limiar de FA pra mascara de materia cinzenta (so' usado com --msmt)")
    ap.add_argument("--msmt-csf-fa-thr", type=float, default=0.15,
                     help="limiar de FA pra mascara de liquor (so' usado com --msmt)")
    ap.add_argument("--msmt-gm-md-thr", type=float, default=0.001,
                     help="limiar de MD pra mascara de materia cinzenta (so' usado com --msmt)")
    ap.add_argument("--msmt-csf-md-thr", type=float, default=0.0032,
                     help="limiar de MD pra mascara de liquor (so' usado com --msmt)")
    ap.add_argument("--make-glyphs", action="store_true",
                     help="(2026-09-02) gera tambem uma figura com um painel de glifo FOD por "
                          "(protocolo, shell) -- ver docstring do modulo. Requer --out-fig.")
    ap.add_argument("--out-fig", default=None,
                     help="caminho do PNG da figura de glifos -- so' necessario com "
                          "--make-glyphs")
    ap.add_argument("--search-radius", type=int, default=15,
                     help="mesmo sentido de scripts/12_visualize_fod_glyphs.py -- raio (voxels) "
                          "do sub-volume centrado no centroide da mascara onde o melhor patch "
                          "de cruzamento e' procurado (so' usado com --make-glyphs)")
    ap.add_argument("--patch-size", type=int, default=4,
                     help="lado (voxels) da janela de glifos (so' usado com --make-glyphs)")
    ap.add_argument("--slice-axis", type=int, default=2, choices=[0, 1, 2],
                     help="0=sagital,1=coronal,2=axial (so' usado com --make-glyphs)")
    ap.add_argument("--min-mask-frac", type=float, default=0.5,
                     help="mesmo sentido de scripts/12_visualize_fod_glyphs.py (so' usado com "
                          "--make-glyphs)")
    ap.add_argument("--glyph-scale", type=float, default=0.45)
    ap.add_argument("--glyph-n-angles", type=int, default=72)
    ap.add_argument("--normalize", choices=["global", "per_voxel"], default="per_voxel",
                     help="'per_voxel' (default aqui, diferente de 12_visualize_fod_glyphs.py): "
                          "cada painel normalizado pelo seu proprio pico -- faz mais sentido "
                          "quando NENHUM painel e' um 'ground truth' comum (protocolos com "
                          "b-values/contrastes diferentes nao tem magnitude de FOD diretamente "
                          "comparavel). 'global': todos normalizados pelo pico do PRIMEIRO "
                          "painel -- com --fixed-patch-reference (mesmo voxel fisico em todos "
                          "os paineis) isso passa a ser uma comparacao de magnitude tambem "
                          "valida (ainda sujeita a diferenca de contraste por b-value em si, "
                          "nao so por escolha de voxel).")
    ap.add_argument("--fixed-patch-reference", default=None,
                     help="(2026-09-02) nome do protocolo (subpasta, ex. 'SeqA1') cujo melhor "
                          "patch de cruzamento sera usado como referencia -- o MESMO patch "
                          "(mesmas coordenadas de voxel) e' entao reaproveitado em TODOS os "
                          "outros paineis, em vez de cada protocolo buscar seu proprio melhor "
                          "patch (default, sem esta flag). SO FAZ SENTIDO SE os protocolos "
                          "estiverem no MESMO grid/afim (mesmo shape espacial) -- o script "
                          "confere isso e PULA (com aviso) o painel de qualquer protocolo cujo "
                          "shape difira do de referencia, em vez de recortar um voxel errado "
                          "silenciosamente (comparacao voxel-a-voxel sem registro previo nao e' "
                          "valida, ver docstring do modulo). Sem esta flag, comportamento "
                          "identico ao de sempre (cada painel busca seu proprio melhor patch).")
    ap.add_argument("--fixed-patch-shell", type=float, default=None,
                     help="so' usado com --fixed-patch-reference: qual shell (b-value nominal) "
                          "do protocolo de referencia usar pra localizar o patch, se ele tiver "
                          "mais de uma shell nao-zero -- default (None) usa a MENOR shell "
                          "nao-zero processada desse protocolo.")
    args = ap.parse_args()

    if args.make_glyphs and args.out_fig is None:
        sys.exit("--make-glyphs precisa de --out-fig")
    if args.fixed_patch_reference and not args.make_glyphs:
        sys.exit("--fixed-patch-reference so' faz sentido junto com --make-glyphs")
    if args.fixed_patch_shell is not None and not args.fixed_patch_reference:
        sys.exit("--fixed-patch-shell so' faz sentido junto com --fixed-patch-reference")

    roi_tracts = [t.strip() for t in args.roi_tracts.split(",")] if args.roi_tracts else None
    glyph_panels = []  # (label, shm_patch, sh_order, crossing_frac)

    found = discover_dwi_files(args.data_root, name_suffix=args.name_suffix)
    if not found:
        sys.exit(f"Nenhum trio nii+bval+bvec terminando em {args.name_suffix!r} encontrado em "
                  f"{args.data_root}")
    if args.protocols:
        wanted = {p.strip() for p in args.protocols.split(",")}
        found = [e for e in found if e["subject"] in wanted]
        missing = wanted - {e["subject"] for e in found}
        if missing:
            print(f"[aviso] protocolo(s) pedido(s) e nao encontrado(s): {sorted(missing)}", flush=True)
    print(f"{len(found)} protocolo(s) encontrado(s): {[e['subject'] for e in found]}", flush=True)

    # (2026-09-02) --fixed-patch-reference: localiza o patch UMA VEZ no
    # protocolo de referencia, ANTES do loop principal, pra reaproveitar a
    # mesma localizacao (coordenadas absolutas) em todos os paineis do
    # loop abaixo. So' funciona pra protocolos com o MESMO shape espacial
    # do de referencia -- checado por protocolo dentro do loop principal
    # (nao aqui), com aviso+pulo do painel se divergir (comparacao
    # voxel-a-voxel sem registro previo nao e' valida, ver docstring do
    # modulo).
    fixed_idx3 = None
    fixed_shape = None
    if args.fixed_patch_reference:
        ref_entry = next((e for e in found if e["subject"] == args.fixed_patch_reference), None)
        if ref_entry is None:
            sys.exit(f"--fixed-patch-reference={args.fixed_patch_reference!r} nao encontrado "
                      f"entre os protocolos descobertos: {[e['subject'] for e in found]}")
        bvals_r, bvecs_r = load_bval_bvec(str(ref_entry["bval_path"]), str(ref_entry["bvec_path"]))
        data_r, _affine_r, _header_r = load_dwi(str(ref_entry["dwi_path"]))
        fixed_shape = data_r.shape[:3]
        shells_r = split_shells(bvals_r, tol=args.shell_tol)
        b0_idx_r = shells_r.get(0, np.array([], dtype=int))
        if b0_idx_r.size == 0:
            sys.exit(f"--fixed-patch-reference={args.fixed_patch_reference!r} nao tem volume b0")
        shell_keys_r = sorted(k for k in shells_r if k != 0)
        if not shell_keys_r:
            sys.exit(f"--fixed-patch-reference={args.fixed_patch_reference!r} nao tem shell "
                      f"nao-zero nenhuma")
        if args.fixed_patch_shell is not None:
            chosen_shell_r = min(shell_keys_r, key=lambda k: abs(k - args.fixed_patch_shell))
            if abs(chosen_shell_r - args.fixed_patch_shell) > args.shell_tol:
                sys.exit(f"--fixed-patch-shell={args.fixed_patch_shell} nao bate (tol="
                          f"{args.shell_tol}) com nenhuma shell de "
                          f"{args.fixed_patch_reference!r}: {shell_keys_r}")
        else:
            chosen_shell_r = shell_keys_r[0]
        shell_idx_r = np.asarray(shells_r[chosen_shell_r], dtype=int)
        b0_mean_r = data_r[..., b0_idx_r].mean(axis=-1)
        mask_r = load_or_build_mask(str(ref_entry["dwi_path"]), b0_mean_r,
                                     mask_suffix=args.mask_suffix)
        sh_order_r = args.sh_order or max_order_for_n_directions(len(shell_idx_r))
        idx_r = np.concatenate([b0_idx_r, shell_idx_r])
        idx_r.sort()
        n_peaks_map_r, _shm_coeff_r = fit_csd_peaks(
            data_r[..., idx_r], bvals_r[idx_r], bvecs_r[idx_r], mask_r, sh_order_r,
            args.npeaks, args.relative_peak_threshold, args.min_separation_angle)
        located_r = locate_patch_absolute(mask_r, n_peaks_map_r, args)
        if located_r is None:
            sys.exit(f"nenhum patch de cruzamento encontrado em "
                      f"{args.fixed_patch_reference!r} (b={chosen_shell_r:.0f}) -- tente "
                      f"--search-radius maior ou --min-mask-frac menor")
        fixed_idx3, _crossing_frac_r = located_r
        print(f"--fixed-patch-reference={args.fixed_patch_reference!r} (b={chosen_shell_r:.0f}, "
              f"shape={fixed_shape}): patch fixo localizado em {fixed_idx3} -- reaproveitado em "
              f"todos os protocolos com o MESMO shape espacial", flush=True)

    all_rows = []
    ref_shape = None
    for entry in found:
        tag = entry["subject"]
        try:
            bvals, bvecs = load_bval_bvec(str(entry["bval_path"]), str(entry["bvec_path"]))
            data, affine, _header = load_dwi(str(entry["dwi_path"]))
            if ref_shape is None:
                ref_shape = data.shape[:3]
            elif data.shape[:3] != ref_shape:
                print(f"[aviso] {tag}: shape espacial {data.shape[:3]} difere do primeiro "
                      f"protocolo processado {ref_shape} -- mesmo com 'mesma resolucao' "
                      f"nominal, confira FOV/registro antes de comparar voxel-a-voxel (este "
                      f"script so agrega por mascara/ROI, entao nao e afetado diretamente, "
                      f"mas vale saber).", flush=True)

            shells = split_shells(bvals, tol=args.shell_tol)
            b0_idx = shells.get(0, np.array([], dtype=int))
            if b0_idx.size == 0:
                print(f"[aviso] {tag}: sem volume b0 -- pulando", flush=True)
                continue
            b0_mean = data[..., b0_idx].mean(axis=-1)
            mask = load_or_build_mask(str(entry["dwi_path"]), b0_mean, mask_suffix=args.mask_suffix)

            shell_keys = sorted(k for k in shells if k != 0)
            for shell_key in shell_keys:
                shell_idx = np.asarray(shells[shell_key], dtype=int)
                sh_order = args.sh_order or max_order_for_n_directions(len(shell_idx))
                try:
                    rows, n_peaks_map, shm_coeff = process_protocol_shell(
                        entry, shell_key, shell_idx, b0_idx, bvals, bvecs, data, mask,
                        sh_order, args, roi_tracts)
                    all_rows.extend(rows)
                    print(f"{tag} b={shell_key:.0f} (n_dirs={len(shell_idx)}, "
                          f"sh_order={sh_order}): {len(rows)} linha(s)", flush=True)
                    if args.make_glyphs:
                        skip_shape_mismatch = (fixed_idx3 is not None
                                                and data.shape[:3] != fixed_shape)
                        if skip_shape_mismatch:
                            print(f"[aviso] {tag} b={shell_key:.0f}: shape espacial "
                                  f"{data.shape[:3]} difere do protocolo de referencia "
                                  f"{args.fixed_patch_reference!r} {fixed_shape} -- patch fixo "
                                  f"exige o MESMO grid (sem registro previo, comparacao "
                                  f"voxel-a-voxel nao e' valida) -- pulando este painel",
                                  flush=True)
                            found_glyph = None
                        else:
                            found_glyph = glyph_patch_for_protocol_shell(
                                mask, n_peaks_map, shm_coeff, args, fixed_idx3=fixed_idx3)
                        if found_glyph is None:
                            if not skip_shape_mismatch:
                                print(f"[aviso] {tag} b={shell_key:.0f}: nenhum patch de "
                                      f"cruzamento encontrado pro glifo (tente --search-radius "
                                      f"maior ou --min-mask-frac menor) -- pulando este painel",
                                      flush=True)
                        else:
                            shm_patch, crossing_frac = found_glyph
                            label = f"{tag}\nb{shell_key:.0f} n{len(shell_idx)}"
                            glyph_panels.append((label, shm_patch, sh_order, crossing_frac))
                except Exception:
                    print(f"[erro] {tag} b={shell_key:.0f}: CSD falhou -- pulando esta "
                          f"combinacao e continuando. Traceback completo abaixo:", flush=True)
                    traceback.print_exc()

            if args.msmt:
                if len(shell_keys) < 2:
                    print(f"[aviso] {tag}: so' {len(shell_keys)} shell(s) nao-zero -- "
                          f"--msmt precisa de >=2, pulando MSMT-CSD pra este protocolo",
                          flush=True)
                else:
                    n_dirs_total_msmt = int(sum(len(shells[k]) for k in shell_keys))
                    sh_order_msmt = args.sh_order or max_order_for_n_directions(n_dirs_total_msmt)
                    shells_str = "+".join(str(int(k)) for k in shell_keys)
                    try:
                        rows_msmt, n_peaks_map_msmt, shm_coeff_msmt = process_protocol_msmt(
                            entry, bvals, bvecs, data, mask, sh_order_msmt, args, roi_tracts)
                        all_rows.extend(rows_msmt)
                        print(f"{tag} MSMT-CSD (shells={shells_str}, n_dirs={n_dirs_total_msmt}, "
                              f"sh_order={sh_order_msmt}): {len(rows_msmt)} linha(s)", flush=True)
                        if args.make_glyphs:
                            skip_shape_mismatch = (fixed_idx3 is not None
                                                    and data.shape[:3] != fixed_shape)
                            if skip_shape_mismatch:
                                print(f"[aviso] {tag} MSMT-CSD: shape espacial "
                                      f"{data.shape[:3]} difere do protocolo de referencia "
                                      f"{args.fixed_patch_reference!r} {fixed_shape} -- "
                                      f"pulando este painel", flush=True)
                                found_glyph = None
                            else:
                                found_glyph = glyph_patch_for_protocol_shell(
                                    mask, n_peaks_map_msmt, shm_coeff_msmt, args,
                                    fixed_idx3=fixed_idx3)
                            if found_glyph is None:
                                if not skip_shape_mismatch:
                                    print(f"[aviso] {tag} MSMT-CSD: nenhum patch de cruzamento "
                                          f"encontrado pro glifo -- pulando este painel",
                                          flush=True)
                            else:
                                shm_patch, crossing_frac = found_glyph
                                label = f"{tag}\nMSMT b{shells_str}"
                                glyph_panels.append(
                                    (label, shm_patch, sh_order_msmt, crossing_frac))
                    except Exception:
                        print(f"[erro] {tag}: MSMT-CSD falhou (shells={shells_str}) -- pulando "
                              f"e continuando. Traceback completo abaixo (confira a versao do "
                              f"dipy instalada contra dipy.reconst.mcsd se a excecao for de "
                              f"assinatura de funcao -- ver aviso no docstring do modulo):",
                              flush=True)
                        traceback.print_exc()
        except Exception:
            print(f"[erro] falha processando protocolo {tag} -- pulando e continuando. "
                  f"Traceback completo abaixo:", flush=True)
            traceback.print_exc()

    if not all_rows:
        sys.exit("Nenhum resultado -- confira --data-root/--name-suffix/--protocols.")

    df = pd.DataFrame(all_rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("\nMetricas salvas em", out_csv)

    cols = ["protocol", "roi", "fit_type", "shells_used", "b_value", "n_dirs", "n_b0",
            "sh_order", "n_voxels", "frac_0peaks", "frac_1peak", "frac_crossing",
            "mean_n_peaks", "energy_frac_high_order"]
    summary = df[cols].sort_values(
        ["roi", "fit_type", "b_value", "n_dirs"]).set_index(["roi", "fit_type", "protocol"])
    print("\nResumo (ordenado por ROI, depois fit_type, depois b_value, depois n_dirs):")
    print(summary)

    print("\nLeitura -- tres comparacoes que este script foi desenhado pra habilitar "
          "(ver docstring do modulo pro racional completo):\n"
          "1) REPETIBILIDADE (mesmo b, mesmo N, so o numero de b0 muda -- ex. protocolos "
          "com o mesmo b_value/n_dirs nesta tabela): quanto 'frac_crossing'/"
          "'energy_frac_high_order' variam entre eles e' o PISO DE RUIDO do proprio fit de "
          "CSD -- qualquer diferenca entre b-values MENOR que isso nao e' um efeito real de "
          "b-value, e' so ruido de reajuste.\n"
          "2) SENSIBILIDADE A N (mesmo b_value, n_dirs diferente): se 'frac_crossing'/"
          "'energy_frac_high_order' caem bastante com menos direcoes mesmo no MESMO b, isso "
          "e' limitacao de amostragem angular, nao de contraste de difusao.\n"
          "3) SENSIBILIDADE AO B-VALUE (n_dirs parecido, b_value diferente): se "
          "'frac_crossing'/'energy_frac_high_order' sobem consistentemente com b_value maior "
          "(alem do piso de ruido medido em 1), isso confirma que o GT em b1000 (usado hoje "
          "como referencia em toda a pipeline) estruturalmente sub-representa cruzamento "
          "comparado a um b mais alto -- um teto real na precisao de qualquer metodo avaliado "
          "contra esse GT, nao um defeito da reconstrucao em si.")

    if args.msmt and (df["fit_type"] == "multishell_msmt").any():
        print("\n4) MSMT-CSD como ANCORA (linhas com fit_type='multishell_msmt', "
              "b_value=NaN, shells_used lista as shells combinadas): compare "
              "'frac_crossing'/'energy_frac_high_order' do MSMT-CSD do protocolo mais rico "
              "(ex. SeqB/SeqC) contra o b1000/64dir single_shell (SeqA1/A5/A6) NO MESMO "
              "'roi' -- a diferenca entre os dois e' uma estimativa direta de quanto o GT "
              "single-shell que a pipeline usa hoje fica abaixo do 'melhor palpite possivel' "
              "pra esse sujeito. Isso NAO substitui a comparacao 3 acima (sensibilidade ao "
              "b-value, sempre single-shell) -- e' um teto superior adicional, nao outro ponto "
              "da mesma escala.")

    if args.make_glyphs:
        if len(glyph_panels) < 1:
            print("\n[aviso] --make-glyphs pedido mas nenhum painel de glifo foi gerado com "
                  "sucesso (ver avisos acima) -- figura NAO salva.", flush=True)
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        directions = in_plane_directions(args.slice_axis, n_angles=args.glyph_n_angles)
        fig, axes = plt.subplots(1, len(glyph_panels), figsize=(3.0 * len(glyph_panels), 3.6))
        if len(glyph_panels) == 1:
            axes = [axes]

        amplitude_ref = None
        for i, (ax, (label, shm_patch, panel_sh_order, crossing_frac)) in enumerate(
                zip(axes, glyph_panels)):
            ref = amplitude_ref if args.normalize == "global" else None
            peak = render_glyph_field(ax, shm_patch, directions, panel_sh_order,
                                       args.glyph_scale, amplitude_ref=ref)
            if i == 0 and args.normalize == "global":
                amplitude_ref = peak
            ax.set_title(f"{label}\n({crossing_frac:.0%} cruzamento)", fontsize=8)

        if args.fixed_patch_reference:
            title = (f"Glifos FOD por protocolo/shell -- MESMO patch fisico em todos "
                     f"(referencia: {args.fixed_patch_reference}, protocolos com shape "
                     f"diferente foram pulados) -- normalizacao={args.normalize}")
        else:
            title = (f"Glifos FOD por protocolo/shell -- melhor patch de cruzamento de CADA "
                     f"um (nao necessariamente o mesmo voxel fisico entre paineis) -- "
                     f"normalizacao={args.normalize}")
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_fig = Path(args.out_fig)
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=150)
        print(f"\nFigura de glifos salva em {out_fig}", flush=True)


if __name__ == "__main__":
    main()