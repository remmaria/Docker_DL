#!/usr/bin/env python3
"""
Etapa 12 (diagnostico visual): glifos do FOD (CSD) lado a lado -- ground
truth vs. cada metodo de reconstrucao -- numa regiao de cruzamento de
fibras genuino, escolhida AUTOMATICAMENTE (a pedido da usuaria, 2026-08-31:
"queria tb gerar o glifo, mostrando os formatinhos.. pra mostrar a
diferenca?").

O QUE E' DESENHADO: cada glifo e' o perfil 2D do FOD -- amplitude(theta)
amostrada em 72 direcoes DENTRO DO PLANO da fatia escolhida (plano axial
por padrao, --slice-axis controla), plotado como um poligono fechado
(a "rosinha"/"borboleta" classica de figuras de cruzamento em papers de
dMRI). NAO e' um glifo 3D completo (que precisaria renderizar off-screen
com VTK/fury -- dependencia pesada e nao confirmada disponivel no
login/compute node do cluster; mesma cautela ja registrada no docstring de
scripts/07_visualize_triplet.py sobre depender so do que ja se sabe
disponivel). LIMITACAO INERENTE dessa projecao 2D: um lobulo do FOD que
aponta bem PRA FORA do plano escolhido aparece menor/achatado no glifo
(projecao), nao desaparece -- a mesma limitacao de qualquer figura 2D de
glifos publicada. Se a regiao de cruzamento escolhida automaticamente
parecer com formato pouco nitido, tente rodar de novo com
--slice-axis diferente (0=sagital, 1=coronal, 2=axial) -- o cruzamento
pode estar mais alinhado com outro plano.

COMO A REGIAO E' ESCOLHIDA: 1) recorta um sub-volume cubico centrado no
centroide da mascara (lado 2*--search-radius+1, default 31 voxels --
grande o suficiente pra auto_response_ssst(roi_radii=10) ter uma ROI
centrada valida nas 3 dimensoes, evitando fitar CSD no cerebro inteiro,
que e' caro -- ver addendum, "CSD por sujeito e' lento"); 2) ajusta CSD do
GROUND TRUTH nesse sub-volume e conta picos por voxel
(peaks_from_model, mesma convencao de scripts/11_peak_confusion_by_roi.py);
3) desliza uma janela --patch-size x --patch-size (default 4x4) sobre cada
fatia do sub-volume ao longo de --slice-axis, escolhendo a posicao com
MAIOR fracao de voxels mascarados com >=2 picos no GT (exige uma fracao
mascarada minima, --min-mask-frac, pra nao escolher um patch mal coberto).

FOCAR NUM VOXEL ESPECIFICO (--center-voxel): a busca automatica acima e'
o default, mas se voce ja rodou o script uma vez e quer focar so num voxel
especifico do patch encontrado (ex.: o de cruzamento mais nitido, pra uma
figura de tese mais legivel com --patch-size menor), passe
--center-voxel "X,Y,Z" em coordenadas GLOBAIS do volume -- mesmo sistema
impresso em "Centroide da mascara: (...)"/"sub-volume: (...)" pela rodada
anterior. Isso PULA a busca automatica de cruzamento inteira e centra o
patch (`centered_patch()`) nesse voxel especifico.

Reusa a MESMA logica de CSD de scripts/11_peak_confusion_by_roi.py
(ConstrainedSphericalDeconvModel/auto_response_ssst/peaks_from_model,
return_sh=True) reproduzida aqui como fit_shm_and_npeaks() -- sem import
cruzado entre scripts de etapas (mesmo padrao ja usado pra
_resolve_shell_key/sh_energy_by_order duplicados em 11_peak_confusion_by_roi.py).
Reusa exatamente a MESMA montagem "full = data.copy(); full[...,
target_idx] = recon" de 11_peak_confusion_by_roi.py:_process_subject pra
avaliar cada metodo de reconstrucao.

--SUBSAMPLED-ONLY (2026-09-02, a pedido da usuaria -- "no script de
visualizar glifos a gente colocou tb o subamostrado?"): ADITIVO, requer
--triplets-dir. Mesma ideia do --subsampled-only de
11_peak_confusion_by_roi.py -- SEM reconstrucao nenhuma, ajusta CSD so nas
direcoes de entrada REAIS (exclui as direcoes-alvo do esquema de
subamostragem via `exclude_idx` em fit_shm_and_npeaks(), mesma semantica
de `exclude_idx` em 11_peak_confusion_by_roi.py:fit_peaks), numa ordem SH
tipicamente MENOR (`max_order_for_n_directions(n_level)`, auto por
default, --sh-order-subsampled-only pra forcar outra). Mostra visualmente
"e' isso que da pra ver sem interpolacao nenhuma" ao lado do ground_truth/
baseline_sh/rcae/etc. na mesma figura.

Uso (mesma convencao --baseline-dir/--rcae-dir/--extra-method de
06_evaluate_reconstruction.py/11_peak_confusion_by_roi.py):
    python scripts/12_visualize_fod_glyphs.py \
        --manifest work_dir/manifest.csv \
        --baseline-dir work_dir/baseline_recon \
        --extra-method "naive_blend=work_dir/naive_blend_recon,rrin_n16_star610=work_dir/rrin_star_recon_rrin_n16_star610,rcae_n16=work_dir/rcae_recon_rcae_n16" \
        --subsampled-only --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --subjects 20170417094841_802780_20170417094841_802780 \
        --out work_dir/figures/fod_glyphs_shell1000_n16.png

Requer DIPY (ConstrainedSphericalDeconvModel/auto_response_ssst/
peaks_from_model/sh_to_sf) + nibabel + matplotlib. Nao executado neste
ambiente de desenvolvimento (sem dipy/nibabel) -- so a logica de selecao
de patch e a geometria do poligono do glifo (funcoes puramente numpy,
sem dipy) foram testadas isoladamente nesta sessao (ver mensagem de
entrega).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, load_dwi, split_shells
from utils.masking import load_or_build_mask
from utils.sh_basis import max_order_for_n_directions


def _resolve_shell_key(shells: dict, shell_b: float, tol: float) -> float:
    """Mesma logica de scripts/11_peak_confusion_by_roi.py -- duplicada de
    proposito (sem import cruzado entre scripts de etapas, mesmo padrao ja
    usado nesse arquivo)."""
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


def bounding_box(mask, center, radius):
    """Sub-volume cubico (2*radius+1)^3 centrado em `center` (voxel
    inteiro), recortado (clip) pra caber dentro de `mask.shape`. Retorna
    (slices, origin) -- `slices` uma tupla de 3 `slice` objects prontos
    pra indexar arrays (X,Y,Z,...), `origin` o canto inferior (x0,y0,z0)
    em coordenadas GLOBAIS (pra traduzir indices do sub-volume de volta
    pro volume inteiro depois)."""
    shape = mask.shape
    lo = [max(0, c - radius) for c in center]
    hi = [min(shape[d], center[d] + radius + 1) for d in range(3)]
    slices = tuple(slice(lo[d], hi[d]) for d in range(3))
    return slices, tuple(lo)


def find_best_crossing_patch(n_peaks_map, mask, patch_size, slice_axis,
                              min_peaks_for_crossing=2, min_mask_frac=0.5):
    """Desliza uma janela quadrada `patch_size x patch_size` sobre cada
    fatia perpendicular a `slice_axis` de `n_peaks_map` (shape (X,Y,Z),
    -1 fora da mascara), escolhendo a posicao com MAIOR fracao de voxels
    mascarados com `n_peaks_map >= min_peaks_for_crossing`, exigindo que
    pelo menos `min_mask_frac` dos voxels do patch estejam dentro da
    mascara (senao um patch quase todo fora do cerebro, com fracao de
    cruzamento calculada so sobre os 1-2 voxels mascarados que sobraram,
    poderia vencer por acidente).

    Retorna (origin_in_out_axes, slice_index, crossing_frac) onde
    `origin_in_out_axes` e' (o0, o1) o canto inferior do patch nos DOIS
    eixos que NAO sao `slice_axis` (na ordem crescente desses eixos), e
    `slice_index` a posicao ao longo de `slice_axis`. Retorna None se
    nenhum patch candidato atingir `min_mask_frac`.
    """
    shape = n_peaks_map.shape
    out_axes = [a for a in range(3) if a != slice_axis]
    mask_bool = mask.astype(bool)
    crossing_bool = mask_bool & (n_peaks_map >= min_peaks_for_crossing)

    best = None  # (frac, origin0, origin1, slice_idx)
    for s in range(shape[slice_axis]):
        idx3 = [slice(None)] * 3
        idx3[slice_axis] = s
        idx3 = tuple(idx3)
        mask_slice = mask_bool[idx3]          # shape (len(out_axes[0]), len(out_axes[1]))
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


def centered_patch(local_center, patch_size, slice_axis, shape):
    """Alternativa a `find_best_crossing_patch` pra quando a usuaria ja quer
    um voxel ESPECIFICO (nao a busca automatica) -- ex.: focar so no voxel
    de cruzamento mais nitido de uma figura anterior. `local_center` e' um
    voxel (x,y,z) em coordenadas LOCAIS do sub-volume (ja subtraida a
    `origin` de `bounding_box`). Centra um patch `patch_size x patch_size`
    nesse voxel nos dois eixos que NAO sao `slice_axis`, recortado (clip)
    pra caber dentro de `shape` (mesmo espirito de clipping de
    `bounding_box`). Retorna (origin_in_out_axes, slice_index) -- mesma
    convencao de saida (parcial, sem crossing_frac) de
    `find_best_crossing_patch`. Retorna None se o voxel central cair fora
    do sub-volume ou se `shape` for menor que `patch_size` em algum dos
    eixos de saida."""
    out_axes = [a for a in range(3) if a != slice_axis]
    if shape[out_axes[0]] < patch_size or shape[out_axes[1]] < patch_size:
        return None
    slice_idx = local_center[slice_axis]
    if slice_idx < 0 or slice_idx >= shape[slice_axis]:
        return None
    origin_out = []
    for ax in out_axes:
        c = local_center[ax]
        lo = c - patch_size // 2
        lo = max(0, min(lo, shape[ax] - patch_size))
        origin_out.append(lo)
    return tuple(origin_out), slice_idx


def fit_shm_and_npeaks(data, bvals, bvecs, shell_b, mask, shell_tol, sh_order,
                        relative_peak_threshold, min_separation_angle, npeaks,
                        exclude_idx=None):
    """CSD single-shell single-tissue (Tournier07), mesma convencao de
    scripts/11_peak_confusion_by_roi.py:fit_peaks -- devolve so
    (n_peaks_map, shm_coeff), sem peak_dirs/peak_values (nao usados aqui,
    os glifos sao desenhados a partir do shm_coeff diretamente, nao dos
    picos discretos). `exclude_idx` (ADITIVO, default None, 2026-09-02):
    remove esses indices do conjunto de direcoes usadas no ajuste -- mesma
    semantica de `exclude_idx` em 11_peak_confusion_by_roi.py:fit_peaks,
    usado pelo modo --subsampled-only (ver main())."""
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

    response, ratio = auto_response_ssst(gtab, vol, roi_radii=10, fa_thr=0.7)
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


def in_plane_directions(slice_axis, n_angles=72):
    """`n_angles` vetores unitarios DENTRO do plano perpendicular a
    `slice_axis` (0=X/sagital, 1=Y/coronal, 2=Z/axial), espacados
    uniformemente em angulo -- usados como as direcoes de amostragem do
    FOD pro glifo 2D (ver docstring do modulo pra a limitacao dessa
    projecao)."""
    theta = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    zeros = np.zeros_like(theta)
    if slice_axis == 2:      # axial (plano XY)
        dirs = np.stack([cos_t, sin_t, zeros], axis=-1)
    elif slice_axis == 1:    # coronal (plano XZ)
        dirs = np.stack([cos_t, zeros, sin_t], axis=-1)
    else:                     # sagital (plano YZ)
        dirs = np.stack([zeros, cos_t, sin_t], axis=-1)
    return dirs.astype(np.float64)


def glyph_polygon_xy(amplitudes, center_xy, glyph_scale, clip_negative=True):
    """Converte um perfil de amplitude(theta) (shape (n_angles,), theta
    igualmente espacado 0..2pi) num poligono 2D fechado centrado em
    `center_xy`: ponto k = center + glyph_scale*amplitude[k]*(cos(theta_k),
    sin(theta_k)). `clip_negative=True` zera amplitudes negativas (CSD com
    restricao de nao-negatividade produz FOD>=0 na pratica, mas erro
    numerico residual pode gerar valores levemente negativos -- sem clip,
    o poligono cruzaria o proprio centro de forma visualmente enganosa).
    Retorna array (n_angles, 2)."""
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
    """Desenha um campo de glifos 2D pra um patch (shape (P,Q,n_coef) de
    coeficientes SH) num eixo matplotlib 2D ja criado. `directions` vem de
    in_plane_directions(). `amplitude_ref` (escalar, opcional): se dado,
    normaliza TODOS os glifos por esse valor global (em vez de cada um
    pelo seu proprio maximo) -- usado pra manter a escala comparavel entre
    ground_truth e os metodos de reconstrucao na mesma figura (ver
    --normalize no main()). Retorna o maior valor de amplitude visto
    (util pra escolher `amplitude_ref` a partir do ground_truth antes de
    desenhar os demais)."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--rcae-dir", default=None)
    ap.add_argument("--extra-method", action="append", default=[],
                     help="'nome=caminho', repetivel ou 'a=x,b=y' -- mesma convencao de "
                          "06_evaluate_reconstruction.py/11_peak_confusion_by_roi.py")
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subjects", default=None,
                     help="tag(s) separadas por virgula -- se omitido, usa o primeiro "
                          "sujeito do split com dados disponiveis")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--sh-order", type=int, default=None,
                     help="default: auto via max_order_for_n_directions do total de "
                          "direcoes da shell (mesma convencao de 11_peak_confusion_by_roi.py)")
    ap.add_argument("--relative-peak-threshold", type=float, default=0.5)
    ap.add_argument("--min-separation-angle", type=float, default=25.0)
    ap.add_argument("--npeaks", type=int, default=5)
    ap.add_argument("--search-radius", type=int, default=15,
                     help="raio (voxels) do sub-volume cubico centrado no centroide da "
                          "mascara onde o CSD e' ajustado -- precisa ser >= ~12 pra "
                          "auto_response_ssst(roi_radii=10) ter uma ROI centrada valida")
    ap.add_argument("--patch-size", type=int, default=4,
                     help="lado (voxels) da janela de glifos desenhada (patch_size x patch_size)")
    ap.add_argument("--slice-axis", type=int, default=2, choices=[0, 1, 2],
                     help="0=sagital,1=coronal,2=axial -- tanto o plano de busca do "
                          "cruzamento quanto o plano de amostragem do glifo 2D")
    ap.add_argument("--min-peaks-for-crossing", type=int, default=2)
    ap.add_argument("--min-mask-frac", type=float, default=0.5)
    ap.add_argument("--subsampled-only", action="store_true",
                     help="(2026-09-02) tambem desenha um painel 'subsampled_only' -- CSD "
                          "ajustado SO nas direcoes de entrada reais (sem nenhuma "
                          "reconstrucao/preenchimento), mesma logica de --subsampled-only em "
                          "11_peak_confusion_by_roi.py. Requer --triplets-dir.")
    ap.add_argument("--triplets-dir", default=None,
                     help="pasta com '<tag>_rrin_triplets.npz' (ex.: work_dir/subsampling) -- "
                          "so' necessario com --subsampled-only, pra saber quais direcoes sao "
                          "'alvo' (excluidas do ajuste) pra este shell/n_level")
    ap.add_argument("--sh-order-subsampled-only", type=int, default=None,
                     help="ordem SH do CSD so' para o painel --subsampled-only (default: auto "
                          "via max_order_for_n_directions(n_level) -- tipicamente MENOR que "
                          "--sh-order, ja que ha menos direcoes reais disponiveis)")
    ap.add_argument("--center-voxel", default=None,
                     help="'X,Y,Z' em coordenadas GLOBAIS (mesmo sistema impresso em "
                          "'Centroide da mascara'/sub-volume por uma rodada anterior) -- "
                          "quando dado, IGNORA a busca automatica de cruzamento e centra o "
                          "patch de glifos nesse voxel especifico, pra focar num voxel ja "
                          "identificado antes (ex.: o voxel de cruzamento mais nitido de uma "
                          "figura com --patch-size maior).")
    ap.add_argument("--glyph-scale", type=float, default=0.45)
    ap.add_argument("--glyph-n-angles", type=int, default=72)
    ap.add_argument("--normalize", choices=["global", "per_voxel"], default="global",
                     help="'global' (default): todos os glifos (GT e metodos) normalizados "
                          "pelo mesmo pico do GT, preservando diferenca de magnitude entre "
                          "metodos -- recomendado pra comparacao. 'per_voxel': cada glifo "
                          "normalizado pelo seu proprio pico (so mostra FORMA, esconde "
                          "diferenca de magnitude).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.subsampled_only and args.triplets_dir is None:
        sys.exit("--subsampled-only precisa de --triplets-dir")

    import nibabel as nib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]

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

    shell_key = _resolve_shell_key(shells, args.shell_b, args.shell_tol)
    n_dirs_full = int(shells[0].size + shells[shell_key].size)
    sh_order = args.sh_order or max_order_for_n_directions(n_dirs_full)
    print(f"sh_order={sh_order} (n_dirs_full={n_dirs_full})", flush=True)

    # sub-volume cubico centrado no centroide da mascara (ou no voxel manual
    # de --center-voxel, se dado) -- ver docstring do modulo pra a
    # justificativa de nao ajustar CSD no cerebro inteiro.
    mask_bool = mask.astype(bool)
    manual_center_voxel = None
    if args.center_voxel:
        parts = [p.strip() for p in args.center_voxel.split(",")]
        if len(parts) != 3:
            sys.exit(f"--center-voxel precisa ser 'X,Y,Z' (3 valores), recebi "
                      f"{args.center_voxel!r}.")
        try:
            manual_center_voxel = tuple(int(p) for p in parts)
        except ValueError:
            sys.exit(f"--center-voxel precisa ser 3 inteiros separados por virgula, recebi "
                      f"{args.center_voxel!r}.")
        centroid = manual_center_voxel
    else:
        centroid = tuple(int(round(c)) for c in np.argwhere(mask_bool).mean(axis=0))
    slices, origin = bounding_box(mask_bool, centroid, args.search_radius)
    print(f"{'Voxel central (fixado manualmente)' if manual_center_voxel else 'Centroide da mascara'}: "
          f"{centroid}; sub-volume: {slices} (origem {origin})", flush=True)

    sub_data = data[slices]
    sub_mask = mask_bool[slices]

    gt_n_peaks, gt_shm = fit_shm_and_npeaks(
        sub_data, bvals, bvecs, args.shell_b, sub_mask, args.shell_tol, sh_order,
        args.relative_peak_threshold, args.min_separation_angle, args.npeaks)

    def _crossing_frac_for_patch(o0, o1, slice_idx):
        """Fracao de voxels mascarados com >=--min-peaks-for-crossing picos
        dentro de um patch JA POSICIONADO (usado so pra exibir a mesma
        estatistica de find_best_crossing_patch quando o patch vem de
        centered_patch em vez da busca automatica)."""
        crossing_bool = sub_mask.astype(bool) & (gt_n_peaks >= args.min_peaks_for_crossing)
        idx3 = [slice(None)] * 3
        idx3[args.slice_axis] = slice_idx
        idx3 = tuple(idx3)
        mask_slice = sub_mask.astype(bool)[idx3]
        cross_slice = crossing_bool[idx3]
        patch_mask = mask_slice[o0:o0 + args.patch_size, o1:o1 + args.patch_size]
        patch_cross = cross_slice[o0:o0 + args.patch_size, o1:o1 + args.patch_size]
        n_masked = int(patch_mask.sum())
        return float(patch_cross.sum()) / n_masked if n_masked > 0 else float("nan")

    if manual_center_voxel is not None:
        local_center = tuple(manual_center_voxel[d] - origin[d] for d in range(3))
        found_manual = centered_patch(local_center, args.patch_size, args.slice_axis,
                                       sub_mask.shape)
        if found_manual is None:
            sys.exit(
                f"--center-voxel {manual_center_voxel} caiu fora do sub-volume recortado "
                f"(origem {origin}, --search-radius {args.search_radius}) ou "
                f"--patch-size nao cabe no sub-volume -- aumente --search-radius.")
        (o0, o1), slice_idx = found_manual
        crossing_frac = _crossing_frac_for_patch(o0, o1, slice_idx)
        print(f"Voxel central fixado manualmente (global {manual_center_voxel}); patch "
              f"centrado ali (fracao de voxels com cruzamento nesse patch = "
              f"{crossing_frac:.1%}, indice={slice_idx}, origem no patch={(o0, o1)})",
              flush=True)
    else:
        found = find_best_crossing_patch(
            gt_n_peaks, sub_mask, args.patch_size, args.slice_axis,
            min_peaks_for_crossing=args.min_peaks_for_crossing,
            min_mask_frac=args.min_mask_frac)
        if found is None:
            sys.exit(
                "Nenhum patch candidato atingiu --min-mask-frac dentro do sub-volume -- tente "
                "aumentar --search-radius (mais candidatos) ou diminuir --min-mask-frac/"
                "--patch-size.")
        (o0, o1), slice_idx, crossing_frac = found
        print(f"Melhor patch: fracao de voxels com cruzamento (>= "
              f"{args.min_peaks_for_crossing} picos) = {crossing_frac:.1%} "
              f"(slice_axis={args.slice_axis}, indice={slice_idx}, origem no patch={(o0, o1)})",
              flush=True)

    def _patch_slices(o0, o1, s, axis, size):
        idx3 = [None, None, None]
        out_axes = [a for a in range(3) if a != axis]
        idx3[out_axes[0]] = slice(o0, o0 + size)
        idx3[out_axes[1]] = slice(o1, o1 + size)
        idx3[axis] = slice(s, s + 1)
        return tuple(idx3)

    patch_slices_sub = _patch_slices(o0, o1, slice_idx, args.slice_axis, args.patch_size)
    gt_shm_patch = gt_shm[patch_slices_sub].reshape(args.patch_size, args.patch_size, -1)

    directions = in_plane_directions(args.slice_axis, n_angles=args.glyph_n_angles)

    methods_to_try = [("ground_truth", None), ("baseline_sh", args.baseline_dir),
                       ("rcae", args.rcae_dir)]
    for spec in args.extra_method:
        for pair in spec.split(","):
            name, _, path = pair.partition("=")
            if name and path:
                methods_to_try.append((name.strip(), path.strip()))

    panels = []  # (label, shm_patch, sh_order_desse_painel)
    for method, recon_dir in methods_to_try:
        if method == "ground_truth":
            panels.append((method, gt_shm_patch, sh_order))
            continue
        if recon_dir is None:
            continue
        sub_dir = Path(recon_dir) / tag / f"shell{int(args.shell_b)}" / f"n{args.n_level}"
        recon_path = sub_dir / "recon_target.nii.gz"
        if not recon_path.exists():
            print(f"[aviso] sem reconstrucao {method} para {tag}, pulando", flush=True)
            continue
        recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
        target_idx = np.load(sub_dir / "target_idx.npy")
        full = data.copy()
        full[..., target_idx] = recon
        full_sub = full[slices]
        try:
            _n_peaks, shm = fit_shm_and_npeaks(
                full_sub, bvals, bvecs, args.shell_b, sub_mask, args.shell_tol, sh_order,
                args.relative_peak_threshold, args.min_separation_angle, args.npeaks)
        except Exception as exc:
            print(f"[aviso] {tag}: CSD falhou pro metodo {method} "
                  f"({type(exc).__name__}: {exc}), pulando", flush=True)
            continue
        shm_patch = shm[patch_slices_sub].reshape(args.patch_size, args.patch_size, -1)
        panels.append((method, shm_patch, sh_order))

    # --subsampled-only (2026-09-02): SEM reconstrucao nenhuma -- ajusta CSD
    # so' nas direcoes de entrada reais (exclui as direcoes-alvo do esquema
    # de subamostragem via exclude_idx), mesma logica/mesma fonte de
    # target_idx de 11_peak_confusion_by_roi.py. sh_order_sub tipicamente
    # difere (e' MENOR) que `sh_order` (menos direcoes reais disponiveis),
    # por isso cada painel carrega sua PROPRIA ordem (ver render_glyph_field
    # abaixo -- sh_to_sf precisa da ordem que bate com o numero de
    # coeficientes do shm_patch daquele painel especifico).
    if args.subsampled_only:
        trip_path = Path(args.triplets_dir) / f"{tag}_rrin_triplets.npz"
        trip_key = f"{args.shell_b}__{args.n_level}__target"
        if not trip_path.exists() or trip_key not in np.load(trip_path).files:
            print(f"[aviso] {tag}: sem trincas para --subsampled-only "
                  f"(esperado {trip_path}, chave {trip_key!r}) -- pulando esse painel",
                  flush=True)
        else:
            target_idx = np.load(trip_path)[trip_key]
            sh_order_sub = (args.sh_order_subsampled_only
                             or max_order_for_n_directions(args.n_level))
            try:
                _n_peaks_sub, shm_sub = fit_shm_and_npeaks(
                    sub_data, bvals, bvecs, args.shell_b, sub_mask, args.shell_tol, sh_order_sub,
                    args.relative_peak_threshold, args.min_separation_angle, args.npeaks,
                    exclude_idx=target_idx)
                shm_patch_sub = shm_sub[patch_slices_sub].reshape(
                    args.patch_size, args.patch_size, -1)
                panels.append(("subsampled_only", shm_patch_sub, sh_order_sub))
                print(f"subsampled_only: sh_order={sh_order_sub} "
                      f"(auto via max_order_for_n_directions({args.n_level}))"
                      if args.sh_order_subsampled_only is None
                      else f"subsampled_only: sh_order={sh_order_sub} (forcado via "
                           f"--sh-order-subsampled-only)", flush=True)
            except Exception as exc:
                print(f"[aviso] {tag}: CSD falhou pro painel subsampled_only "
                      f"({type(exc).__name__}: {exc}), pulando", flush=True)

    if len(panels) < 2:
        sys.exit("Menos de 2 paineis disponiveis (GT + pelo menos 1 metodo) -- confira "
                  "--baseline-dir/--rcae-dir/--extra-method/--subsampled-only.")

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    if len(panels) == 1:
        axes = [axes]

    amplitude_ref = None
    for ax, (label, shm_patch, panel_sh_order) in zip(axes, panels):
        ref = amplitude_ref if args.normalize == "global" else None
        peak = render_glyph_field(ax, shm_patch, directions, panel_sh_order, args.glyph_scale,
                                   amplitude_ref=ref)
        if label == "ground_truth" and args.normalize == "global":
            amplitude_ref = peak
        ax.set_title(label, fontsize=10)

    fig.suptitle(f"{tag} -- shell{int(args.shell_b)}/n{args.n_level} -- glifos FOD "
                 f"(plano {'sagital' if args.slice_axis == 0 else 'coronal' if args.slice_axis == 1 else 'axial'}, "
                 f"cruzamento em {crossing_frac:.0%} dos voxels do GT)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Figura salva em {out_path}", flush=True)


if __name__ == "__main__":
    main()