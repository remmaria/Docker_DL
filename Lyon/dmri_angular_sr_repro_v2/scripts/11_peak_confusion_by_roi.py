#!/usr/bin/env python3
"""
Etapa 11 (downstream, metricas de deteccao de picos por ROI): ajusta CSD
(single-shell single-tissue, Tournier07, via DIPY -- mesma familia de
scripts/crossing_fiber_stratified_eval.py) tanto no GROUND TRUTH quanto em
cada metodo de reconstrucao pedido, e conta, por voxel/ROI, quantos picos
de FOD casam entre os dois (verdadeiro positivo), quantos a reconstrucao
"inventou" que o ground truth nao tem (falso positivo) e quantos o ground
truth tem mas a reconstrucao perdeu (falso negativo) -- alem de contar
voxels onde os dois concordam em "nenhum pico" (verdadeiro negativo,
agregado por voxel, nao por pico individual, ja que nao faz sentido
"parear" a ausencia de alguma coisa).

Casamento de picos: PARA CADA VOXEL, ordena todos os pares (pico
predito, pico real) por distancia angular crescente e vai casando greedy
(o par mais proximo primeiro, sem reusar nenhum dos dois lados), so aceita
o casamento se a distancia for <= --peak-match-threshold-deg (default 25
graus -- valor comum na literatura de comparacao de picos de FOD/CSD,
mesma ordem de grandeza do --min-separation-angle usado pelo proprio CSD
pra distinguir dois picos DENTRO do mesmo voxel). Sobra de picos preditos
sem par vira FP; sobra de picos reais sem par vira FN.

Duas situacoes de comparacao (mutuamente compativeis, pode rodar as duas
juntas numa mesma chamada):

1. Metodos de reconstrucao "volume completo" (baseline_sh, RRIN, AMT, HFD,
   naive_blend, ...) via --baseline-dir/--rcae-dir/--extra-method (mesma
   convencao de 06_evaluate_reconstruction.py/07_downstream_dti_noddi.py):
   o volume comparado tem o MESMO NUMERO de direcoes do ground truth (so
   os alvos held-out mudam de valor, nao de contagem) -- entao o CSD dos
   dois lados usa a MESMA ordem SH (--sh-order, auto por
   utils.sh_basis.max_order_for_n_directions do total de direcoes da shell
   se nao for passado), comparacao justa "mesma informacao nominal, dado
   diferente".
2. --subsampled-only (ADITIVO, requer --triplets-dir): sem reconstrucao
   nenhuma, CSD ajustado só com as direcoes de ENTRADA reais (n_level) --
   aqui a ordem SH e' NECESSARIAMENTE menor (auto por
   max_order_for_n_directions(n_level), ou --sh-order-subsampled-only pra
   forcar outra) porque ha' de fato menos direcoes medidas. Serve de piso:
   quanto a reconstrucao esta ajudando a achar os picos certos, comparado
   a so aceitar a resolucao angular mais baixa da aquisicao subamostrada.

Uso:
    python scripts/11_peak_confusion_by_roi.py \
        --manifest work_dir/manifest.csv \
        --baseline-dir work_dir/baseline_recon \
        --extra-method rrin=work_dir/rrin_recon \
        --extra-method naive_blend=work_dir/naive_blend_recon \
        --subsampled-only --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --roi-tracts FX,CGC,CGH,UF \
        --out-csv work_dir/metrics/peak_confusion_shell1000_n16.csv

Requer DIPY (nao disponivel neste ambiente de desenvolvimento -- verificado
so por python3 -m py_compile e revisao manual, mesma disciplina ja usada
em crossing_fiber_stratified_eval.py).
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# evita que o resumo impresso no log corte colunas com "..." (a tabela por
# method x roi ficou mais larga com as colunas de energia SH por ordem) --
# so afeta a IMPRESSAO, o CSV salvo nunca foi truncado.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, load_dwi, split_shells
from utils.masking import load_or_build_mask, load_roi_masks, JHU_TRACT_LABELS
from utils.sh_basis import max_order_for_n_directions


def _resolve_shell_key(shells: dict, shell_b: float, tol: float) -> float:
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
              exclude_idx=None):
    """CSD single-shell single-tissue (Tournier07) -- devolve
    (n_peaks_map, peak_dirs, peak_values, shm_coeff), todos com shape
    espacial igual a `mask` (peak_dirs/peak_values ganham uma dimensao
    extra `npeaks`, peak_dirs mais uma de tamanho 3, shm_coeff ganha uma
    dimensao `n_coef` -- convencao descoteaux07, mesma ordem l crescente/
    m=-l..l usada em sh_energy_by_order). `exclude_idx` (ADITIVO, default
    None): remove esses indices do conjunto de direcoes usadas no ajuste --
    usado pelo modo --subsampled-only (ver main()), mesma semantica de
    exclude_idx em scripts/07_downstream_dti_noddi.py:fit_dti.
    """
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
    return n_peaks_map, peaks.peak_dirs, peaks.peak_values, peaks.shm_coeff


def sh_energy_by_order(shm_coeff, sh_order, mask):
    """Decompoe a energia dos coeficientes SH (shm_coeff, ...,n_coef) por
    ordem l (0,2,4,...,sh_order) -- MESMA funcao/racional de
    scripts/poc_csd_direction_count.py:sh_energy_by_order (reproduzida
    aqui, sem import cruzado entre scripts de etapas diferentes, mesmo
    padrao ja usado para _resolve_shell_key neste arquivo). Convencao
    descoteaux07: coeficientes ordenados por l crescente, dentro de cada l
    por m=-l..l (bloco contiguo de tamanho 2l+1).

    Responde "a reconstrucao carrega estrutura angular real de ordem alta
    (l>=4, a unica capaz de representar cruzamento de fibras) ou so
    engordou a ordem baixa/suavizou?" -- um metodo pode ter nmse/rmse
    'aceitavel' (seção 16 do addendum) mas MESMO ASSIM ter a energia
    concentrada em l=0,2 enquanto o ground truth tem energia real em l>=4:
    isso e' a assinatura numerica de "parece plausivel mas nao recuperou
    informacao angular genuina".

    Devolve dict {l: (energia_media_por_voxel, fracao_da_energia_total)},
    medias sobre os voxels da mascara (energia = soma dos coef^2 no bloco
    de ordem l; fracao = energia(l) / soma de todas as ordens, por voxel,
    depois com media sobre voxels)."""
    mask_bool = mask.astype(bool)
    coeffs_masked = shm_coeff[mask_bool]  # (n_voxels, n_coef)
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


def match_peaks_voxel(true_dirs, true_vals, pred_dirs, pred_vals, threshold_deg):
    """Casamento greedy (mais proximo primeiro) entre os picos REAIS de UM
    voxel e os picos PREDITOS do mesmo voxel. Cada lado e' uma lista de
    vetores unitarios (so os slots com peak_values>0 contam -- padding de
    peaks_from_model tem peak_values==0). Direcoes tratadas com simetria
    antipodal (v == -v, convencao usual de picos de FOD), entao a
    distancia usada e' min(angulo, 180-angulo).

    Retorna (n_tp, n_fp, n_fn) para esse voxel.
    """
    true_list = [true_dirs[k] for k in range(true_dirs.shape[0]) if true_vals[k] > 0]
    pred_list = [pred_dirs[k] for k in range(pred_dirs.shape[0]) if pred_vals[k] > 0]
    n_true, n_pred = len(true_list), len(pred_list)
    if n_true == 0 and n_pred == 0:
        return 0, 0, 0  # voxel "vazio" dos dois lados -- nao entra em TP/FP/FN (ver TN a parte)

    pairs = []
    for i, tv in enumerate(true_list):
        for j, pv in enumerate(pred_list):
            cos = abs(float(np.dot(tv, pv)))  # abs() = simetria antipodal
            cos = min(1.0, max(-1.0, cos))
            ang = np.degrees(np.arccos(cos))
            pairs.append((ang, i, j))
    pairs.sort(key=lambda x: x[0])

    matched_true, matched_pred = set(), set()
    n_tp = 0
    for ang, i, j in pairs:
        if ang > threshold_deg:
            break  # ja ordenado por angulo -- nenhum par restante passa no limiar
        if i in matched_true or j in matched_pred:
            continue
        matched_true.add(i)
        matched_pred.add(j)
        n_tp += 1

    n_fn = n_true - len(matched_true)
    n_fp = n_pred - len(matched_pred)
    return n_tp, n_fp, n_fn


def confusion_for_roi(gt_n_peaks, gt_dirs, gt_vals, pred_n_peaks, pred_dirs, pred_vals,
                       roi_mask, threshold_deg):
    """Agrega TP/FP/FN (por pico) e TN (por voxel, ambos os lados com 0
    picos) dentro de uma ROI. Retorna dict com as 4 contagens + n_voxels
    (voxels da ROI com pelo menos 1 pico de algum lado, ou seja
    TP+FP+FN>0 contabilizados a nivel de picos, mais os TN a nivel de
    voxel)."""
    idxs = np.argwhere(roi_mask)
    tp = fp = fn = tn_voxels = 0
    for (x, y, z) in idxs:
        gt_n = gt_n_peaks[x, y, z]
        pr_n = pred_n_peaks[x, y, z]
        if gt_n < 0 or pr_n < 0:
            continue  # fora da mascara em algum dos dois lados (sentinela -1)
        if gt_n == 0 and pr_n == 0:
            tn_voxels += 1
            continue
        vtp, vfp, vfn = match_peaks_voxel(
            gt_dirs[x, y, z], gt_vals[x, y, z], pred_dirs[x, y, z], pred_vals[x, y, z],
            threshold_deg)
        tp += vtp
        fp += vfp
        fn += vfn
    return {"TP": tp, "FP": fp, "FN": fn, "TN_voxels": tn_voxels}


def _process_subject(e, tag, args, roi_tracts):
    bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
    data, _affine, _header = load_dwi(e.dwi_path)
    shells = split_shells(bvals, tol=args.shell_tol)
    b0_idx = shells.get(0, np.array([], dtype=int))
    if b0_idx.size == 0:
        print(f"[erro] {tag}: nenhum volume b0 encontrado, pulando")
        return []
    b0_mean = data[..., b0_idx].mean(axis=-1)
    mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

    rois = {"whole_mask": mask.astype(bool)}
    if roi_tracts:
        rois.update(load_roi_masks(e.dwi_path, roi_tracts, base_mask=mask))

    shell_key = _resolve_shell_key(shells, args.shell_b, args.shell_tol)
    n_dirs_full = int(shells[0].size + shells[shell_key].size)
    sh_order_full = args.sh_order or max_order_for_n_directions(n_dirs_full)

    gt_n_peaks, gt_dirs, gt_vals, gt_shm_coeff = fit_peaks(
        data, bvals, bvecs, args.shell_b, mask, args.shell_tol, sh_order_full,
        args.relative_peak_threshold, args.min_separation_angle, args.npeaks)

    # energia SH por ordem l do GROUND TRUTH, ja calculada por ROI aqui (uma
    # vez por sujeito, reaproveitada em todos os metodos abaixo) -- e a
    # referencia contra a qual "energy_frac_high_order" de cada metodo e'
    # comparada (ver sh_energy_by_order). sh_order_full e' o mesmo em todas
    # as condicoes de "volume completo" (baseline_sh/rrin/etc.), entao o
    # gt_energy_by_roi calculado com sh_order_full serve pra todas elas;
    # --subsampled-only usa sh_order_sub (tipicamente menor) e por isso tem
    # sua PROPRIA referencia de GT recalculada mais abaixo, no mesmo order.
    gt_energy_by_roi_full = {
        roi_name: sh_energy_by_order(gt_shm_coeff, sh_order_full, roi_mask)
        for roi_name, roi_mask in rois.items()
    }

    rows = []

    def _failed_rows(method, order):
        # mesma logica do PoC (poc_csd_direction_count.py): um ajuste de CSD
        # pode falhar de verdade (ex.: sistema severamente sub-determinado,
        # LinAlgError) sem que isso deva derrubar o sujeito INTEIRO -- grava
        # uma linha fit_failed=True por ROI (contagens NaN) e segue pros
        # proximos metodos/sujeitos, em vez de propagar a excecao.
        return [{"subject": e.subject, "tag": tag, "method": method,
                 "shell": args.shell_b, "n_level": args.n_level, "roi": roi_name,
                 "sh_order": order, "fit_failed": True,
                 "TP": np.nan, "FP": np.nan, "FN": np.nan, "TN_voxels": np.nan,
                 "energy_frac_high_order": np.nan, "ref_energy_frac_high_order": np.nan}
                for roi_name in rois]

    methods_to_try = [("baseline_sh", args.baseline_dir), ("rcae", args.rcae_dir)] + args.extra_methods
    for method, recon_dir in methods_to_try:
        if recon_dir is None:
            continue
        sub_dir = Path(recon_dir) / tag / f"shell{int(args.shell_b)}" / f"n{args.n_level}"
        recon_path = sub_dir / "recon_target.nii.gz"
        if not recon_path.exists():
            print(f"[aviso] sem reconstrucao {method} para {tag}")
            continue
        import nibabel as nib
        recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
        target_idx = np.load(sub_dir / "target_idx.npy")
        full = data.copy()
        full[..., target_idx] = recon

        try:
            pred_n_peaks, pred_dirs, pred_vals, pred_shm_coeff = fit_peaks(
                full, bvals, bvecs, args.shell_b, mask, args.shell_tol, sh_order_full,
                args.relative_peak_threshold, args.min_separation_angle, args.npeaks)
        except Exception as exc:
            print(f"[aviso] {tag}: CSD falhou pro metodo {method} "
                  f"({type(exc).__name__}: {exc}) -- gravando fit_failed e seguindo pros "
                  f"outros metodos")
            rows.extend(_failed_rows(method, sh_order_full))
            continue

        for roi_name, roi_mask in rois.items():
            conf = confusion_for_roi(gt_n_peaks, gt_dirs, gt_vals, pred_n_peaks, pred_dirs,
                                      pred_vals, roi_mask, args.peak_match_threshold_deg)
            energy_by_l = sh_energy_by_order(pred_shm_coeff, sh_order_full, roi_mask)
            row = {"subject": e.subject, "tag": tag, "method": method,
                   "shell": args.shell_b, "n_level": args.n_level, "roi": roi_name,
                   "sh_order": sh_order_full, "fit_failed": False, **conf,
                   "energy_frac_high_order": _energy_frac_high_order(energy_by_l),
                   "ref_energy_frac_high_order": _energy_frac_high_order(
                       gt_energy_by_roi_full.get(roi_name, {}))}
            for l, (_e, frac) in energy_by_l.items():
                row[f"energy_l{l}_frac"] = frac
            rows.append(row)

    if args.subsampled_only:
        trip_path = Path(args.triplets_dir) / f"{tag}_rrin_triplets.npz"
        trip_key = f"{args.shell_b}__{args.n_level}__target"
        if not trip_path.exists() or trip_key not in np.load(trip_path).files:
            print(f"[aviso] {tag}: sem trincas para --subsampled-only, pulando esse metodo")
        else:
            target_idx = np.load(trip_path)[trip_key]
            sh_order_sub = (args.sh_order_subsampled_only
                             or max_order_for_n_directions(args.n_level))
            try:
                pred_n_peaks, pred_dirs, pred_vals, pred_shm_coeff = fit_peaks(
                    data, bvals, bvecs, args.shell_b, mask, args.shell_tol, sh_order_sub,
                    args.relative_peak_threshold, args.min_separation_angle, args.npeaks,
                    exclude_idx=target_idx)
            except Exception as exc:
                print(f"[aviso] {tag}: CSD falhou pro metodo subsampled_only "
                      f"({type(exc).__name__}: {exc}) -- gravando fit_failed e seguindo")
                rows.extend(_failed_rows("subsampled_only", sh_order_sub))
                pred_n_peaks = None
            if pred_n_peaks is not None:
                # sh_order_sub tipicamente difere de sh_order_full (menos
                # direcoes reais), entao a referencia de energia do GT
                # precisa vir de um CSD do GT ajustado NESSA MESMA ordem --
                # nao da pra so truncar gt_shm_coeff (ajustado em
                # sh_order_full): os coeficientes de um fit CSD dependem da
                # ordem pedida (nao e' so preencher com zero as ordens
                # extras), entao truncar o array na marra usaria coeficientes
                # calibrados pra ordem errada E ainda normalizaria a fracao
                # pela energia total ERRADA (incluiria energia de l>sh_order_sub
                # que um fit real nessa ordem nunca teria). Reajusta o GT do
                # zero em sh_order_sub (mesmo custo de um metodo a mais).
                if sh_order_sub == sh_order_full:
                    gt_energy_by_roi_sub = gt_energy_by_roi_full
                else:
                    _gt_n_sub, _gt_dirs_sub, _gt_vals_sub, gt_shm_coeff_sub = fit_peaks(
                        data, bvals, bvecs, args.shell_b, mask, args.shell_tol, sh_order_sub,
                        args.relative_peak_threshold, args.min_separation_angle, args.npeaks)
                    gt_energy_by_roi_sub = {
                        roi_name: sh_energy_by_order(gt_shm_coeff_sub, sh_order_sub, roi_mask)
                        for roi_name, roi_mask in rois.items()
                    }
                for roi_name, roi_mask in rois.items():
                    conf = confusion_for_roi(gt_n_peaks, gt_dirs, gt_vals, pred_n_peaks, pred_dirs,
                                              pred_vals, roi_mask, args.peak_match_threshold_deg)
                    energy_by_l = sh_energy_by_order(pred_shm_coeff, sh_order_sub, roi_mask)
                    row = {"subject": e.subject, "tag": tag, "method": "subsampled_only",
                           "shell": args.shell_b, "n_level": args.n_level, "roi": roi_name,
                           "sh_order": sh_order_sub, "fit_failed": False, **conf,
                           "energy_frac_high_order": _energy_frac_high_order(energy_by_l),
                           "ref_energy_frac_high_order": _energy_frac_high_order(
                               gt_energy_by_roi_sub.get(roi_name, {}))}
                    for l, (_e, frac) in energy_by_l.items():
                        row[f"energy_l{l}_frac"] = frac
                    rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--rcae-dir", default=None)
    ap.add_argument("--extra-method", action="append", default=[],
                     help="mesma convencao NOME=CAMINHO de 06_evaluate_reconstruction.py/"
                          "07_downstream_dti_noddi.py, repetivel.")
    ap.add_argument("--subsampled-only", action="store_true",
                     help="ADITIVO, requer --triplets-dir: tambem inclui o metodo "
                          "'subsampled_only' (CSD so nas direcoes de entrada reais, sem "
                          "reconstruir nada -- ver docstring do modulo).")
    ap.add_argument("--triplets-dir", default=None)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--subjects", default=None,
                     help="lista separada por virgula de 'tag' de sujeito (subject, ou "
                          "subject_session se houver sessao) para rodar em sujeitos "
                          "ESPECIFICOS em vez de todo o --split -- mesma convencao de "
                          "--subjects em poc_csd_direction_count.py. Util pra testar rapido "
                          "num sujeito so antes de rodar o dataset inteiro (CSD por sujeito e "
                          "caro). Quando usado, o sharding via --shard-index/--shard-count "
                          "ainda funciona normalmente sobre a lista resultante.")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--sh-order", type=int, default=None,
                     help="ordem SH do CSD para o ground truth e para os metodos de volume "
                          "completo (default None = auto via "
                          "utils.sh_basis.max_order_for_n_directions do total de direcoes "
                          "da shell -- mesma ordem dos dois lados, ja que tem a mesma "
                          "contagem nominal de direcoes).")
    ap.add_argument("--sh-order-subsampled-only", type=int, default=None,
                     help="ordem SH do CSD so para o metodo --subsampled-only (default "
                          "None = auto via max_order_for_n_directions(n_level), "
                          "necessariamente menor que --sh-order pois ha menos direcoes reais).")
    ap.add_argument("--npeaks", type=int, default=3)
    ap.add_argument("--relative-peak-threshold", type=float, default=0.5)
    ap.add_argument("--min-separation-angle", type=float, default=25.0)
    ap.add_argument("--peak-match-threshold-deg", type=float, default=25.0,
                     help="tolerancia angular (graus) para considerar que um pico predito e "
                          "um pico do ground truth sao 'o mesmo' pico (default 25 -- mesma "
                          "ordem de grandeza do --min-separation-angle do proprio CSD).")
    ap.add_argument("--roi-tracts", default=None,
                     help="mesma convencao de --roi-tracts em 07_downstream_dti_noddi.py. "
                          "Tratos JHU conhecidos: " + ", ".join(
                              f"{k} ({v})" for k, v in JHU_TRACT_LABELS.items()))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    extra_methods = []
    for spec in args.extra_method:
        if "=" not in spec:
            sys.exit(f"--extra-method invalido: {spec!r} (esperado NOME=CAMINHO)")
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            sys.exit(f"--extra-method invalido: {spec!r} (NOME e CAMINHO nao podem ser vazios)")
        extra_methods.append((name, path))
    args.extra_methods = extra_methods

    if args.baseline_dir is None and args.rcae_dir is None and not extra_methods and not args.subsampled_only:
        sys.exit("Passe pelo menos --baseline-dir, --rcae-dir, --extra-method ou --subsampled-only")
    if args.subsampled_only and args.triplets_dir is None:
        sys.exit("--subsampled-only precisa de --triplets-dir")
    if not (0 <= args.shard_index < max(args.shard_count, 1)):
        sys.exit(f"--shard-index ({args.shard_index}) fora do intervalo [0, {args.shard_count})")

    roi_tracts = [t.strip() for t in args.roi_tracts.split(",") if t.strip()] if args.roi_tracts else []

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]

    def _tag_of(e):
        return e.subject if not e.session else f"{e.subject}_{e.session}"

    if args.subjects:
        wanted = {t.strip() for t in args.subjects.split(",") if t.strip()}
        entries = [e for e in entries if _tag_of(e) in wanted]
        missing = wanted - {_tag_of(e) for e in entries}
        if missing:
            print(f"[aviso] --subjects pediu {sorted(missing)}, mas nao encontrei no split "
                  f"{args.split!r} do manifesto.", flush=True)
        if not entries:
            sys.exit(f"Nenhum dos sujeitos pedidos em --subjects foi encontrado no split "
                      f"{args.split!r} -- nada a fazer.")

    if args.shard_count > 1:
        entries = entries[args.shard_index::args.shard_count]
        print(f"[shard {args.shard_index}/{args.shard_count}] {len(entries)} sujeitos neste shard",
              flush=True)

    out_csv = Path(args.out_csv)
    if args.shard_count > 1:
        out_csv = out_csv.with_name(
            f"{out_csv.stem}.shard{args.shard_index}of{args.shard_count}{out_csv.suffix}")

    all_rows = []
    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        try:
            rows = _process_subject(e, tag, args, roi_tracts)
            all_rows.extend(rows)
            print(f"{tag}: {len(rows)} linhas (metodo x roi)", flush=True)
        except Exception:
            print(f"[erro] falha processando {tag} -- pulando este sujeito e continuando. "
                  f"Traceback completo abaixo:", flush=True)
            traceback.print_exc()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["subject", "tag", "method", "shell", "n_level", "roi", "sh_order", "fit_failed",
            "TP", "FP", "FN", "TN_voxels", "energy_frac_high_order", "ref_energy_frac_high_order"]
    if not all_rows:
        print(f"[shard {args.shard_index}/{args.shard_count}] nenhum resultado neste shard -- "
              f"gravando CSV vazio em {out_csv}", flush=True)
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False)
        return

    df = pd.DataFrame(all_rows)
    df["precision"] = df["TP"] / (df["TP"] + df["FP"]).replace(0, np.nan)
    df["recall"] = df["TP"] / (df["TP"] + df["FN"]).replace(0, np.nan)
    df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"])
    df.to_csv(out_csv, index=False)
    print("Metricas de confusao de picos (TP/FP/FN/TN) salvas em", out_csv)

    n_failed = int(df["fit_failed"].fillna(False).sum())
    if n_failed:
        print(f"\n{n_failed} ajuste(s) de CSD falharam (ver [aviso] nos logs acima -- "
              f"excluidos do resumo abaixo, igual a poc_csd_direction_count.py faz).")
    ok = df[~df["fit_failed"].fillna(False)]
    summary = ok.groupby(["method", "roi"])[["TP", "FP", "FN", "TN_voxels"]].sum()
    summary["precision"] = summary["TP"] / (summary["TP"] + summary["FP"]).replace(0, np.nan)
    summary["recall"] = summary["TP"] / (summary["TP"] + summary["FN"]).replace(0, np.nan)
    print(summary)

    if "energy_frac_high_order" in ok.columns:
        energy_summary = ok.groupby(["method", "roi"])[
            ["energy_frac_high_order", "ref_energy_frac_high_order"]].mean()
        print("\nEnergia SH media em ordem l>=4 (fracao do total), metodo vs. ground truth:")
        print(energy_summary)
        print("\nLeitura: quanto mais 'energy_frac_high_order' se aproxima de "
              "'ref_energy_frac_high_order' (a mesma coluna calculada no ground truth), mais "
              "a reconstrucao preserva estrutura angular de ordem alta real (a unica capaz de "
              "representar cruzamento) em vez de so suavizar pra ordem baixa -- um metodo pode "
              "ter recall/precision de picos ok e MESMO ASSIM ter essa fracao bem abaixo da "
              "referencia, sinal de que a estrutura recuperada e mais fraca/instavel do que a "
              "contagem de picos sozinha sugere.")


if __name__ == "__main__":
    main()