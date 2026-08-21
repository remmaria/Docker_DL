#!/usr/bin/env python3
"""
Prova de conceito (independente do RCAE, roda com o que voce ja tem hoje):
demonstra empiricamente que CSD (single-shell single-tissue, Tournier07)
fica mal-posto/nao-confiavel quando ajustado com poucas direcoes, e
compara 3 condicoes por sujeito, na MESMA ordem SH (--sh-order, default 8
-- precisa de (8+1)(8+2)/2=45 coeficientes, batendo com o "45-60+ direcoes"
citado como o que hoje se precisa adquirir de verdade pra CSD confiavel):

  1. REFERENCIA: CSD ajustado em TODAS as direcoes reais da shell na
     aquisicao original (ex.: ~60 direcoes) -- o "gabarito".
  2. BRUTO-N: CSD ajustado usando SO as `n_level` direcoes medidas de
     entrada (ex.: 10), na MESMA ordem 8 -- sistema severamente
     sub-determinado (45 coeficientes, so 10-ish equacoes reais). Objetivo:
     mostrar que isso da FOD instavel/picos espurios (ou falha de ajuste
     direto), mesmo com a regularizacao do CSD.
  2b. BRUTO-N-ORDEM-MAX: a mesma condicao (so as `n_level` direcoes), mas
     na ordem MAXIMA que essas direcoes honestamente sustentam (mesma
     formula do baseline SH -- ex.: ordem 2 pra n_level=10). Existe pra
     responder de frente a objecao "nao seria mais justo comparar na
     ordem que 10 direcoes sustentam?": ordem 2 e estruturalmente incapaz
     de representar cruzamento (1 lobo so), entao o recall de cruzamento
     aqui e esperado ficar em ~0% SEMPRE, nao por falta de dado, mas por
     limitacao matematica da propria ordem baixa -- essa condicao prova
     isso com numero real, em vez de so argumento.
  3. PREENCHIDO-SH: CSD ajustado num volume "completo" reconstruido -- as
     `n_level` direcoes medidas de verdade + o resto preenchido pela
     reconstrucao SH do baseline (etapa 3, ja rodada). Numericamente bem-
     posto (tem "direcoes" suficiente de novo), mas objetivo e mostrar que
     mesmo assim o CSD erra os picos em voxels de fibra cruzando -- porque
     o preenchimento SH de ordem baixa (~2, com so 10 direcoes de entrada)
     nao carrega estrutura angular de ordem alta nenhuma, so suaviza.

Se/quando o RCAE terminar de treinar, --rcae-dir liga uma 4a condicao
(mesma logica do PREENCHIDO-SH, mas preenchido pelo RCAE) -- e o teste
direto de "o RCAE recupera estrutura de fibra cruzando que o SH nao
recupera, usando o mesmo numero de direcoes medidas".

Metricas reportadas (por condicao, vs. REFERENCIA, dentro da mascara):
  - confusao de classificacao "tem cruzamento" (n_picos>=2) vs "fibra
    unica" (n_picos==1) vs "nada" (n_picos==0)
  - erro angular medio do pico principal (graus), so nos voxels onde a
    condicao E a referencia concordam ter >=1 pico
  - taxa de falha de ajuste do CSD por sujeito/condicao (LinAlgError etc.)

Uso (roda numa amostra pequena de sujeitos por padrao -- CSD x3 condicoes
por sujeito e caro; aumente --n-subjects se quiser mais poder estatistico):
    python scripts/poc_csd_direction_count.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --baseline-dir work_dir/baseline_recon \
        --shell-b 1000 --n-level 10 --split test \
        --n-subjects 8 --seed 0 \
        --out-csv work_dir/metrics/poc_csd_shell1000_n10.csv

Requer DIPY.
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, load_dwi, split_shells
from utils.masking import load_or_build_mask


def max_order_for_n_directions(n_dirs: int) -> int:
    """Maior ordem par l_max tal que o numero de coeficientes SH,
    (l_max+1)(l_max+2)/2, seja <= n_dirs (mesma formula usada no baseline
    SH em utils/sh_basis.py -- aqui reimplementada pra nao acoplar este
    script de prova de conceito ao modulo de fit SH real)."""
    l_max = 0
    while True:
        next_l = l_max + 2
        n_coef = (next_l + 1) * (next_l + 2) // 2
        if n_coef > n_dirs:
            break
        l_max = next_l
    return l_max


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


def fit_csd_peaks(vol, bvals_sub, bvecs_sub, mask, sh_order, npeaks,
                   relative_peak_threshold, min_separation_angle):
    """Ajusta CSD single-shell single-tissue no subconjunto de volumes
    dado (`vol`, ja incluindo os b0) e devolve (n_peaks_map, peak_dirs,
    peak_values, shm_coeff) dentro da mascara. `peak_dirs` tem shape
    (X,Y,Z,npeaks,3) -- direcao (unitaria) de cada pico, zero se nao
    detectado. `shm_coeff` (X,Y,Z,n_coef) sao os coeficientes SH da FOD
    ajustada (convencao descoteaux07, mesma ordem l crescente/m=-l..l usada
    em utils/sh_basis.py -- ver sh_energy_by_order pra decompor por l)."""
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
    return n_peaks_map, peaks.peak_dirs, peaks.peak_values, peaks.shm_coeff


def sh_energy_by_order(shm_coeff, sh_order, mask):
    """Decompoe a energia dos coeficientes SH (shm_coeff, ...,n_coef) por
    ordem l (0,2,4,...,sh_order) -- convencao descoteaux07: coeficientes
    ordenados por l crescente, dentro de cada l por m=-l..l (mesmo bloco
    contiguo de tamanho 2l+1). Isso e o que permite responder de frente
    "ele so ajusta as ordens baixas e o resto fica em nivel de ruido, ou
    realmente usa toda a base?" com numero: se a energia de l>=4 (a parte
    que sozinha poderia representar cruzamento) e desprezivel/instavel
    quando so ha poucas direcoes reais, mesmo com sh_order alto pedido no
    ajuste, isso mostra a regularizacao/prior dominando esses
    coeficientes, nao dado real.

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


def angular_error_primary_peak(dirs_a, dirs_b, sel):
    """Erro angular (graus, 0-90) entre o pico principal (peak_dirs[...,0,:])
    de duas condicoes, restrito aos voxels em `sel` (bool mask). Direcoes
    de difusao nao tem sinal (antipodais equivalentes), entao usa o angulo
    minimo entre v e -v."""
    a = dirs_a[..., 0, :][sel]
    b = dirs_b[..., 0, :][sel]
    an = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)
    bn = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8, None)
    cos = np.abs(np.sum(an * bn, axis=-1))  # abs() = ja trata antipodal
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def classify(n_peaks_map):
    """0 picos / 1 pico (fibra unica) / 2+ picos (cruzamento), por voxel."""
    out = np.full(n_peaks_map.shape, "none", dtype=object)
    out[n_peaks_map == 1] = "single"
    out[n_peaks_map >= 2] = "crossing"
    return out


def process_subject(e, tag, args):
    bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
    data, _affine, _header = load_dwi(e.dwi_path)
    shells = split_shells(bvals, tol=args.shell_tol)
    b0_idx = shells.get(0, np.array([], dtype=int))
    if b0_idx.size == 0:
        return None, f"{tag}: sem volume b0"
    b0_mean = data[..., b0_idx].mean(axis=-1)
    mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

    shell_key = _resolve_shell_key(shells, args.shell_b, args.shell_tol)
    shell_idx_all = np.asarray(shells[shell_key], dtype=int)

    scheme_path = Path(args.scheme_dir) / f"{tag}_scheme.npz"
    if not scheme_path.exists():
        return None, f"{tag}: sem esquema de subamostragem"
    scheme = np.load(scheme_path)
    key = f"{args.shell_b}__{args.n_level}"
    if f"{key}__input" not in scheme.files:
        return None, f"{tag}: sem combo shell={args.shell_b}/n={args.n_level} no esquema"
    input_idx = scheme[f"{key}__input"]
    target_idx = scheme[f"{key}__target"]

    conditions = {}  # name -> (vol, bvals_sub, bvecs_sub, order)

    # 1. referencia: todas as direcoes reais dessa shell, ordem alta (args.sh_order)
    idx_ref = np.concatenate([b0_idx, shell_idx_all])
    idx_ref.sort()
    conditions["referencia"] = (data[..., idx_ref], bvals[idx_ref], bvecs[idx_ref], args.sh_order)

    # 2. bruto-N, ordem alta (args.sh_order) -- pergunta "da pra sustentar um
    # ajuste de ordem alta so com as N direcoes medidas?". Sub-determinado
    # de proposito -- a instabilidade/falha esperada AQUI e o ponto.
    idx_raw = np.concatenate([b0_idx, input_idx])
    idx_raw.sort()
    conditions["bruto_n"] = (data[..., idx_raw], bvals[idx_raw], bvecs[idx_raw], args.sh_order)

    # 2b. bruto-N, ordem MAXIMA que N direcoes honestamente sustentam
    # (mesma formula do baseline SH) -- pergunta diferente: "no melhor
    # ajuste possivel com so essas direcoes, ele acha cruzamento?". Como
    # essa ordem e <=2 pra n_level tipico (6-10), a resposta e
    # estruturalmente "nunca" (ordem 2 = 1 lobo so, nao representa 2+
    # picos) -- reportar isso com numero real fecha a duvida de "seria
    # mais justo testar na ordem que 10 direcoes sustentam" com dado, nao
    # so com argumento.
    order_matched = max_order_for_n_directions(len(input_idx))
    conditions["bruto_n_ordem_max"] = (data[..., idx_raw], bvals[idx_raw], bvecs[idx_raw], order_matched)

    # 3. preenchido-SH: entrada real + alvo reconstruido pelo baseline SH,
    # remontado no MESMO conjunto de indices da referencia (mesmo "N" de
    # direcoes que a referencia tem, so que parte reconstruida) -- ordem
    # alta de proposito (testa se o preenchimento da pra sustentar ordem
    # alta de verdade, ou so parece bem-posto numericamente).
    if args.baseline_dir is not None:
        sub_dir = Path(args.baseline_dir) / tag / f"shell{int(args.shell_b)}" / f"n{args.n_level}"
        recon_path = sub_dir / "recon_target.nii.gz"
        if recon_path.exists():
            import nibabel as nib
            recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
            recon_target_idx = np.load(sub_dir / "target_idx.npy")
            filled = data.copy()
            filled[..., recon_target_idx] = recon
            idx_filled = np.concatenate([b0_idx, input_idx, recon_target_idx])
            idx_filled = np.unique(idx_filled)
            conditions["preenchido_sh"] = (filled[..., idx_filled], bvals[idx_filled],
                                            bvecs[idx_filled], args.sh_order)

    # 4. (opcional) preenchido pelo RCAE, mesma logica, quando disponivel
    if args.rcae_dir is not None:
        sub_dir = Path(args.rcae_dir) / tag / f"shell{int(args.shell_b)}" / f"n{args.n_level}"
        recon_path = sub_dir / "recon_target.nii.gz"
        if recon_path.exists():
            import nibabel as nib
            recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
            recon_target_idx = np.load(sub_dir / "target_idx.npy")
            filled = data.copy()
            filled[..., recon_target_idx] = recon
            idx_filled = np.unique(np.concatenate([b0_idx, input_idx, recon_target_idx]))
            conditions["preenchido_rcae"] = (filled[..., idx_filled], bvals[idx_filled],
                                              bvecs[idx_filled], args.sh_order)

    results = {}
    fit_errors = []
    for name, (vol, bv, bvc, order) in conditions.items():
        try:
            n_peaks_map, peak_dirs, peak_vals, shm_coeff = fit_csd_peaks(
                vol, bv, bvc, mask, order, args.npeaks,
                args.relative_peak_threshold, args.min_separation_angle)
            energy_by_l = sh_energy_by_order(shm_coeff, order, mask)
            results[name] = (n_peaks_map, peak_dirs, peak_vals, len(bv), order, energy_by_l)
        except Exception as exc:
            fit_errors.append(f"{name}: {type(exc).__name__}: {exc}")
            results[name] = None

    if results.get("referencia") is None:
        return None, f"{tag}: CSD falhou na REFERENCIA ({fit_errors}) -- sujeito descartado"

    ref_peaks, ref_dirs, _ref_vals, ref_n, ref_order, ref_energy_by_l = results["referencia"]
    ref_mask = mask.astype(bool)
    ref_class = classify(ref_peaks)

    rows = []
    for name in ("bruto_n", "bruto_n_ordem_max", "preenchido_sh", "preenchido_rcae"):
        if name not in results:
            continue
        n_dirs_used = conditions[name][1].shape[0] if name in conditions else None
        order_used = conditions[name][3] if name in conditions else None
        if results[name] is None:
            rows.append({"subject": e.subject, "tag": tag, "condition": name,
                         "shell": args.shell_b, "n_level": args.n_level,
                         "n_dirs_used": n_dirs_used, "sh_order_used": order_used,
                         "fit_failed": True})
            continue
        cond_peaks, cond_dirs, _cond_vals, cond_n, cond_order, cond_energy_by_l = results[name]
        cond_class = classify(cond_peaks)

        both_valid = ref_mask & (cond_peaks >= 0)
        n_voxels = int(both_valid.sum())

        # confusao "tem cruzamento" (>=2 picos) vs referencia
        ref_crossing = (ref_class == "crossing") & both_valid
        cond_crossing = (cond_class == "crossing") & both_valid
        true_pos = int((ref_crossing & cond_crossing).sum())
        false_neg = int((ref_crossing & ~cond_crossing).sum())
        false_pos = int((~ref_crossing & cond_crossing).sum())
        true_neg = int((~ref_crossing & ~cond_crossing).sum())
        n_ref_crossing = int(ref_crossing.sum())
        recall_crossing = true_pos / n_ref_crossing if n_ref_crossing else float("nan")
        precision_crossing = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else float("nan")

        both_have_peak = both_valid & (ref_peaks >= 1) & (cond_peaks >= 1)
        ang_err = angular_error_primary_peak(ref_dirs, cond_dirs, both_have_peak)

        row = {
            "subject": e.subject, "tag": tag, "condition": name,
            "shell": args.shell_b, "n_level": args.n_level,
            "n_dirs_used": n_dirs_used, "sh_order_used": order_used, "fit_failed": False,
            "n_voxels": n_voxels, "n_ref_crossing_voxels": n_ref_crossing,
            "recall_crossing": recall_crossing, "precision_crossing": precision_crossing,
            "true_pos": true_pos, "false_neg": false_neg, "false_pos": false_pos, "true_neg": true_neg,
            "primary_peak_angular_error_mean_deg": float(np.mean(ang_err)) if ang_err.size else float("nan"),
            "primary_peak_angular_error_median_deg": float(np.median(ang_err)) if ang_err.size else float("nan"),
            "n_voxels_both_have_peak": int(both_have_peak.sum()),
        }

        # energia SH por ordem l (0,2,4,6,8) -- da condicao e da referencia
        # lado a lado, pra comparar direto na mesma linha sem precisar de
        # join. "energy_frac_high_order" soma as fracoes de l>=4 (a parte
        # que sozinha poderia representar cruzamento) -- ver resposta que
        # motivou essa adicao: mostra se o ajuste de ordem alta realmente
        # tem energia sustentada por dado, ou se e so ruido/prior.
        for l, (energy_mean, energy_frac) in cond_energy_by_l.items():
            row[f"energy_l{l}_mean"] = energy_mean
            row[f"energy_l{l}_frac"] = energy_frac
        row["energy_frac_high_order"] = float(sum(
            frac for l, (_e, frac) in cond_energy_by_l.items() if l >= 4)) if cond_energy_by_l else float("nan")
        for l, (energy_mean, energy_frac) in ref_energy_by_l.items():
            row[f"ref_energy_l{l}_frac"] = energy_frac
        row["ref_energy_frac_high_order"] = float(sum(
            frac for l, (_e, frac) in ref_energy_by_l.items() if l >= 4)) if ref_energy_by_l else float("nan")

        rows.append(row)
    return rows, ("; ".join(fit_errors) if fit_errors else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--rcae-dir", default=None)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--sh-order", type=int, default=8,
                     help="ordem SH da FOD do CSD, IGUAL nas 3-4 condicoes de proposito "
                          "(default 8 -- precisa de (8+1)(8+2)/2=45 coeficientes, batendo "
                          "com o '45-60+ direcoes' que a literatura recomenda pra CSD "
                          "confiavel). Forcar a mesma ordem em todas as condicoes e o que "
                          "torna a comparacao justa/direta.")
    ap.add_argument("--npeaks", type=int, default=3)
    ap.add_argument("--relative-peak-threshold", type=float, default=0.5)
    ap.add_argument("--min-separation-angle", type=float, default=25.0)
    ap.add_argument("--n-subjects", type=int, default=8,
                     help="amostra aleatoria de sujeitos do split (default 8 -- CSD x3-4 "
                          "condicoes por sujeito e caro; 0 = todos os sujeitos do split).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]
    if not entries:
        sys.exit(f"Nenhum sujeito no split={args.split!r}")
    if args.n_subjects and args.n_subjects > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(entries), size=min(args.n_subjects, len(entries)), replace=False)
        entries = [entries[i] for i in sorted(idx)]
    print(f"Rodando prova de conceito em {len(entries)} sujeito(s) "
          f"(shell={args.shell_b}, n_level={args.n_level}, sh_order={args.sh_order})", flush=True)

    all_rows = []
    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        try:
            rows, note = process_subject(e, tag, args)
            if rows is None:
                print(f"[aviso] {note}", flush=True)
                continue
            all_rows.extend(rows)
            if note:
                print(f"[aviso] {tag}: falhas parciais de ajuste -- {note}", flush=True)
            print(f"{tag}: ok ({len(rows)} condicoes avaliadas)", flush=True)
        except Exception:
            print(f"[erro] falha processando {tag} -- pulando. Traceback completo abaixo:",
                  flush=True)
            traceback.print_exc()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not all_rows:
        print("Nenhum resultado -- gravando CSV vazio em", out_csv, flush=True)
        pd.DataFrame(columns=["subject", "tag", "condition", "shell", "n_level", "n_dirs_used",
                               "fit_failed", "n_voxels", "recall_crossing", "precision_crossing",
                               "primary_peak_angular_error_mean_deg"]).to_csv(out_csv, index=False)
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print("\nResultados salvos em", out_csv)

    print("\n=== resumo (media entre sujeitos, por condicao) ===")
    ok = df[~df["fit_failed"].fillna(False)]
    if not ok.empty:
        summary_cols = ["n_dirs_used", "sh_order_used", "recall_crossing", "precision_crossing",
                        "primary_peak_angular_error_mean_deg", "energy_frac_high_order"]
        summary = ok.groupby("condition")[[c for c in summary_cols if c in ok.columns]].mean()
        print(summary)
        if "ref_energy_frac_high_order" in ok.columns:
            print(f"\n(referencia: fracao media de energia SH em ordem l>=4 = "
                  f"{ok['ref_energy_frac_high_order'].mean():.4f} -- compare com "
                  f"'energy_frac_high_order' de cada condicao acima. Quanto mais perto do "
                  f"valor da referencia, mais a condicao preserva estrutura angular de ordem "
                  f"alta real, em vez de ruido/prior da regularizacao.)")
    n_failed = int(df["fit_failed"].fillna(False).sum())
    if n_failed:
        print(f"\n{n_failed} ajuste(s) de CSD falharam (provavelmente a condicao 'bruto_n', "
              f"exatamente o comportamento que a prova de conceito quer demonstrar -- "
              f"CSD de ordem {args.sh_order} exige mais direcoes do que {args.n_level} pra "
              f"nem conseguir rodar de forma estavel).")
    print("\nLeitura: 'recall_crossing' baixo em bruto_n/preenchido_sh (vs. referencia) = "
          "o metodo perde voxels de fibra cruzando que a aquisicao densa real detectaria. "
          "'primary_peak_angular_error_mean_deg' alto = mesmo quando acerta que tem pico, "
          "a orientacao principal esta errada. Se preenchido_rcae aparecer com recall mais "
          "alto e erro angular mais baixo que preenchido_sh, e a prova direta de que a "
          "super-resolucao aprendida recupera estrutura de fibra cruzando que o SH nao "
          "recupera, usando o MESMO numero de direcoes medidas.\n"
          "'bruto_n_ordem_max' e o CSD ajustado na ordem que 10 direcoes honestamente "
          "sustentam (sh_order_used baixo, tipicamente 2) -- responde a objecao de "
          "'nao seria mais justo comparar na ordem que os dados sustentam?': espere "
          "recall_crossing ~0 aqui SEMPRE, porque ordem 2 e estruturalmente incapaz de "
          "representar 2+ picos, nao importa quao bom seja o dado -- ou seja, comparar "
          "nessa ordem 'justa' nao testa nada sobre deteccao de cruzamento, so confirma "
          "a limitacao matematica da ordem baixa. E por isso bruto_n/preenchido_sh/"
          "preenchido_rcae sao testados na ordem alta (--sh-order) de proposito: e a "
          "unica ordem em que a pergunta 'da pra detectar cruzamento' faz sentido.")


if __name__ == "__main__":
    main()