#!/usr/bin/env python3
"""
Etapa 7 (validacao downstream): ajusta DTI (sempre) e, opcionalmente, NODDI
sobre os dados reconstruidos (baseline SH e/ou RCAE) e sobre o ground
truth, e compara FA/MD/AD/RD (e NDI/ODI/ISOVF, se --run-noddi) dentro da
mascara de substancia branca.

A logica: para reconstruir o "volume completo" de uma shell dada uma
reconstrucao, mantemos os volumes de entrada (in-sample, ja medidos de
verdade) e substituimos apenas os volumes held-out (indices em
target_idx.npy) pelos valores preditos (recon_target.nii.gz). Isso reflete
o cenario real de uso: voce mede N direcoes, o modelo preenche o resto.

Uso:
    python scripts/07_downstream_dti_noddi.py \
        --manifest work_dir/manifest.csv \
        --baseline-dir work_dir/baseline_recon \
        --rcae-dir work_dir/rcae_recon \
        --shell-b 1000 --n-level 10 \
        --out-dir work_dir/downstream \
        --run-noddi   # opcional, requer pacote AMICO instalado e configurado

Metodo(s) adicional(is) alem de baseline_sh/rcae (ex.: a linha RRIN/VFI-por-
triplets, ver protocolo secao 10.1/10.2) via --extra-method NOME=CAMINHO,
repetivel -- mesma convencao de 06_evaluate_reconstruction.py, ex.:
    --extra-method rrin=work_dir/rrin_recon

DTI usa DIPY (obrigatorio). NODDI usa AMICO (opcional; se nao instalado, a
etapa e pulada com aviso, o resto do script continua normalmente).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_dwi, load_bval_bvec, split_shells
from utils.masking import load_or_build_mask, load_roi_masks, JHU_TRACT_LABELS


def build_full_volume(gt_data, target_idx, recon_dir, tag, shell_b, n_level):
    """Substitui, numa copia dos dados originais, os volumes held-out
    (target_idx) pelos valores reconstruidos. Se nao houver reconstrucao
    disponivel para esse sujeito, retorna None.
    """
    sub_dir = Path(recon_dir) / tag / f"shell{int(shell_b)}" / f"n{n_level}"
    recon_path = sub_dir / "recon_target.nii.gz"
    if not recon_path.exists():
        return None
    import nibabel as nib
    recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
    out = gt_data.copy()
    out[..., target_idx] = recon
    return out


def fit_dti(data, bvals, bvecs, shell_b, mask, shell_tol=100.0):
    """Ajusta DTI usando apenas b0s + a shell indicada (comportamento
    padrao de DTI classico, que nao deve misturar shells de b muito
    diferentes sem um modelo multi-shell dedicado).
    """
    import dipy.reconst.dti as dti
    from dipy.core.gradients import gradient_table

    shells = split_shells(bvals, tol=shell_tol)
    idx = np.concatenate([shells[0], shells[shell_b]])
    idx.sort()
    gtab = gradient_table(bvals[idx], bvecs[idx])
    model = dti.TensorModel(gtab)
    fit = model.fit(data[..., idx], mask=mask)
    return {"FA": fit.fa, "MD": fit.md, "AD": fit.ad, "RD": fit.rd}


def try_fit_noddi(data, bvals, bvecs, mask, work_dir: Path):
    """Tentativa best-effort de ajuste NODDI via AMICO. Requer multi-shell
    (>=2 shells nao-zero) e o pacote `amico` instalado + kernels gerados
    (baixados na primeira execucao, AMICO cuida disso via
    amico.core.setup()). Se algo faltar, retorna None e imprime instrucoes.

    Isso e mais fragil que o DTI porque a API do AMICO espera arquivos em
    disco (scheme, mask, dwi) em vez de arrays em memoria -- por isso
    gravamos temporarios em work_dir.
    """
    try:
        import amico
        import nibabel as nib
    except ImportError:
        print("[aviso] pacote 'amico' nao encontrado -- pulando NODDI. "
              "Instale com `pip install dmri-amico` no cluster.")
        return None

    work_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), work_dir / "dwi.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), work_dir / "mask.nii.gz")
    np.savetxt(work_dir / "dwi.bval", bvals.reshape(1, -1), fmt="%d")
    np.savetxt(work_dir / "dwi.bvec", bvecs.T, fmt="%.6f")

    try:
        amico.core.setup()
        ae = amico.Evaluation(str(work_dir), str(work_dir))
        amico.util.fsl2scheme(str(work_dir / "dwi.bval"), str(work_dir / "dwi.bvec"),
                               str(work_dir / "dwi.scheme"))
        ae.load_data(dwi_filename=str(work_dir / "dwi.nii.gz"),
                     scheme_filename=str(work_dir / "dwi.scheme"),
                     mask_filename=str(work_dir / "mask.nii.gz"), b0_thr=50)
        ae.set_model("NODDI")
        ae.generate_kernels(regenerate=False)
        ae.load_kernels()
        ae.fit()
        ae.save_results()
        ndi = nib.load(str(work_dir / "AMICO" / "NODDI" / "FIT_ICVF.nii.gz")).get_fdata()
        odi = nib.load(str(work_dir / "AMICO" / "NODDI" / "FIT_OD.nii.gz")).get_fdata()
        isovf = nib.load(str(work_dir / "AMICO" / "NODDI" / "FIT_ISOVF.nii.gz")).get_fdata()
        return {"NDI": ndi, "ODI": odi, "ISOVF": isovf}
    except Exception as exc:  # AMICO tem muitas formas de falhar (kernels, versao, etc.)
        print(f"[aviso] ajuste NODDI via AMICO falhou ({exc}); pulando essa etapa "
              f"e mantendo o restante da avaliacao (DTI).")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--rcae-dir", default=None)
    ap.add_argument("--extra-method", action="append", default=[],
                     help="metodo(s) adicional(is) a avaliar em DTI/NODDI, no formato "
                          "NOME=CAMINHO -- mesma convencao de --extra-method em "
                          "06_evaluate_reconstruction.py (ex.: a linha RRIN/VFI-por-triplets, "
                          "ver protocolo secao 10.1/10.2): --extra-method rrin=work_dir/rrin_recon. "
                          "Pode repetir a flag pra mais de um metodo extra. Cada CAMINHO precisa "
                          "ter a mesma estrutura de --baseline-dir/--rcae-dir "
                          "(<tag>/shell<B>/n<N>/recon_target.nii.gz + target_idx.npy). NOME vira "
                          "o valor da coluna 'method' no CSV de saida.")
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--run-noddi", action="store_true")
    ap.add_argument("--roi-tracts", default=None,
                     help="lista separada por virgula de tratos JHU-ICBM para restringir "
                          "as metricas alem da mascara inteira (ex.: 'FX,CGC,CGH,UF' -- "
                          "relevantes para Alzheimer). Cada trato precisa de um arquivo "
                          "'JHU-ICBM-labels-1mm_warped_s_<TRATO>_<R/L>.nii.gz' (ou sem "
                          "sufixo de lado, ex. FX) na mesma pasta do dwi do sujeito. "
                          "Tratos disponiveis conhecidos: " + ", ".join(
                              f"{k} ({v})" for k, v in JHU_TRACT_LABELS.items()) +
                          ". Sem essa flag, so a metrica de mascara inteira ('whole_mask') "
                          "e calculada (comportamento antigo, inalterado).")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--shard-index", type=int, default=0,
                     help="ver mesmo flag em 06_evaluate_reconstruction.py -- paraleliza por "
                          "SUJEITO dentro do mesmo combo shell/n_level em vez de rodar todos "
                          "em sequencia num job so.")
    ap.add_argument("--shard-count", type=int, default=1,
                     help="numero total de shards (default 1 = sem sharding). Com "
                          "shard-count>1 o CSV final ganha sufixo '.shardIofN' -- junte "
                          "depois com scripts/merge_shard_csvs.py.")
    ap.add_argument("--subjects", default=None,
                     help="lista separada por virgula de 'tag' de sujeito (subject, ou "
                          "subject_session se houver sessao) para restringir o downstream a "
                          "esses sujeitos especificos -- mesma convencao de --subjects em "
                          "scripts/05_reconstruct_rcae.py e 06_evaluate_reconstruction.py. "
                          "Util pra nao rodar DTI/NODDI em sujeitos que nao tem reconstrucao "
                          "'rcae' de qualquer forma (ex.: um checkpoint reconstruido so num "
                          "sujeito via RECON_SUBJECTS pra smoke test).")
    args = ap.parse_args()

    extra_methods = []  # lista de (nome, caminho), parseada de --extra-method NOME=CAMINHO
    for spec in args.extra_method:
        if "=" not in spec:
            sys.exit(f"--extra-method invalido: {spec!r} (esperado NOME=CAMINHO)")
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            sys.exit(f"--extra-method invalido: {spec!r} (NOME e CAMINHO nao podem ser vazios)")
        extra_methods.append((name, path))

    if args.baseline_dir is None and args.rcae_dir is None and not extra_methods:
        sys.exit("Passe pelo menos --baseline-dir, --rcae-dir ou --extra-method")
    if not (0 <= args.shard_index < max(args.shard_count, 1)):
        sys.exit(f"--shard-index ({args.shard_index}) fora do intervalo [0, {args.shard_count})")
    roi_tracts = [t.strip() for t in args.roi_tracts.split(",") if t.strip()] if args.roi_tracts else []

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]
    if args.subjects:
        wanted = {t.strip() for t in args.subjects.split(",") if t.strip()}
        def _tag_of(e):
            return e.subject if not e.session else f"{e.subject}_{e.session}"
        entries = [e for e in entries if _tag_of(e) in wanted]
        found = {_tag_of(e) for e in entries}
        missing = wanted - found
        if missing:
            print(f"[aviso] --subjects pediu {sorted(missing)}, mas nao encontrei no split "
                  f"{args.split!r} do manifesto.", flush=True)
        if not entries:
            sys.exit(f"Nenhum dos sujeitos pedidos em --subjects foi encontrado no split "
                      f"{args.split!r} -- nada a fazer.")
    if args.shard_count > 1:
        entries = entries[args.shard_index::args.shard_count]
        print(f"[shard {args.shard_index}/{args.shard_count}] {len(entries)} sujeitos "
              f"neste shard", flush=True)
    out_dir = Path(args.out_dir)
    rows = []

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
        gt_data, affine, header = load_dwi(e.dwi_path)
        shells = split_shells(bvals, tol=args.shell_tol)
        if args.shell_b not in shells:
            print(f"[aviso] {tag}: shell {args.shell_b} nao encontrada, pulando")
            continue
        b0_mean = gt_data[..., shells[0]].mean(axis=-1)
        mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

        rois = {"whole_mask": mask.astype(bool)}
        if roi_tracts:
            tract_masks = load_roi_masks(e.dwi_path, roi_tracts, base_mask=mask)
            rois.update(tract_masks)

        variants = {"ground_truth": gt_data}
        methods_to_try = [("baseline_sh", args.baseline_dir), ("rcae", args.rcae_dir)] + extra_methods
        for method, recon_dir in methods_to_try:
            if recon_dir is None:
                continue
            sub_dir = Path(recon_dir) / tag / f"shell{int(args.shell_b)}" / f"n{args.n_level}"
            target_idx_path = sub_dir / "target_idx.npy"
            if not target_idx_path.exists():
                print(f"[aviso] sem reconstrucao {method} para {tag}")
                continue
            target_idx = np.load(target_idx_path)
            full = build_full_volume(gt_data, target_idx, recon_dir, tag, args.shell_b, args.n_level)
            if full is not None:
                variants[method] = full

        dti_maps = {}
        for method, vol in variants.items():
            dti_maps[method] = fit_dti(vol, bvals, bvecs, args.shell_b, mask, shell_tol=args.shell_tol)

        noddi_maps = {}
        if args.run_noddi:
            if not e.is_multishell:
                print(f"[aviso] {tag}: sujeito single-shell (so 1 shell nao-zero), "
                      f"NODDI e mal-posto sem >=2 shells -- pulando NODDI para esse sujeito "
                      f"(DTI continua normalmente).")
            else:
                for method, vol in variants.items():
                    noddi_maps[method] = try_fit_noddi(vol, bvals, bvecs, mask,
                                                         out_dir / "noddi_tmp" / tag / method)

        gt_dti = dti_maps["ground_truth"]
        for roi_name, roi_mask in rois.items():
            m = roi_mask
            n_vox = int(m.sum())
            for method in variants:
                if method == "ground_truth":
                    continue
                row = {"subject": e.subject, "method": method, "shell": args.shell_b,
                       "n_level": args.n_level, "roi": roi_name, "n_voxels": n_vox,
                       "acquisition_context": "from_multishell" if e.is_multishell else "native_single_shell"}
                for metric in ("FA", "MD", "AD", "RD"):
                    diff = dti_maps[method][metric][m] - gt_dti[metric][m]
                    row[f"{metric}_mae"] = float(np.nanmean(np.abs(diff)))
                    # correlacao pixel-a-pixel exige pelo menos 2 voxels validos
                    # (ROIs pequenas de trato podem ter poucos voxels)
                    row[f"{metric}_corr"] = float(np.corrcoef(
                        dti_maps[method][metric][m], gt_dti[metric][m])[0, 1]) if n_vox >= 2 else np.nan
                if args.run_noddi and noddi_maps.get(method) and noddi_maps.get("ground_truth"):
                    for metric in ("NDI", "ODI", "ISOVF"):
                        diff = noddi_maps[method][metric][m] - noddi_maps["ground_truth"][metric][m]
                        row[f"{metric}_mae"] = float(np.nanmean(np.abs(diff)))
                        row[f"{metric}_corr"] = float(np.corrcoef(
                            noddi_maps[method][metric][m], noddi_maps["ground_truth"][metric][m])[0, 1]) if n_vox >= 2 else np.nan
                rows.append(row)

        # salva os mapas DTI do sujeito (todas as variantes) para inspecao visual
        import nibabel as nib
        maps_dir = out_dir / "dti_maps" / tag
        maps_dir.mkdir(parents=True, exist_ok=True)
        for method, maps in dti_maps.items():
            for metric, arr in maps.items():
                nib.save(nib.Nifti1Image(arr.astype(np.float32), affine),
                          maps_dir / f"{method}_{metric}.nii.gz")

        print(f"{tag}: DTI ajustado para {list(variants.keys())}; ROIs avaliadas: "
              f"{list(rois.keys())}")

    out_csv = out_dir / f"dti_noddi_metrics_shell{int(args.shell_b)}_n{args.n_level}.csv"
    if args.shard_count > 1:
        out_csv = out_csv.with_name(
            f"{out_csv.stem}.shard{args.shard_index}of{args.shard_count}{out_csv.suffix}")

    if not rows:
        if args.shard_count > 1:
            # mesmo raciocinio de 06_evaluate_reconstruction.py: um shard
            # sem nenhum resultado (sujeito(s) sem baseline/rcae pra essa
            # shell/nivel) e valido, nao erro -- grava CSV vazio em vez de
            # sys.exit, senao o merge_shard_csvs.py fica esperando pra
            # sempre por um shard que nunca vai ter dado.
            print(f"[shard {args.shard_index}/{args.shard_count}] nenhum resultado neste "
                  f"shard -- gravando CSV vazio em {out_csv}", flush=True)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["subject", "method", "shell", "n_level", "roi", "n_voxels",
                                   "acquisition_context"]).to_csv(out_csv, index=False)
            return
        sys.exit("Nenhum resultado -- confira os diretorios de reconstrucao e a shell/nivel pedidos")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Metricas downstream salvas em", out_csv)
    print(df.groupby(["roi", "method"])[[c for c in df.columns if c.endswith("_mae")]].mean())


if __name__ == "__main__":
    main()