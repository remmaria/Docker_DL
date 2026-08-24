#!/usr/bin/env python3
"""
Etapa 6: avalia as reconstrucoes (baseline SH e/ou RCAE) contra o ground
truth (as direcoes held-out que existem no dado original completo), para
cada sujeito, shell e nivel de subamostragem.

Uso:
    python scripts/06_evaluate_reconstruction.py \
        --manifest work_dir/manifest.csv \
        --baseline-dir work_dir/baseline_recon \
        --rcae-dir work_dir/rcae_recon \
        --shell-b 1000 --n-level 10 \
        --out-csv work_dir/metrics/signal_metrics_shell1000_n10.csv

--baseline-dir e --rcae-dir sao opcionais individualmente (pode rodar so
para um dos dois), mas pelo menos um precisa ser passado.

Metricas por volume-direcao (dentro da mascara): PSNR, SSIM.
Metricas por voxel agregadas nas direcoes: NMSE, RMSE.
ACC: tratado aqui como a correlacao angular entre o vetor de sinal nas
direcoes held-out (reconstruido vs. real) por voxel -- um proxy direto de
"quao bem a forma angular do sinal foi recuperada", nao um ACC classico
sobre coeficientes SH de ordem alta (que exigiria muito mais direcoes por
voxel do que tipicamente sobra no conjunto held-out). Para um ACC no
sentido classico (comparando ODFs), ajuste SH nos dados reconstruidos
completos (entrada + predito) e no dado original completo, depois use
utils.metrics.angular_correlation_coefficient nos coeficientes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_dwi
from utils.metrics import psnr, ssim3d, nmse, rmse, angular_correlation_coefficient


def evaluate_subject_method(recon_dir: Path, tag: str, shell_b: float, n_level: int,
                             gt_data: np.ndarray, method: str, subject: str,
                             acquisition_context: str):
    sub_dir = recon_dir / tag / f"shell{int(shell_b)}" / f"n{n_level}"
    recon_path = sub_dir / "recon_target.nii.gz"
    if not recon_path.exists():
        return None

    import nibabel as nib
    recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
    target_idx = np.load(sub_dir / "target_idx.npy")
    # mask.npy fica um nivel acima (compartilhada entre todos os n_level dessa shell)
    mask = np.load(sub_dir.parent / "mask.npy")

    gt_target = gt_data[..., target_idx]

    rows = []
    for t in range(target_idx.shape[0]):
        p = recon[..., t]
        g = gt_target[..., t]
        rows.append({
            "subject": subject, "method": method, "shell": shell_b, "n_level": n_level,
            "acquisition_context": acquisition_context,
            "target_volume_idx": int(target_idx[t]), "metric_scope": "per_volume",
            "psnr": psnr(p, g, mask=mask), "ssim": ssim3d(p, g, mask=mask),
        })

    m = mask.astype(bool)
    nmse_val = nmse(recon[m], gt_target[m])
    rmse_val = rmse(recon[m], gt_target[m])
    acc = angular_correlation_coefficient(recon[m], gt_target[m])
    rows.append({
        "subject": subject, "method": method, "shell": shell_b, "n_level": n_level,
        "acquisition_context": acquisition_context,
        "target_volume_idx": -1, "metric_scope": "aggregate",
        "nmse": nmse_val, "rmse": rmse_val, "acc_mean": float(np.nanmean(acc)),
        "acc_std": float(np.nanstd(acc)),
    })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--rcae-dir", default=None)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--shard-index", type=int, default=0,
                     help="indice (0-based) deste shard, para paralelizar por SUJEITO "
                          "dentro do mesmo combo shell/n_level (ex.: via SLURM array) em vez "
                          "de rodar os N sujeitos do split em sequencia num unico job. Use "
                          "junto com --shard-count -- ver slurm/05_evaluate_and_downstream.sh "
                          "para o modo que decide automaticamente entre 'array = combo' e "
                          "'array = shard de sujeitos'.")
    ap.add_argument("--shard-count", type=int, default=1,
                     help="numero total de shards (default 1 = sem sharding, roda todos os "
                          "sujeitos do split num job so). Com shard-count>1 o --out-csv "
                          "ganha um sufixo '.shardIofN' automaticamente (evita 2+ processos "
                          "escrevendo por cima do mesmo arquivo ao mesmo tempo) -- junte os "
                          "shards depois com scripts/merge_shard_csvs.py.")
    ap.add_argument("--subjects", default=None,
                     help="lista separada por virgula de 'tag' de sujeito (subject, ou "
                          "subject_session se houver sessao) para restringir a avaliacao a "
                          "esses sujeitos especificos, em vez do split inteiro -- mesma "
                          "convencao de --subjects em scripts/05_reconstruct_rcae.py. Util "
                          "pra avaliar rapido um checkpoint reconstruido com "
                          "--subjects/RECON_SUBJECTS restrito, sem esperar o resto do split "
                          "(que nao tem reconstrucao 'rcae' mesmo) ser processado a toa.")
    args = ap.parse_args()

    if args.baseline_dir is None and args.rcae_dir is None:
        sys.exit("Passe pelo menos --baseline-dir ou --rcae-dir")
    if not (0 <= args.shard_index < max(args.shard_count, 1)):
        sys.exit(f"--shard-index ({args.shard_index}) fora do intervalo [0, {args.shard_count})")

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
        out_csv_path = Path(args.out_csv)
        args.out_csv = str(out_csv_path.with_name(
            f"{out_csv_path.stem}.shard{args.shard_index}of{args.shard_count}{out_csv_path.suffix}"))
    all_rows = []
    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        gt_data, _, _ = load_dwi(e.dwi_path)

        for method, recon_dir in (("baseline_sh", args.baseline_dir), ("rcae", args.rcae_dir)):
            if recon_dir is None:
                continue
            acq_ctx = "from_multishell" if e.is_multishell else "native_single_shell"
            rows = evaluate_subject_method(Path(recon_dir), tag, args.shell_b, args.n_level,
                                            gt_data, method, e.subject, acq_ctx)
            if rows is None:
                print(f"[aviso] sem reconstrucao {method} para {tag}")
                continue
            all_rows.extend(rows)

    if not all_rows:
        if args.shard_count > 1:
            # NAO usa sys.exit aqui quando esta sharded -- um shard pode
            # legitimamente nao ter NENHUM resultado (ex.: por acaso so tem
            # sujeito(s) sem essa shell/nivel, entao nem baseline nem rcae
            # tem reconstrucao pra avaliar -- nao e erro, e so "esse pedaco
            # do split nao se aplica"). Antes isso matava o job com erro, o
            # shard nunca escrevia CSV, e o merge_shard_csvs.py ficava
            # esperando pra sempre por um shard que NUNCA vai ter dado (viu
            # isso na pratica: 18 dos 145 sujeitos nao tem shell b=1000, e
            # alguns shards do sharding por sujeito cairam so nesse grupo).
            # Grava um CSV vazio (0 linhas, ainda um arquivo valido) pra
            # sinalizar "rodou com sucesso, sem dado mesmo" -- o merge
            # inclui ele normalmente (0 linhas contribuidas).
            print(f"[shard {args.shard_index}/{args.shard_count}] nenhum resultado neste "
                  f"shard (sujeito(s) sem baseline/rcae pra essa shell/nivel, provavelmente "
                  f"nao tem essa shell) -- gravando CSV vazio em {args.out_csv}", flush=True)
            df = pd.DataFrame(columns=["subject", "method", "shell", "n_level",
                                        "acquisition_context", "target_volume_idx",
                                        "metric_scope"])
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out_csv, index=False)
            return
        sys.exit("Nenhum resultado para avaliar -- confira os diretorios de reconstrucao")

    df = pd.DataFrame(all_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Metricas salvas em", args.out_csv)
    agg = df[df.metric_scope == "aggregate"]
    if not agg.empty:
        print(agg.groupby("method")[["nmse", "rmse", "acc_mean"]].mean())


if __name__ == "__main__":
    main()