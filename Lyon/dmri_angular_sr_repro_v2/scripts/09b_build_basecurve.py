#!/usr/bin/env python3
"""
Etapa 9b (diagnostico auxiliar, nao faz parte do pipeline principal):
monta a curva "erro do baseline SH vs. numero de direcoes de entrada"
(--levels) para UMA shell, restringindo a amostra a sujeitos com uma
contagem FIXA de direcoes disponiveis nessa shell (default: o maximo
observado, ex. 64 para b=1000 no seu dataset) -- ver discussao no chat/
protocolo: sem essa restricao, sujeitos com menos direcoes saem da amostra
conforme --levels sobe, e qualquer tendencia na curva passa a misturar
"efeito do numero de direcoes de entrada" com "mudanca de quem esta na
amostra" (composicao de sujeitos). Fixando a amostra, cada ponto da curva
usa exatamente o mesmo grupo de sujeitos, so variando quantas direcoes de
entrada o baseline recebeu.

So avalia 'baseline_sh' (nao RCAE/RRIN) -- pensado para a investigacao
"como o baseline se comporta ao longo de N direcoes" feita numa pasta de
reconstrucao separada (ver scripts/02_subsample_directions.py --out-dir
<work_dir>/subsampling_basecurve e scripts/03_baseline_sh_interpolation.py
--scheme-dir/--out-dir apontando pra la), sem tocar no
subsampling/baseline_recon canonico usado por RCAE/RRIN.

Metricas: mesmas de scripts/06_evaluate_reconstruction.py (PSNR/SSIM por
volume-direcao held-out; NMSE/RMSE/ACC agregados por sujeito) -- a logica
de avaliacao foi reaproveitada aqui (nao importada, ja que o nome do
arquivo original comeca com digito e nao e importavel como modulo).

Uso (sem sharding, roda todos os sujeitos num job so):
    python scripts/09b_build_basecurve.py \
        --manifest work_dir/manifest.csv \
        --baseline-dir work_dir/baseline_recon \
        --shell-b 1000 \
        --levels 6 10 16 20 24 32 48 54 \
        --out-csv work_dir/basecurve_metrics_shell1000.csv

Alem do CSV completo (uma linha por volume/sujeito/nivel), tambem escreve
uma TABELA RESUMIDA (uma linha por n_level, com media/desvio/contagem de
NMSE/RMSE/ACC/PSNR/SSIM) em <out-csv com _summary antes do .csv> -- ex.:
work_dir/basecurve_metrics_shell1000_summary.csv. So acontece quando NAO
esta em modo shard (--shard-count 1, o default) -- rodando em shards, a
tabela resumida so pode ser calculada depois de juntar todos os shards
(ver scripts/09c_merge_basecurve.py).

--exact-directions (default: None -- calculado automaticamente como o
MAXIMO de n_available observado entre os sujeitos do split, na shell
pedida) fixa manualmente o numero de direcoes exigido, se voce quiser um
valor diferente do maximo (ex. testar restringindo a 35 em vez de 64, se
quiser incluir mais sujeitos ao custo de nao cobrir os niveis mais altos).

--shard-index / --shard-count (default 0/1 -- sem sharding): divide a
AMOSTRA FIXADA (nao a lista de sujeitos bruta) em --shard-count fatias
intercaladas e processa so a fatia --shard-index nesta execucao -- pensado
pra rodar varios shards em paralelo via `sbatch --array` (ver
slurm/09b_build_basecurve.sh), acelerando o job ao dividir os sujeitos
entre varias tasks da fila em vez de um job sequencial so. Cada shard
escreve seu proprio CSV parcial (sufixo .shard<indice>.csv); depois que
TODOS os shards terminarem, rode scripts/09c_merge_basecurve.py (ou
slurm/09c_merge_basecurve.sh) pra juntar tudo num CSV final + tabela
resumida.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_dwi
from utils.metrics import psnr, ssim3d, nmse, rmse, angular_correlation_coefficient


def evaluate_subject_baseline(recon_dir: Path, tag: str, shell_b: float, n_level: int,
                               gt_data: np.ndarray, subject: str, acquisition_context: str):
    """Mesma logica de evaluate_subject_method em 06_evaluate_reconstruction.py,
    especializada para 'baseline_sh' (unico metodo avaliado aqui)."""
    sub_dir = recon_dir / tag / f"shell{int(shell_b)}" / f"n{n_level}"
    recon_path = sub_dir / "recon_target.nii.gz"
    if not recon_path.exists():
        return None

    import nibabel as nib
    recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
    target_idx = np.load(sub_dir / "target_idx.npy")
    mask = np.load(sub_dir.parent / "mask.npy")

    gt_target = gt_data[..., target_idx]

    rows = []
    for t in range(target_idx.shape[0]):
        p = recon[..., t]
        g = gt_target[..., t]
        rows.append({
            "subject": subject, "method": "baseline_sh", "shell": shell_b, "n_level": n_level,
            "acquisition_context": acquisition_context,
            "target_volume_idx": int(target_idx[t]), "metric_scope": "per_volume",
            "psnr": psnr(p, g, mask=mask), "ssim": ssim3d(p, g, mask=mask),
        })

    m = mask.astype(bool)
    nmse_val = nmse(recon[m], gt_target[m])
    rmse_val = rmse(recon[m], gt_target[m])
    acc = angular_correlation_coefficient(recon[m], gt_target[m])
    rows.append({
        "subject": subject, "method": "baseline_sh", "shell": shell_b, "n_level": n_level,
        "acquisition_context": acquisition_context,
        "target_volume_idx": -1, "metric_scope": "aggregate",
        "nmse": nmse_val, "rmse": rmse_val, "acc_mean": float(np.nanmean(acc)),
        "acc_std": float(np.nanstd(acc)),
    })
    return rows


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela resumida (uma linha por n_level) a partir do CSV completo --
    reaproveitada tanto no modo sem-shard (aqui) quanto no merge de shards
    (scripts/09c_merge_basecurve.py), pra garantir que os dois caminhos
    produzam exatamente a mesma tabela. Importante: NAO da pra soh tirar a
    media das medias de cada shard (o desvio-padrao, em especial, ficaria
    errado) -- por isso a tabela resumida sempre e calculada sobre o
    conjunto COMPLETO de sujeitos, nunca por shard."""
    agg = df[df["metric_scope"] == "aggregate"]
    pv = df[df["metric_scope"] == "per_volume"]

    summary_agg = agg.groupby("n_level")[["nmse", "rmse", "acc_mean"]].agg(["mean", "std", "count"])
    summary_pv = pv.groupby("n_level")[["psnr", "ssim"]].agg(["mean", "std"])
    summary = summary_agg.join(summary_pv)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline-dir", required=True,
                     help="ex.: work_dir/baseline_recon")
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--levels", type=int, nargs="+", required=True,
                     help="mesma lista passada a scripts/02_subsample_directions.py "
                          "pra gerar o esquema alternativo (ex.: 6 10 16 20 24 32 48 54)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--exact-directions", type=int, default=None,
                     help="numero de direcoes exigido na shell pedida para o sujeito "
                          "entrar na amostra (default: calculado automaticamente como o "
                          "MAXIMO observado entre os sujeitos do split nessa shell)")
    ap.add_argument("--shard-index", type=int, default=0,
                     help="indice desta shard (0-based). Default 0 (sem sharding).")
    ap.add_argument("--shard-count", type=int, default=1,
                     help="numero total de shards. Default 1 (roda tudo num job so).")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    if args.split != "all":
        entries = [e for e in entries if e.split == args.split]
    if not entries:
        sys.exit(f"Nenhum sujeito no split {args.split!r} -- confira o manifesto")

    # 1a passada: conta direcoes disponiveis na shell pedida, por sujeito --
    # usa o resumo ja calculado no manifesto (SubjectEntry.n_dirs_for_shell,
    # ver utils/manifest.py) em vez de reabrir bval/bvec de cada sujeito.
    n_available_by_tag = {}
    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        n_dirs = e.n_dirs_for_shell(args.shell_b, tol=args.shell_tol)
        if n_dirs is None:
            continue
        n_available_by_tag[tag] = n_dirs

    if not n_available_by_tag:
        sys.exit(f"Nenhum sujeito do split {args.split!r} tem a shell {args.shell_b} -- "
                  f"confira --shell-b/--shell-tol")

    exact = args.exact_directions
    if exact is None:
        exact = max(n_available_by_tag.values())
        print(f"[auto] --exact-directions nao passado -- usando o maximo observado: "
              f"{exact} direcoes", flush=True)

    kept_tags = {tag for tag, n in n_available_by_tag.items() if n == exact}
    dropped = len(n_available_by_tag) - len(kept_tags)
    print(f"Amostra fixada em {len(kept_tags)} sujeito(s) com exatamente {exact} "
          f"direcoes na shell {args.shell_b} (descartados {dropped} sujeito(s) do split "
          f"{args.split!r} com contagem diferente).", flush=True)
    if not kept_tags:
        sys.exit("Nenhum sujeito sobrou com essa contagem exata de direcoes -- confira "
                  "--exact-directions ou a distribuicao real (ver aviso [auto] acima)")

    # Ordenacao deterministica (mesmo resultado sempre, independente de
    # hash de set) -- essencial pra sharding: cada shard precisa fatiar a
    # MESMA lista ordenada, senao shards diferentes poderiam processar o
    # mesmo sujeito duas vezes (ou nenhum) por pura sorte de ordem de set.
    kept_tags_sorted = sorted(kept_tags)
    if args.shard_count > 1:
        shard_tags = set(kept_tags_sorted[args.shard_index::args.shard_count])
        print(f"[shard] {args.shard_index}/{args.shard_count} -- processando "
              f"{len(shard_tags)} de {len(kept_tags_sorted)} sujeito(s) da amostra "
              f"fixada nesta shard.", flush=True)
    else:
        shard_tags = set(kept_tags_sorted)

    if not shard_tags:
        print("Nenhum sujeito sobrou pra esta shard (mais shards do que sujeitos na "
              "amostra?) -- nada a fazer, encerrando sem erro.", flush=True)
        return

    recon_dir = Path(args.baseline_dir)
    all_rows = []
    n_subjects_with_data = 0
    n_done = 0
    t_start = time.time()

    # Nome do CSV desta execucao -- se estiver em modo shard, grava num
    # arquivo por-shard (nao no --out-csv final) pra nao ter varias tasks
    # do array escrevendo no mesmo arquivo ao mesmo tempo.
    out_csv = Path(args.out_csv)
    if args.shard_count > 1:
        this_csv = out_csv.with_name(f"{out_csv.stem}.shard{args.shard_index}{out_csv.suffix}")
    else:
        this_csv = out_csv

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        if tag not in shard_tags:
            continue
        gt_data, _affine, _header = load_dwi(e.dwi_path)
        acq_ctx = "from_multishell" if e.is_multishell else "native_single_shell"
        found_any = False
        for n_level in args.levels:
            rows = evaluate_subject_baseline(recon_dir, tag, args.shell_b, n_level,
                                              gt_data, e.subject, acq_ctx)
            if rows is not None:
                all_rows.extend(rows)
                found_any = True
        if found_any:
            n_subjects_with_data += 1
        else:
            print(f"[aviso] {tag}: esta na amostra fixada mas nao tem nenhuma reconstrucao "
                  f"baseline em {recon_dir} para shell={args.shell_b} -- confira se "
                  f"scripts/03_baseline_sh_interpolation.py ja rodou pra esse sujeito/shell.")

        # log de progresso -- sem isso, com dezenas de sujeitos x varios
        # n_level (cada um calculando SSIM/PSNR/NMSE/ACC sobre volumes 3D
        # inteiros), o job pode ficar minutos sem imprimir nada, o que fica
        # indistinguivel de travado no .out do slurm.
        n_done += 1
        elapsed = time.time() - t_start
        rate = elapsed / n_done
        eta_s = rate * (len(shard_tags) - n_done)
        print(f"[progresso] {n_done}/{len(shard_tags)} sujeitos desta shard processados "
              f"({elapsed:.0f}s decorridos, ~{rate:.1f}s/sujeito, ETA ~{eta_s/60:.1f} min)",
              flush=True)

        # salva um CSV PARCIAL a cada 10 sujeitos -- se o job morrer no meio
        # (preempcao, timeout), voce nao perde tudo: o parcial fica em
        # <arquivo-desta-shard>.partial.csv com o que ja foi calculado ate ali.
        if n_done % 10 == 0:
            partial_path = this_csv.with_suffix(".partial.csv")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_rows).to_csv(partial_path, index=False)

    if not all_rows:
        sys.exit("Nenhum resultado -- confira --baseline-dir e se a reconstrucao ja rodou "
                  "para os n_level pedidos")

    df = pd.DataFrame(all_rows)
    this_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(this_csv, index=False)
    print(f"\n{n_subjects_with_data}/{len(shard_tags)} sujeitos desta shard tinham "
          f"reconstrucao disponivel. Metricas salvas em {this_csv}")

    if args.shard_count > 1:
        print(f"\nModo shard -- rode scripts/09c_merge_basecurve.py (ou "
              f"slurm/09c_merge_basecurve.sh) depois que TODAS as {args.shard_count} "
              f"shards terminarem, pra juntar tudo em {out_csv} + tabela resumida.")
        return

    summary_df = compute_summary(df)
    summary_csv = out_csv.with_name(f"{out_csv.stem}_summary{out_csv.suffix}")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Tabela resumida (uma linha por n_level) salva em {summary_csv}")

    print("\nCurva (media entre sujeitos, amostra fixa) -- erro do baseline_sh vs. n_level:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()