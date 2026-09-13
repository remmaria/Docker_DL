#!/usr/bin/env python3
"""
Etapa 2e (calibracao, opcional): responde a pergunta em aberto na docstring
de utils.gradients.electrostatic_repulsion_sampling -- "n_starts=80 foi
validado para N=16; revalidar para outros N, o espaco de busca cresce com
N" -- e ao gap pequeno (mediana <1.3 graus) observado no min_angle_deg do
electrostatic vs. fps em N=32/48/54 (ver subsampling_quality.csv real vs.
subsampling_quality_fps.csv): sera que aumentar n_starts fecha esse gap, ou
o gap e um efeito do regime quase-saturado (poucas direcoes de fora pra
trocar, quando N esta perto do total disponivel) que n_starts maior nao
resolve?

Para cada combinacao (shell, n_level) pedida, amostra alguns sujeitos do
manifesto que tem aquela shell com direcoes suficientes, roda
electrostatic_repulsion_sampling varias vezes com n_starts crescente (grade
fixa), mede a energia eletrostatica final (a mesma quantidade que o metodo
otimiza -- quanto menor, melhor) e o tempo gasto, e recomenda o MENOR
n_starts cuja energia mediana fica dentro de --tol-rel (default 1%) da
energia mediana obtida no MAIOR n_starts testado (o "assintota" pratico da
grade). Isso da uma resposta objetiva por nivel: "80 e suficiente aqui" ou
"precisa de mais" -- em vez de usar o mesmo valor pra todo N so porque foi
o que deu certo pra N=16.

So LE bval/bvec dos sujeitos amostrados (nao precisa de GPU) mas pode ser
lento (grade de n_starts x niveis x shells x sujeitos, cada combinacao roda
o multi-start + busca local do zero) -- rode num node CPU com --time
generoso, ou restrinja --shells/--levels aos casos que realmente importam
(ex.: so os N altos onde o gap foi observado) em vez da grade completa.

Uso (exemplo focado nos N altos, onde o gap fps-vs-electrostatic apareceu):
    python scripts/02e_calibrate_electrostatic_nstarts.py \
        --manifest work_dir/manifest.csv \
        --shells 1000 1500 2000 \
        --levels 32 48 54 \
        --n-subjects 5 \
        --n-starts-grid 10 20 40 80 160 320 \
        --out-csv work_dir/electrostatic_nstarts_calibration.csv
"""
import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, split_shells, electrostatic_repulsion_sampling, direction_set_quality


def _pick_subjects_for_combo(entries, shell, n_level, n_subjects, rng, shell_tol):
    candidates = [e for e in entries if e.has_shell(shell, tol=shell_tol)
                  and (e.n_dirs_for_shell(shell, tol=shell_tol) or 0) >= n_level]
    if not candidates:
        return []
    k = min(n_subjects, len(candidates))
    return rng.sample(candidates, k)


def _shell_bvecs(bvals, bvecs, target_b, shell_tol):
    shells = split_shells(bvals, tol=shell_tol)
    keys = [k for k in shells.keys() if k != 0]
    if not keys:
        return None
    best_key = min(keys, key=lambda k: abs(k - target_b))
    if abs(best_key - target_b) > shell_tol:
        return None
    return bvecs[shells[best_key]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shells", type=float, nargs="+",
                     default=[500.0, 700.0, 750.0, 1000.0, 1500.0, 2000.0])
    ap.add_argument("--levels", type=int, nargs="+",
                     default=[6, 10, 16, 20, 24, 32, 48, 54])
    ap.add_argument("--n-subjects", type=int, default=5,
                     help="quantos sujeitos amostrar por combinacao (shell, n_level)")
    ap.add_argument("--n-starts-grid", type=int, nargs="+",
                     default=[10, 20, 40, 80, 160])
    ap.add_argument("--max-local-iter", type=int, default=150,
                     help="fixo pra todos os pontos da grade -- so n_starts e calibrado aqui "
                          "(max_local_iter ja foi validado como robusto ate N=16, ver notas "
                          "do projeto; convergencia tipica em ~11 iteracoes)")
    ap.add_argument("--tol-rel", type=float, default=0.01,
                     help="tolerancia relativa (default 1%%) de energia acima da assintota da "
                          "grade pra considerar um n_starts 'suficiente'")
    ap.add_argument("--shell-tol", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default=None,
                     help="se passado, grava a tabela agregada (shell,n_level,n_starts,"
                          "energia_mediana,tempo_mediano_s,n_sujeitos)")
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    rng = random.Random(args.seed)
    n_starts_grid = sorted(set(args.n_starts_grid))
    max_ns = n_starts_grid[-1]

    print(f"[info] {len(entries)} sujeitos no manifesto, grade de n_starts={n_starts_grid}, "
          f"max_local_iter={args.max_local_iter} (fixo), tol_rel={args.tol_rel*100:.1f}%\n")

    bvec_cache = {}  # subject -> (bvals, bvecs) carregado uma vez, reusado entre combos

    # agg[(shell, n_level, n_starts)] = lista de (energia, tempo_s)
    agg = defaultdict(list)
    raw_rows = []

    for shell in args.shells:
        for n_level in args.levels:
            subjects = _pick_subjects_for_combo(entries, shell, n_level, args.n_subjects,
                                                 rng, args.shell_tol)
            if not subjects:
                print(f"[shell={shell} N={n_level}] nenhum sujeito com direcoes suficientes, pulado")
                continue
            if len(subjects) < args.n_subjects:
                print(f"[shell={shell} N={n_level}] so {len(subjects)}/{args.n_subjects} "
                      f"sujeito(s) disponivel(is) com N>={n_level} nessa shell")

            for e in subjects:
                key = e.subject
                if key not in bvec_cache:
                    try:
                        bvec_cache[key] = load_bval_bvec(e.bval_path, e.bvec_path)
                    except Exception as exc:
                        print(f"  [aviso] falha ao carregar {key}: {exc}, pulando esse sujeito")
                        bvec_cache[key] = None
                loaded = bvec_cache[key]
                if loaded is None:
                    continue
                bvals, bvecs = loaded
                shell_bvecs = _shell_bvecs(bvals, bvecs, shell, args.shell_tol)
                if shell_bvecs is None or shell_bvecs.shape[0] < n_level:
                    continue

                for ns in n_starts_grid:
                    t0 = time.perf_counter()
                    sel = electrostatic_repulsion_sampling(
                        shell_bvecs, n_level, n_starts=ns,
                        max_local_iter=args.max_local_iter, seed=args.seed)
                    elapsed = time.perf_counter() - t0
                    energy = direction_set_quality(shell_bvecs[sel])["electrostatic_energy"]
                    agg[(shell, n_level, ns)].append((energy, elapsed))
                    raw_rows.append((shell, n_level, key, shell_bvecs.shape[0], ns, energy, elapsed))

    if not agg:
        print("\n[erro] nenhuma combinacao (shell, n_level) produziu dados -- confira "
              "--shells/--levels/--n-subjects contra o manifesto")
        return

    # ---- agregacao e recomendacao por (shell, n_level) ----
    combos = sorted({(s, n) for (s, n, _ns) in agg.keys()})
    print("\n=== Recomendacao de n_starts por (shell, n_level) ===")
    print(f"{'shell':>7} {'N':>4} {'n_sujeitos':>10} {'n_starts_rec':>13} "
          f"{'energia_rec':>12} {'energia_max_grade':>18} {'gap_%':>7} "
          f"{'tempo_rec_s':>12} {'tempo_max_grade_s':>18}")

    summary_rows = []
    for (shell, n_level) in combos:
        per_ns = {}
        for ns in n_starts_grid:
            vals = agg.get((shell, n_level, ns))
            if not vals:
                continue
            energies = [v[0] for v in vals]
            times = [v[1] for v in vals]
            per_ns[ns] = (median(energies), median(times), len(vals))
        if not per_ns:
            continue
        tested_ns = sorted(per_ns.keys())
        asymptote_ns = tested_ns[-1]
        asymptote_energy = per_ns[asymptote_ns][0]
        threshold = asymptote_energy * (1.0 + args.tol_rel)

        recommended_ns = asymptote_ns
        for ns in tested_ns:
            if per_ns[ns][0] <= threshold:
                recommended_ns = ns
                break

        rec_energy, rec_time, n_sub = per_ns[recommended_ns]
        max_energy, max_time, _ = per_ns[asymptote_ns]
        gap_pct = 100.0 * (rec_energy - max_energy) / max_energy if max_energy else 0.0

        print(f"{shell:>7.0f} {n_level:>4d} {n_sub:>10d} {recommended_ns:>13d} "
              f"{rec_energy:>12.2f} {max_energy:>18.2f} {gap_pct:>6.2f}% "
              f"{rec_time:>12.4f} {max_time:>18.4f}")

        summary_rows.append({
            "shell": shell, "n_level": n_level, "n_subjects": n_sub,
            "recommended_n_starts": recommended_ns,
            "energy_at_recommended": rec_energy,
            "energy_at_max_tested": max_energy,
            "gap_pct": gap_pct,
            "time_at_recommended_s": rec_time,
            "time_at_max_tested_s": max_time,
        })

    # destaca onde o default atual (80) fica aquem do recomendado (i.e. caso
    # em que valeria a pena SUBIR n_starts pra fechar o gap observado vs. fps)
    print("\n=== Onde o default atual (n_starts=80) pode estar insuficiente ===")
    any_flag = False
    for row in summary_rows:
        if row["recommended_n_starts"] > 80:
            any_flag = True
            print(f"  shell={row['shell']:.0f} N={row['n_level']}: recomendado "
                  f"n_starts={row['recommended_n_starts']} (>80) pra ficar dentro de "
                  f"{args.tol_rel*100:.1f}% da energia assintotica testada")
    if not any_flag:
        print("  nenhum -- n_starts=80 ja fica dentro da tolerancia em todas as combinacoes "
              "testadas (grade maxima testada: "
              f"{max_ns}); se o gap vs. fps persistir em N alto, e provavelmente efeito do "
              "regime quase-saturado (poucas direcoes de fora pra trocar), nao de "
              "sub-otimizacao por falta de sementes")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\n[info] tabela de recomendacoes gravada em: {args.out_csv}")


if __name__ == "__main__":
    main()