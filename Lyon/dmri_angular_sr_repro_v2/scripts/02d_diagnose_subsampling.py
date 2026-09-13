#!/usr/bin/env python3
"""
Etapa 2d (diagnostico, opcional): audita a QUALIDADE GEOMETRICA dos
esquemas de subamostragem (.npz) ja gerados por scripts/02_subsample_directions.py
-- angulo minimo entre direcoes escolhidas, angulo medio ao vizinho mais
proximo, |centro de massa| e energia eletrostatica do subconjunto de
ENTRADA de cada (sujeito, shell, n_level) -- ver
utils.gradients.direction_set_quality para a definicao de cada metrica.

Motivacao: nem "fps" nem "electrostatic" (ver scripts/02_subsample_directions.py
--method) imprimem esses diagnosticos por sujeito durante a geracao (rodar
o dataset inteiro ja e lento o bastante sem I/O extra por sujeito) -- este
script le os .npz DEPOIS de prontos (rapido, sem GPU/volume) e agrega numa
tabela resumo por (shell,n_level), pra confirmar que a dispersao angular
ficou boa em todo o dataset, nao so no sujeito de teste que voce olhou na
hora. Tambem serve para comparar dois metodos/diretorios lado a lado (ex.:
fps vs electrostatic) rodando o script duas vezes com --scheme-dir
diferente e comparando as tabelas.

So o subconjunto de ENTRADA (input_idx) e avaliado por padrao -- e o que
importa pra dispersao angular da amostragem em si; use --include-target
para tambem reportar as mesmas metricas do lado do alvo/held-out (menos
interessante geometricamente, mas disponivel).

Uso:
    python scripts/02d_diagnose_subsampling.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --out-csv work_dir/subsampling_quality.csv
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, direction_set_quality


def _enumerate_combos(scheme_npz):
    combos = []
    for key in scheme_npz.files:
        if key.endswith("__input"):
            base = key[: -len("__input")]
            shell_str, level_str = base.rsplit("__", 1)
            combos.append((shell_str, int(level_str)))
    return combos


def _percentiles(values, ps=(10, 50, 90)):
    if not values:
        return {p: float("nan") for p in ps}
    arr = np.asarray(values, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--out-csv", default=None,
                     help="se passado, grava 1 linha por (sujeito,shell,n_level) com as 4 "
                          "metricas -- alem do resumo agregado impresso no console")
    ap.add_argument("--shells", type=float, nargs="*", default=None,
                     help="filtra so essas shells (default: todas presentes nos .npz)")
    ap.add_argument("--levels", type=int, nargs="*", default=None,
                     help="filtra so esses n_levels (default: todos presentes nos .npz)")
    ap.add_argument("--include-target", action="store_true",
                     help="tambem calcula/reporta as metricas do subconjunto ALVO "
                          "(target_idx), nao so o de entrada")
    ap.add_argument("--limit", type=int, default=None,
                     help="so os N primeiros sujeitos do manifesto (previa rapida)")
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    if args.limit:
        entries = entries[: args.limit]
    scheme_dir = Path(args.scheme_dir)

    # {(shell_str, n_level): {"input": {"min_angle_deg": [...], ...}, "target": {...}}}
    agg = defaultdict(lambda: {"input": defaultdict(list), "target": defaultdict(list)})
    rows = []
    n_missing_npz = 0
    n_subjects_seen = 0
    # methods_seen[(shell_str, n_level)] = {"fps", "electrostatic", ...} -- qual(is)
    # metodo(s) o campo de proveniencia "__method" (2026-09, ver scripts/
    # 02_subsample_directions.py) registrou pra cada combo. .npz mais antigos
    # (gerados antes desse campo existir) nao tem essa chave -- reportados
    # separadamente como "desconhecido (npz antigo)" em vez de assumir "fps"
    # silenciosamente.
    methods_seen = defaultdict(set)

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        npz_path = scheme_dir / f"{tag}_scheme.npz"
        if not npz_path.exists():
            n_missing_npz += 1
            continue
        n_subjects_seen += 1
        scheme = np.load(npz_path)
        combos = _enumerate_combos(scheme)
        if args.shells is not None:
            combos = [c for c in combos if float(c[0]) in args.shells]
        if args.levels is not None:
            combos = [c for c in combos if c[1] in args.levels]
        if not combos:
            continue

        bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)

        for shell_str, n_level in combos:
            key = f"{shell_str}__{n_level}"
            method_key = f"{key}__method"
            if method_key in scheme.files:
                method_str = str(scheme[method_key])
            else:
                method_str = "desconhecido (npz antigo, sem campo __method)"
            methods_seen[(shell_str, n_level)].add(method_str)
            sides = ["input"] + (["target"] if args.include_target else [])
            for side in sides:
                idx = scheme[f"{key}__{side}"]
                if len(idx) < 2:
                    continue  # nao da pra medir angulo entre pares com <2 direcoes
                q = direction_set_quality(bvecs[idx])
                for metric, val in q.items():
                    agg[(shell_str, n_level)][side][metric].append(val)
                if args.out_csv:
                    row = {"subject": e.subject, "session": e.session, "shell": shell_str,
                           "n_level": n_level, "side": side, "n_directions": len(idx),
                           "method": method_str}
                    row.update(q)
                    rows.append(row)

    if n_missing_npz:
        print(f"[aviso] {n_missing_npz} sujeito(s) do manifesto sem .npz em {scheme_dir} "
              f"(nao gerados ainda ou shard nao completado) -- ignorados neste resumo")
    print(f"[info] {n_subjects_seen} sujeito(s) com .npz encontrados e lidos\n")

    # ---- confirmacao de qual --method gerou cada combo (ver discussao no chat:
    # antes disso nao dava pra saber depois do fato se um subsampling/ foi
    # gerado com fps [o default do script] ou electrostatic [o recomendado
    # a partir de 2026-09] sem ainda ter o log original da etapa 2 a mao) ----
    print("=== metodo usado por (shell, n_level), conforme campo __method gravado no .npz ===")
    any_mixed = False
    for (shell_str, n_level), methods in sorted(methods_seen.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        tag = ", ".join(sorted(methods))
        if len(methods) > 1:
            any_mixed = True
            tag += "  <-- MISTURADO! sujeitos diferentes usaram metodos diferentes nesse combo"
        print(f"  shell={shell_str} n={n_level}: {tag}")
    if any_mixed:
        print("\n[ATENCAO] pelo menos um combo (shell,n_level) tem sujeitos com --method "
              "diferentes no MESMO subsampling/ -- provavelmente uma rodada antiga (fps, "
              "o default) foi complementada depois por uma rodada nova (electrostatic, ou "
              "vice-versa) sem regenerar tudo. Isso mistura os dois metodos no MESMO "
              "conjunto de treino/avaliacao para esse (shell,n_level) -- recomendado "
              "regenerar esse combo inteiro com um --method so antes de treinar/avaliar "
              "em cima dele.")
    print()

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            fieldnames = ["subject", "session", "shell", "n_level", "side", "n_directions",
                          "method", "min_angle_deg", "mean_nn_angle_deg", "centroid_norm",
                          "electrostatic_energy"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Detalhe por (sujeito,shell,n_level) salvo em: {args.out_csv}\n")

    # ---- tabela resumo agregada, uma linha por (shell, n_level, side) ----
    header = (f"{'shell':>8} {'n_level':>7} {'side':>6} {'n_subj':>6} "
              f"{'min_ang_p10':>12} {'min_ang_p50':>12} {'min_ang_p90':>12} "
              f"{'mean_nn_p50':>12} {'centroid_p50':>13} {'energy_p50':>11}")
    print(header)
    print("-" * len(header))
    for (shell_str, n_level), sides in sorted(agg.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        for side in (["input"] + (["target"] if args.include_target else [])):
            metrics = sides.get(side, {})
            min_ang = metrics.get("min_angle_deg", [])
            if not min_ang:
                continue
            n_s = len(min_ang)
            p_min = _percentiles(min_ang, (10, 50, 90))
            p_nn = _percentiles(metrics.get("mean_nn_angle_deg", []), (50,))
            p_cm = _percentiles(metrics.get("centroid_norm", []), (50,))
            p_en = _percentiles(metrics.get("electrostatic_energy", []), (50,))
            print(f"{shell_str:>8} {n_level:>7} {side:>6} {n_s:>6} "
                  f"{p_min[10]:>12.2f} {p_min[50]:>12.2f} {p_min[90]:>12.2f} "
                  f"{p_nn[50]:>12.2f} {p_cm[50]:>13.4f} {p_en[50]:>11.2f}")

    print("\nLeitura: min_ang_p50 (mediana do angulo minimo entre direcoes escolhidas, "
          "graus) e a metrica mais direta de dispersao -- quanto maior, mais espalhado; "
          "compare contra o teto real do protocolo de aquisicao (ver notas do projeto), "
          "nao contra uma expectativa de esfera uniforme de livro-texto. centroid_p50 "
          "perto de 0 = bem balanceado ao redor da esfera. energy_p50 (energia "
          "eletrostatica) so e diretamente comparavel ENTRE metodos rodados na mesma "
          "shell/n_level (varia com N e com a geometria do esquema de origem).")


if __name__ == "__main__":
    main()