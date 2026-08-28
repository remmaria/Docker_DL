#!/usr/bin/env python3
"""
Etapa 2c (diagnostico, opcional): resume a distribuicao de `residual_deg`
(TODOS os alvos, independente do teto usado em 02b) e de `gap_deg`/`t_frac`
(so os alvos ja marcados VALIDOS pelo teto que estava ativo quando 02b
rodou) geradas por scripts/02b_build_rrin_triplets.py, por (shell,n_level),
agregando todos os sujeitos.

Motivacao (ver protocolo secao 10.1): a fracao de trincas validas (impressa
pelo proprio 02b) so mede "quantas passam de um teto especifico" -- nao
mostra a distribuicao inteira de residuo, que e o que voce precisa pra
ESCOLHER esse teto com informacao completa (em vez de tentar valores as
cegas). A tabela de residual_deg abaixo usa o MELHOR par disponivel por
alvo (o mesmo que find_best_bracket acha), sem aplicar nenhum teto -- e a
distancia angular perpendicular real ao plano da circunferencia que passa
pelo par escolhido (ver utils/gradients.py:spherical_triplet_residual).

Repare que a fracao "validos" e o `residual_deg` aqui refletem qualquer
que tenha sido o --max-residual-deg (e --no-require-between) passado
quando scripts/02b_build_rrin_triplets.py rodou por ULTIMO (a selecao do
par em si, nao so o corte de validade, prefere o menor gap_deg DENTRO
desse teto E com o alvo genuinamente ENTRE o par -- ver find_best_bracket
e protocolo secao 10.2) -- se voce mudar o teto, rode 02b de novo antes de
re-rodar isto pra refletir a escolha nova de pares.

Desde a versao com `require_between` (ver protocolo secao 10.2), `valid`
exige DUAS condicoes: residuo baixo (colinear) E 0<=t_frac<=1 (alvo entre
o par, nao extrapolado). Esta script tambem reporta, separadamente, a
fracao "colinear mas extrapolando" (residuo <= teto porem between=False)
-- util pra ver o quanto a exigencia de betweenness custou em relacao a
so exigir colinearidade.

Se o npz tiver sido gerado com `--ensemble-m` (ver
scripts/02b_build_rrin_triplets.py e addendum 2026-08-27 secao 13,
"ensemble em estrela"), este script TAMBEM reporta a distribuicao do
TAMANHO DO POOL do feixe por alvo (quantas posicoes de
`{base}__ens_valid` sao True) -- util pra decidir se
`--ensemble-max-residual-deg` precisa ser mais frouxo que
`--max-residual-deg`: se a maioria dos alvos tiver pool de tamanho 1,
o "ensemble" colapsa pra M=1 na pratica (nao ha o que diversificar),
mesmo que --ensemble-m tenha pedido mais.

`gap_deg`/`t_frac` continuam reportados so para os alvos VALIDOS (ja
existentes no calculo anterior) -- mede se o par escolhido, alem de
colinear o suficiente, tambem esta perto o suficiente pra parecer
"vizinho de video" (ver protocolo secao 10.1).

So le os arquivos .npz ja gerados (rapido, sem volume/GPU).

Uso:
    python scripts/02c_diagnose_rrin_triplets.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest


def _enumerate_combos(triplets_npz):
    combos = []
    for key in triplets_npz.files:
        if key.endswith("__valid"):
            base = key[: -len("__valid")]
            shell_str, level_str = base.rsplit("__", 1)
            combos.append((shell_str, int(level_str)))
    return combos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True)
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    triplets_dir = Path(args.triplets_dir)

    agg = defaultdict(lambda: {"residual_all": [], "gap_valid": [], "t_frac_valid": [],
                                "n_invalid": 0, "n_total": 0,
                                "n_between": 0, "has_between_field": False,
                                "ens_pool_sizes": [], "ensemble_m": None,
                                "has_ensemble_field": False})

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        path = triplets_dir / f"{tag}_rrin_triplets.npz"
        if not path.exists():
            continue
        trip = np.load(path)
        for shell_str, n_level in _enumerate_combos(trip):
            base = f"{shell_str}__{n_level}"
            valid = trip[f"{base}__valid"]
            gap = trip[f"{base}__gap_deg"]
            t_frac = trip[f"{base}__t_frac"]
            residual = trip[f"{base}__residual_deg"]
            key = (shell_str, n_level)
            d = agg[key]
            d["residual_all"].extend(residual.tolist())  # TODOS os alvos, sem filtro
            d["gap_valid"].extend(gap[valid].tolist())
            d["t_frac_valid"].extend(t_frac[valid].tolist())
            d["n_invalid"] += int((~valid).sum())
            d["n_total"] += int(valid.shape[0])
            between_key = f"{base}__between"
            if between_key in trip.files:
                d["has_between_field"] = True
                d["n_between"] += int(trip[between_key].sum())

            ens_valid_key = f"{base}__ens_valid"
            if ens_valid_key in trip.files:
                ens_valid = trip[ens_valid_key]  # (n_alvos, M)
                d["has_ensemble_field"] = True
                d["ensemble_m"] = ens_valid.shape[1]
                d["ens_pool_sizes"].extend(ens_valid.sum(axis=1).tolist())

    if not agg:
        sys.exit("Nenhum <tag>_rrin_triplets.npz encontrado -- rode 02b primeiro")

    print("=== distribuicao de residual_deg (TODOS os alvos, sem aplicar teto -- "
          "melhor par disponivel por alvo) ===")
    print(f"{'shell':>10s} {'n_level':>7s} {'n_alvos':>8s} "
          f"{'p10':>7s} {'p25':>7s} {'p50':>7s} {'p75':>7s} {'p90':>7s} {'p95':>7s} {'max':>7s}")
    for (shell_str, n_level) in sorted(agg.keys()):
        d = agg[(shell_str, n_level)]
        res = np.asarray(d["residual_all"])
        if res.size == 0:
            continue
        print(f"{shell_str:>10s} {n_level:>7d} {res.size:>8d} "
              f"{np.percentile(res,10):>7.1f} {np.percentile(res,25):>7.1f} "
              f"{np.percentile(res,50):>7.1f} {np.percentile(res,75):>7.1f} "
              f"{np.percentile(res,90):>7.1f} {np.percentile(res,95):>7.1f} {res.max():>7.1f}")

    print("\n=== gap_deg / t_frac entre os alvos ja marcados VALIDOS pelo teto usado "
          "na ultima rodada de 02b (mude o teto e rode 02b de novo se quiser outra vista) ===")
    print(f"{'shell':>10s} {'n_level':>7s} {'validos':>15s} "
          f"{'gap_deg p10':>11s} {'p50':>7s} {'p90':>7s} {'max':>7s} "
          f"{'t_frac p10':>11s} {'p50':>7s} {'p90':>7s}")
    for (shell_str, n_level) in sorted(agg.keys()):
        d = agg[(shell_str, n_level)]
        n_valid = d["n_total"] - d["n_invalid"]
        gap = np.asarray(d["gap_valid"])
        t_frac = np.asarray(d["t_frac_valid"])
        frac_str = f"{n_valid}/{d['n_total']}"
        if gap.size == 0:
            print(f"{shell_str:>10s} {n_level:>7d} {frac_str:>15s}  (sem trincas validas)")
            continue
        print(f"{shell_str:>10s} {n_level:>7d} {frac_str:>15s} "
              f"{np.percentile(gap,10):>11.1f} {np.percentile(gap,50):>7.1f} "
              f"{np.percentile(gap,90):>7.1f} {gap.max():>7.1f} "
              f"{np.percentile(t_frac,10):>11.2f} {np.percentile(t_frac,50):>7.2f} "
              f"{np.percentile(t_frac,90):>7.2f}")

    if any(d["has_between_field"] for d in agg.values()):
        print("\n=== fracao de alvos cujo par escolhido tem o alvo genuinamente ENTRE "
              "os dois (0<=t_frac<=1, campo 'between') -- ver protocolo secao 10.2 ===")
        print(f"{'shell':>10s} {'n_level':>7s} {'between':>15s}")
        for (shell_str, n_level) in sorted(agg.keys()):
            d = agg[(shell_str, n_level)]
            if not d["has_between_field"] or d["n_total"] == 0:
                continue
            frac = d["n_between"] / d["n_total"]
            print(f"{shell_str:>10s} {n_level:>7d} "
                  f"{d['n_between']:>6d}/{d['n_total']:<6d} ({frac:.1%})")
        print("\nSe 'between' for bem menor que 100% mesmo em n_level alto, quer dizer "
              "que o par mais 'apertado' (menor gap_deg) disponivel colinear com o alvo "
              "tipicamente o extrapola, nao o interpola -- ver nota grande no topo de "
              "scripts/02b_build_rrin_triplets.py. Isso e capturado automaticamente em "
              "'valid' (2a tabela acima) desde que 02b tenha rodado com "
              "require_between=True (default) -- 'valid' aqui ja exige as duas condicoes "
              "(colinear E between), nao so colinear.")

    if any(d["has_ensemble_field"] for d in agg.values()):
        print("\n=== tamanho do pool do 'ensemble em estrela' por alvo (--ensemble-m, ver "
              "addendum 2026-08-27 secao 13) -- quantas posicoes de ens_valid sao True ===")
        print(f"{'shell':>10s} {'n_level':>7s} {'M pedido':>9s} {'n_alvos':>8s} "
              f"{'pool=1':>9s} {'pool=M':>9s} {'media':>7s}")
        for (shell_str, n_level) in sorted(agg.keys()):
            d = agg[(shell_str, n_level)]
            if not d["has_ensemble_field"]:
                continue
            sizes = np.asarray(d["ens_pool_sizes"])
            m = d["ensemble_m"]
            n_pool1 = int((sizes == 1).sum())
            n_poolM = int((sizes == m).sum())
            print(f"{shell_str:>10s} {n_level:>7d} {m:>9d} {sizes.size:>8d} "
                  f"{n_pool1:>5d} ({n_pool1/sizes.size:.0%}) "
                  f"{n_poolM:>5d} ({n_poolM/sizes.size:.0%}) {sizes.mean():>7.2f}")
        print("\n'pool=1' alto significa que o feixe colapsa pra M=1 na pratica pra a maioria "
              "dos alvos (o pool aceitavel -- residual_deg<=teto E, se ligado, between -- so "
              "tem 1 membro, nao ha o que diversificar via farthest_point_sampling). Se isso "
              "estiver acontecendo e voce quiser um ensemble de verdade, regere o npz com "
              "--ensemble-max-residual-deg mais frouxo que --max-residual-deg (afeta so o pool "
              "do feixe, preserva o par unico -- ver scripts/02b_build_rrin_triplets.py).")

    print("\nLeitura: residual_deg (1a tabela) e a distancia angular PERPENDICULAR real "
          "do alvo ate o plano da circunferencia que passa pelo melhor par disponivel -- "
          "use os percentis pra escolher --max-residual-deg com base na distribuicao real "
          "dos SEUS dados, nao em um numero arbitrario. gap_deg alto (2a tabela, perto de "
          "90 graus, o maximo possivel com simetria antipodal) entre os alvos validos "
          "significa que o par escolhido esta bem afastado -- 'valido' pelo teto de "
          "residuo, mas provavelmente uma interpolacao de baixa qualidade na pratica (a "
          "rede vai ter que extrapolar/interpolar por um arco longo). t_frac perto de 0 "
          "ou 1 e o caso 'facil' (alvo quase coincide com uma entrada).")


if __name__ == "__main__":
    main()