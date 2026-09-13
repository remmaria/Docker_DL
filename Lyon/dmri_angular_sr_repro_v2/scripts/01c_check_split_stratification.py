#!/usr/bin/env python3
"""
Etapa 1c (diagnostico, opcional): verifica se o split treino/val/teste do
manifest.csv ficou bem balanceado -- nao so em tamanho, mas em composicao
de protocolo (single/multi-shell), sexo, estudo de origem e idade -- entre
os 3 conjuntos. Complementa o resumo rapido ja impresso por
scripts/01_prepare_data.py (que so mostra os 5 maiores estudos); este
script cobre TODOS os estudos e adiciona testes qui-quadrado formais.

So le o manifest.csv ja pronto (rapido, sem GPU/volume) -- pode ser
rodado a qualquer momento depois da etapa 1, sem precisar regenerar nada.

Uso:
    python scripts/01c_check_split_stratification.py \
        --manifest work_dir/manifest.csv \
        --out-csv work_dir/split_stratification_check.csv
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest


# Valores criticos de qui-quadrado (alpha=0.05) para graus de liberdade 1-30
# (tabela padrao) -- usados quando scipy nao esta disponivel no ambiente.
_CHI2_CRIT_05 = {
    1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592, 7: 14.067,
    8: 15.507, 9: 16.919, 10: 18.307, 11: 19.675, 12: 21.026, 13: 22.362,
    14: 23.685, 15: 24.996, 16: 26.296, 17: 27.587, 18: 28.869, 19: 30.144,
    20: 31.410, 21: 32.671, 22: 33.924, 23: 35.172, 24: 36.415, 25: 37.652,
    26: 38.885, 27: 40.113, 28: 41.337, 29: 42.557, 30: 43.773,
}


def _chi2_critical_05(dof: int) -> float:
    """Valor critico de qui-quadrado a alpha=0.05 pro dof pedido. Usa a
    tabela padrao ate dof=30; acima disso, aproximacao de Wilson-Hilferty
    (normal->qui-quadrado), que fica bem precisa pra dof grande (>30).
    """
    if dof in _CHI2_CRIT_05:
        return _CHI2_CRIT_05[dof]
    if dof < 1:
        return float("nan")
    z = 1.645  # percentil 95% da normal padrao (cauda superior, alpha=0.05)
    return dof * (1.0 - 2.0 / (9.0 * dof) + z * np.sqrt(2.0 / (9.0 * dof))) ** 3


def chi_square_test(table: np.ndarray):
    """Teste qui-quadrado de independencia manual (sem depender de scipy --
    o ambiente do cluster pode nao ter). Tenta usar scipy.stats.chi2 pra um
    p-valor exato quando disponivel; senao, usa a tabela/aproximacao de
    _chi2_critical_05 pra reportar so significancia a 5%.

    table: matriz de contagem observada (linhas x colunas).
    Retorna dict {chi2, dof, p_value (None se scipy indisponivel),
    significant_05, min_expected (menor contagem esperada em qualquer
    celula -- regra pratica: teste fica pouco confiavel se < 5)}.
    """
    table = np.asarray(table, dtype=float)
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = table.sum()
    if total == 0:
        return None
    expected = row_sums @ col_sums / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    chi2 = float(np.nansum(terms))
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    min_expected = float(expected.min()) if expected.size else float("nan")

    p_value = None
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2, dof))
        significant_05 = p_value < 0.05
    except ImportError:
        crit = _chi2_critical_05(dof) if dof >= 1 else float("nan")
        significant_05 = (chi2 > crit) if dof >= 1 else False

    return {"chi2": chi2, "dof": dof, "p_value": p_value,
            "significant_05": significant_05, "min_expected": min_expected}


def _print_chi2_result(label: str, result):
    if result is None:
        print(f"  [{label}] tabela vazia, nao avaliado")
        return
    p_str = f"p={result['p_value']:.4f}" if result["p_value"] is not None else "p-valor exato indisponivel (scipy ausente)"
    flag = "DESBALANCEADO (p<0.05)" if result["significant_05"] else "ok (nao significativo a 5%)"
    caveat = ""
    if result["min_expected"] < 5:
        caveat = (f" -- ATENCAO: menor contagem esperada em alguma celula = "
                  f"{result['min_expected']:.1f} (<5), teste qui-quadrado pouco confiavel "
                  f"aqui, tratar como indicativo, nao conclusivo")
    print(f"  [{label}] qui2={result['chi2']:.2f}, dof={result['dof']}, {p_str} -> {flag}{caveat}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-csv", default=None,
                     help="se passado, grava a tabela completa estudo x split (contagens e %%)")
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    splits = ["train", "val", "test"]
    n_total = len(entries)
    print(f"[info] {n_total} sujeitos no manifesto\n")

    # ---- tamanho e proporcao geral ----
    print("Tamanho dos splits:")
    for s in splits:
        n = sum(1 for e in entries if e.split == s)
        print(f"  {s}: {n} ({100*n/n_total:.1f}%)")
    print()

    # ---- protocolo (single/multi-shell) ----
    protocols = sorted({e.protocol for e in entries})
    table_proto = np.array([[sum(1 for e in entries if e.split == s and e.protocol == p)
                              for p in protocols] for s in splits])
    print(f"Composicao por protocolo ({', '.join(protocols)}), dentro de cada split:")
    for i, s in enumerate(splits):
        n_s = table_proto[i].sum()
        if n_s == 0:
            continue
        pct = ", ".join(f"{p}={100*c/n_s:.1f}%" for p, c in zip(protocols, table_proto[i]))
        print(f"  {s} (n={n_s}): {pct}")
    _print_chi2_result("protocolo x split", chi_square_test(table_proto))
    print()

    # ---- sexo ----
    def sex_of(e):
        s = (e.patient_sex or "").strip().upper()
        return s if s in ("M", "F") else "UNK"

    sexes = sorted({sex_of(e) for e in entries})
    if len(sexes) > 1 or sexes != ["UNK"]:
        table_sex = np.array([[sum(1 for e in entries if e.split == s and sex_of(e) == sx)
                                for sx in sexes] for s in splits])
        print(f"Composicao por sexo ({', '.join(sexes)}), dentro de cada split:")
        for i, s in enumerate(splits):
            n_s = table_sex[i].sum()
            if n_s == 0:
                continue
            pct = ", ".join(f"{sx}={100*c/n_s:.1f}%" for sx, c in zip(sexes, table_sex[i]))
            print(f"  {s} (n={n_s}): {pct}")
        _print_chi2_result("sexo x split", chi_square_test(table_sex))
    else:
        print("Composicao por sexo: nenhum sujeito com sexo preenchido, pulado")
    print()

    # ---- estudo de origem (TODOS, nao so top-5) ----
    studies = sorted({e.study for e in entries if e.study})
    if studies:
        table_study = np.array([[sum(1 for e in entries if e.split == s and e.study == st)
                                  for st in studies] for s in splits])
        print(f"Composicao por estudo ({len(studies)} distintos) -- tabela completa "
              f"{'salva em ' + args.out_csv if args.out_csv else '(passe --out-csv pra salvar)'}:")
        _print_chi2_result("estudo x split", chi_square_test(table_study))

        if args.out_csv:
            import csv
            with open(args.out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["study", "n_total"] + [f"n_{s}" for s in splits] +
                                 [f"pct_{s}_of_study" for s in splits])
                for j, st in enumerate(studies):
                    counts = table_study[:, j]
                    n_st = counts.sum()
                    pcts = [f"{100*c/n_st:.1f}" if n_st else "" for c in counts]
                    writer.writerow([st, int(n_st)] + [int(c) for c in counts] + pcts)
            print(f"  tabela completa gravada em: {args.out_csv}")

        # estudos com maior desvio: proporcao do estudo em cada split vs.
        # proporcao esperada (= tamanho do split / n_total) -- destaca os
        # estudos mais "puxados" pra um split especifico.
        overall_split_frac = {s: sum(1 for e in entries if e.split == s) / n_total for s in splits}
        deviations = []
        for j, st in enumerate(studies):
            n_st = table_study[:, j].sum()
            if n_st < 3:
                continue  # amostra pequena demais pra essa checagem fazer sentido
            for i, s in enumerate(splits):
                observed_frac = table_study[i, j] / n_st
                dev = abs(observed_frac - overall_split_frac[s])
                deviations.append((dev, st, s, table_study[i, j], n_st))
        deviations.sort(reverse=True)
        if deviations:
            print("\n  Estudos com maior desvio de composicao (top 5, so estudos com n>=3):")
            for dev, st, s, c, n_st in deviations[:5]:
                print(f"    {st}: {c}/{n_st} no split '{s}' ({100*c/n_st:.1f}%, "
                      f"esperado ~{100*overall_split_frac[s]:.1f}% pelo tamanho geral do split)")
    else:
        print("Composicao por estudo: nenhum sujeito com `study` preenchido, pulado")
    print()

    # ---- idade (descritivo -- nao estratificado explicitamente, so conferencia) ----
    ages_by_split = defaultdict(list)
    for e in entries:
        try:
            ages_by_split[e.split].append(float(e.patient_age))
        except (TypeError, ValueError):
            pass
    if any(ages_by_split.values()):
        print("Idade por split (descritivo -- idade NAO e estratificada explicitamente, "
              "so conferencia de que nao ficou torta por acaso):")
        for s in splits:
            ages = ages_by_split.get(s, [])
            if not ages:
                continue
            arr = np.array(ages)
            print(f"  {s} (n={len(arr)}): media={arr.mean():.1f}, mediana={np.median(arr):.1f}, "
                  f"min={arr.min():.1f}, max={arr.max():.1f}")
        means = {s: np.mean(ages_by_split[s]) for s in splits if ages_by_split.get(s)}
        if len(means) >= 2:
            max_gap = max(means.values()) - min(means.values())
            print(f"  maior diferenca de media de idade entre splits: {max_gap:.1f} anos")
    else:
        print("Idade: nenhum sujeito com idade preenchida, pulado")


if __name__ == "__main__":
    main()