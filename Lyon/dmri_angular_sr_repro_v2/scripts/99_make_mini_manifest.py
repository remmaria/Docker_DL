#!/usr/bin/env python3
"""
Etapa 99 (utilitario transversal, sem numero de pipeline fixo -- nao produz
nem consome artefato de nenhuma outra etapa alem do manifesto): recorta um
manifesto.csv existente para um pequeno subconjunto de sujeitos por split,
gerando um manifesto MENOR no MESMO formato (mesmas colunas de
utils/manifest.py:SubjectEntry) -- 100% intercambiavel com qualquer script
`04*`/`05*` via --manifest, ZERO mudanca necessaria nesses scripts.

Motivacao (pedido explicito da usuaria em 2026-09-03, "quero acelerar o
treino... vamos fazer um mini dataset... sem ter que esperar a epoca inteira
pra ver se deu certo"): iterar em mudancas de codigo de treino (ex.:
--init-checkpoint/--freeze-flow em scripts/04i_train_pairflow_star.py) exige
rodar epocas completas contra o dataset inteiro (~521 sujeitos) so' pra
descobrir se o codigo funciona ou se a loss esta descendo. Um manifesto
menor (poucos sujeitos por split) reduz drasticamente o custo de
dataloading/cache por epoca. Combine com --max-train-batches/
--max-val-batches (ver scripts/04i_train_pairflow_star.py) para o ciclo de
debug mais rapido possivel -- as duas tecnicas sao independentes.

IMPORTANTE: isso e' SO' para debug de codigo/sanidade de treino. NAO serve
pra comparar metodos nem pra tirar conclusao cientifica nenhuma (poucos
sujeitos = amostra nao-representativa, ver protocolo secao de avaliacao).

Selecao determinística por padrão (primeiros N sujeitos de cada split, na
ordem em que aparecem no manifesto de origem) -- use --shuffle-seed para
variar quais sujeitos entram (por exemplo, se os primeiros N do manifesto de
origem calharem de nao ter trincas construidas para o (shell_b, n_level)
sob teste).

--filter-triplets (opcional, requer --triplets-dir/--shell-b/--n-level/
--ensemble-m): pula sujeitos que nao tem o <tag>_rrin_triplets.npz
necessario, ou que nao tem ensemble_m suficiente gravado nesse npz (mesma
checagem feita internamente por utils/rrin_dataset.py:RRINTripletDataset.
__init__) -- evita que o mini-treino falhe ou fique com 0 sujeitos
utilizaveis por um motivo trivial (sujeito sem dado pra essa config)
irrelevante ao que esta sendo testado no codigo. USE ISSO para a linha
RCAE/RRIN/AMT/HFD/estrela/PairFlow (scripts/04b/04c/04d/04e/04g/04h/04i).

--filter-scheme (opcional, requer --scheme-dir/--shell-b/--n-level):
mesmo espirito, mas pro formato de arquivo da linha `implicit`
(scripts/04f_train_implicit.py) -- pula sujeitos sem <tag>_scheme.npz ou
sem o campo '<shell_b>__<n_level>__input' nesse npz (MESMA checagem de
utils/dataset.py:DWIPatchDataset.__init__ -- formato DIFERENTE do
`--filter-triplets`: nao ha ensemble_m/valid aqui, e o nome do campo nao
tem sufixo `__ens_pair_a`/`__valid`, so' `__input`). --filter-triplets e
--filter-scheme sao independentes -- use o que correspoder a linha de
modelo que voce vai treinar com o manifesto mini (nunca os dois juntos
na pratica, mas nada impede tecnicamente).

Uso (sem filtro, mais simples e mais rapido):
    python scripts/99_make_mini_manifest.py \
        --manifest work_dir/manifest.csv \
        --out work_dir/manifest_mini.csv \
        --n-train 2 --n-val 1 --n-test 0

Uso (com filtro de trincas -- recomendado quando for usar o mini-manifesto
pra treinar de fato, nao so' importar o CSV):
    python scripts/99_make_mini_manifest.py \
        --manifest work_dir/manifest.csv \
        --out work_dir/manifest_mini.csv \
        --n-train 2 --n-val 1 --n-test 0 \
        --filter-triplets --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 --ensemble-m 3

Uso (com filtro de esquema -- pra treinar a linha `implicit`):
    python scripts/99_make_mini_manifest.py \
        --manifest work_dir/manifest.csv \
        --out work_dir/manifest_mini.csv \
        --n-train 2 --n-val 1 --n-test 0 \
        --filter-scheme --scheme-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16

Nao executado neste ambiente de desenvolvimento com dados reais (nao ha
dataset/trincas aqui) -- verificado com um manifesto sintetico (round-trip
save_manifest/load_manifest) e por compilacao de sintaxe. So' depende de
utils/manifest.py (stdlib + numpy), e de numpy diretamente para o filtro de
trincas (np.load de arquivo .npz, mesma leitura ja feita por
utils/rrin_dataset.py).
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest, save_manifest, SubjectEntry


def _has_usable_triplets(entry: SubjectEntry, triplets_dir: str, shell_b: float,
                          n_level: int, ensemble_m: int, only_valid: bool = True) -> bool:
    """Confere se o sujeito tem um <tag>_rrin_triplets.npz utilizavel para
    este (shell_b, n_level, ensemble_m) -- MESMA checagem, byte-a-byte, de
    utils/rrin_dataset.py:RRINTripletDataset.__init__ (linhas ~116-154),
    reimplementada aqui de forma isolada e so'-leitura pra nao importar a
    classe inteira do dataset so' pra filtrar o manifesto.

    BUG CORRIGIDO (2026-09-03, achado ao rodar de verdade no cluster --
    0/600 sujeitos passavam o filtro): a primeira versao desta funcao usava
    um `tag`/nome de campo do .npz INVENTADOS (`f"{subject}__shell{...}"`
    e `"__ens_pair_a"` sem prefixo) -- o formato REAL, confirmado lendo
    RRINTripletDataset.__init__, e':
      - tag = `entry.subject` (sem sessao) ou `f"{entry.subject}_{entry.session}"`
        (com sessao) -- NUNCA inclui shell_b/n_level no nome do arquivo.
      - dentro do .npz, os campos sao prefixados por
        `key = f"{shell_b}__{n_level}"` (ex.: campo `"1000.0__16__target"`,
        NAO `"__target"` solto) -- um unico .npz por sujeito guarda VARIOS
        (shell_b, n_level) como prefixos diferentes.
    """
    import numpy as np

    tag = entry.subject if not entry.session else f"{entry.subject}_{entry.session}"
    npz_path = Path(triplets_dir) / f"{tag}_rrin_triplets.npz"
    if not npz_path.exists():
        return False
    key = f"{shell_b}__{n_level}"
    try:
        with np.load(npz_path) as trip:
            if f"{key}__target" not in trip.files:
                return False
            if ensemble_m > 0:
                if f"{key}__ens_pair_a" not in trip.files:
                    return False
                actual_m = trip[f"{key}__ens_pair_a"].shape[1]
                if actual_m < ensemble_m:
                    return False
            if only_valid:
                valid = trip[f"{key}__valid"]
                if not valid.any():
                    return False
        return True
    except Exception as e:
        print(f"[aviso] falha lendo {npz_path} ({e}) -- tratando como sujeito sem trincas "
              f"utilizaveis")
        return False


def _has_usable_scheme(entry: SubjectEntry, scheme_dir: str, shell_b: float,
                        n_level: int) -> bool:
    """Confere se o sujeito tem um <tag>_scheme.npz utilizavel para este
    (shell_b, n_level) -- MESMA checagem de
    utils/dataset.py:DWIPatchDataset.__init__ (linhas ~217-226), usada pela
    linha `implicit` (scripts/04f_train_implicit.py). Formato DIFERENTE de
    `_has_usable_triplets`: nao ha ensemble_m/valid aqui -- so' precisa
    existir o campo '<key>__input' (key = f"{shell_b}__{n_level}").
    """
    import numpy as np

    tag = entry.subject if not entry.session else f"{entry.subject}_{entry.session}"
    npz_path = Path(scheme_dir) / f"{tag}_scheme.npz"
    if not npz_path.exists():
        return False
    key = f"{shell_b}__{n_level}"
    try:
        with np.load(npz_path) as scheme:
            return f"{key}__input" in scheme.files
    except Exception as e:
        print(f"[aviso] falha lendo {npz_path} ({e}) -- tratando como sujeito sem esquema "
              f"utilizavel")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifesto.csv de origem (completo).")
    ap.add_argument("--out", required=True, help="caminho do manifesto.csv reduzido a escrever.")
    ap.add_argument("--n-train", type=int, default=2)
    ap.add_argument("--n-val", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=None,
                     help="se informado, embaralha os sujeitos de cada split (com essa semente) "
                          "antes de escolher os N primeiros -- default None = mantem a ordem "
                          "original do manifesto de origem (determinístico, sem semente).")
    ap.add_argument("--filter-triplets", action="store_true",
                     help="so' seleciona sujeitos que ja tem <tag>_rrin_triplets.npz com "
                          "--ensemble-m pares gravados para (--shell-b, --n-level) -- requer "
                          "--triplets-dir/--shell-b/--n-level/--ensemble-m.")
    ap.add_argument("--triplets-dir", default=None)
    ap.add_argument("--shell-b", type=float, default=None)
    ap.add_argument("--n-level", type=int, default=None)
    ap.add_argument("--ensemble-m", type=int, default=3)
    ap.add_argument("--no-only-valid", action="store_true",
                     help="por padrao (only_valid=True), so' aceita sujeitos com pelo menos 1 "
                          "trinca VALIDA no feixe (campo '<key>__valid'.any()) -- MESMO default "
                          "de scripts/04i_train_pairflow_star.py/RRINTripletDataset. Passe esta "
                          "flag pra tambem aceitar sujeitos so' com trincas de par-unico "
                          "invalido (espelha --no-only-valid la').")
    ap.add_argument("--filter-scheme", action="store_true",
                     help="so' seleciona sujeitos que ja tem <tag>_scheme.npz com o campo "
                          "'<shell_b>__<n_level>__input' -- requer --scheme-dir/--shell-b/"
                          "--n-level. Formato da linha `implicit` (scripts/04f_train_implicit."
                          "py), DIFERENTE de --filter-triplets (sem ensemble_m/valid aqui).")
    ap.add_argument("--scheme-dir", default=None)
    args = ap.parse_args()

    if args.filter_triplets:
        missing = [name for name in ("triplets_dir", "shell_b", "n_level")
                   if getattr(args, name) is None]
        if missing:
            sys.exit(f"--filter-triplets requer --{'/--'.join(m.replace('_', '-') for m in missing)}")
    if args.filter_scheme:
        missing = [name for name in ("scheme_dir", "shell_b", "n_level")
                   if getattr(args, name) is None]
        if missing:
            sys.exit(f"--filter-scheme requer --{'/--'.join(m.replace('_', '-') for m in missing)}")

    entries = load_manifest(args.manifest)
    print(f"[info] manifesto de origem: {len(entries)} sujeitos")

    by_split: dict[str, list[SubjectEntry]] = {}
    for e in entries:
        by_split.setdefault(e.split, []).append(e)

    wanted = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    selected: list[SubjectEntry] = []
    for split, n_wanted in wanted.items():
        if n_wanted <= 0:
            continue
        group = list(by_split.get(split, []))
        if args.shuffle_seed is not None:
            random.Random(args.shuffle_seed).shuffle(group)
        chosen = []
        for e in group:
            if len(chosen) >= n_wanted:
                break
            if args.filter_triplets:
                if not _has_usable_triplets(e, args.triplets_dir, args.shell_b, args.n_level,
                                             args.ensemble_m, only_valid=not args.no_only_valid):
                    continue
            if args.filter_scheme:
                if not _has_usable_scheme(e, args.scheme_dir, args.shell_b, args.n_level):
                    continue
            chosen.append(e)
        if len(chosen) < n_wanted:
            filtro_desc = ", ".join(d for d, ativo in (
                ("apos filtro de trincas", args.filter_triplets),
                ("apos filtro de esquema", args.filter_scheme),
            ) if ativo)
            print(f"[aviso] split '{split}': pedido {n_wanted} sujeito(s), so' encontrado(s) "
                  f"{len(chosen)} (dataset de origem tem {len(group)} sujeito(s) nesse split"
                  + (f", {filtro_desc}" if filtro_desc else "") + ")")
        selected.extend(chosen)
        tags = [e.subject for e in chosen]
        print(f"[info] split '{split}': selecionado(s) {len(chosen)} -> {tags}")

    if not selected:
        sys.exit("Nenhum sujeito selecionado (confira --n-train/--n-val/--n-test e o manifesto "
                  "de origem) -- nao escrevendo manifesto vazio.")

    save_manifest(selected, args.out)
    print(f"[ok] manifesto mini escrito em {args.out} ({len(selected)} sujeito(s) no total)")


if __name__ == "__main__":
    main()