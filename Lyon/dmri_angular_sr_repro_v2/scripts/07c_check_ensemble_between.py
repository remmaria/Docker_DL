#!/usr/bin/env python3
"""
Diagnostico rapido (nao faz parte do pipeline, so debug pontual --
addendum 2026-08-27 secao 14): para a MESMA trinca que
scripts/07_visualize_triplet.py escolheria (mesmo --subject/--shell-b/
--n-level/--example/--triplet-index), imprime os valores REAIS de
`between`/`residual_deg`/`gap_deg`/`t_frac` do par canonico e de cada
posicao do feixe "ensemble em estrela" (--ensemble-m, ver secao 13) --
sem precisar plotar nada, so pra confirmar numericamente se um par do
feixe deveria mesmo passar perto do alvo (between=True, residual baixo)
ou se a distancia visual na figura e so efeito de projecao 3D->2D
(camera/paralaxe) da imagem estatica.

Nao precisa de GPU/torch nem de volume DWI -- so le os .npz ja gerados
pelas etapas 2/2b (numpy puro), roda em segundos direto no login node,
sem precisar de sbatch. Ver run_check_ensemble_between.sh para um
wrapper de uma linha.

Uso:
    python scripts/07c_check_ensemble_between.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --subject 20170417094841_802780_20170417094841_802780 \
        --example typical
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest


def _tag_of(e):
    return e.subject if not e.session else f"{e.subject}_{e.session}"


def _pick_subject_and_key(entries, triplets_dir: Path, shell_b: float, n_level: int,
                           wanted_subject: str = None):
    key = f"{shell_b}__{n_level}"
    candidates = [e for e in entries if e.split == "train"] if wanted_subject is None else entries
    for e in candidates:
        tag = _tag_of(e)
        if wanted_subject is not None and tag != wanted_subject:
            continue
        trip_path = triplets_dir / f"{tag}_rrin_triplets.npz"
        scheme_path = triplets_dir / f"{tag}_scheme.npz"
        if not trip_path.exists() or not scheme_path.exists():
            continue
        trip = np.load(trip_path)
        if f"{key}__target" not in trip.files:
            continue
        return e, tag, key
    raise SystemExit(f"Nenhum sujeito encontrado com scheme+trincas para shell_b={shell_b} "
                      f"n_level={n_level}" + (f" (procurando especificamente {wanted_subject})"
                                              if wanted_subject else " no split 'train'") + ".")


def _select_triplet(valid, gap_deg, residual_deg, example: str, triplet_index: int = None):
    n = valid.shape[0]
    if triplet_index is not None:
        if not (0 <= triplet_index < n):
            raise SystemExit(f"--triplet-index {triplet_index} fora do intervalo [0,{n-1}]")
        return triplet_index
    if example == "worst":
        return int(np.argmax(gap_deg))
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        return int(np.argmin(residual_deg))
    if example == "best":
        best_local = np.argmin(residual_deg[valid_idx])
        return int(valid_idx[best_local])
    median_gap = np.median(gap_deg[valid_idx])
    typical_local = np.argmin(np.abs(gap_deg[valid_idx] - median_gap))
    return int(valid_idx[typical_local])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--example", default="typical", choices=["typical", "best", "worst"])
    ap.add_argument("--triplet-index", type=int, default=None)
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    triplets_dir = Path(args.triplets_dir)
    e, tag, key = _pick_subject_and_key(entries, triplets_dir, args.shell_b, args.n_level,
                                         wanted_subject=args.subject)
    trip = np.load(triplets_dir / f"{tag}_rrin_triplets.npz")

    valid = trip[f"{key}__valid"]
    gap_deg = trip[f"{key}__gap_deg"]
    residual_deg = trip[f"{key}__residual_deg"]
    t_frac = trip[f"{key}__t_frac"]
    ti = _select_triplet(valid, gap_deg, residual_deg, args.example, args.triplet_index)

    print(f"sujeito={tag}  shell_b={args.shell_b}  n_level={args.n_level}  trinca #{ti} "
          f"(criterio '{args.example if args.triplet_index is None else 'manual'}')")
    print(f"  par CANONICO: valid={bool(valid[ti])}  residual_deg={residual_deg[ti]:.2f}  "
          f"gap_deg={gap_deg[ti]:.2f}  t_frac={t_frac[ti]:.3f}")

    ens_valid_key = f"{key}__ens_valid"
    if ens_valid_key not in trip.files:
        print("\n[info] este .npz nao tem campos '__ens_*' -- rode 02b_build_rrin_triplets.py "
              "com --ensemble-m primeiro.")
        return

    ens_valid = trip[f"{key}__ens_valid"][ti]
    ens_residual = trip[f"{key}__ens_residual_deg"][ti]
    ens_gap = trip[f"{key}__ens_gap_deg"][ti]
    ens_tfrac = trip[f"{key}__ens_t_frac"][ti]
    ens_a = trip[f"{key}__ens_pair_a"][ti]
    ens_b = trip[f"{key}__ens_pair_b"][ti]
    # scripts/02b_build_rrin_triplets.py NAO grava um campo '__ens_between'
    # separado no .npz (so 'ens_t_frac', de onde between e' derivado --
    # ver find_star_ensemble_batch: between = 0<=t_frac<=1). Deriva aqui
    # em vez de depender de um campo que nao existe.
    ens_between = (ens_tfrac >= 0.0) & (ens_tfrac <= 1.0)

    print(f"\n  feixe 'ensemble em estrela' (M={ens_valid.shape[0]}):")
    for m in range(ens_valid.shape[0]):
        if not ens_valid[m]:
            print(f"    slot {m}: padding (sem par real aqui)")
            continue
        print(f"    slot {m}: par (#{ens_a[m]},#{ens_b[m]})  residual_deg={ens_residual[m]:.2f}  "
              f"gap_deg={ens_gap[m]:.2f}  t_frac={ens_tfrac[m]:.3f}  between={bool(ens_between[m])}")

    print("\nLeitura: se 'between=True' e 'residual_deg' baixo (poucos graus) para um slot, o "
          "arco daquele par DEVE passar geometricamente perto do alvo em 3D (distancia esperada "
          "~2*sin(residual_deg/2) na esfera unitaria) -- se a figura estatica parecer diferente "
          "disso, e mais provavel ser efeito de projecao/camera da imagem 2D do que um erro nos "
          "dados. 'between=False' significa que esse par e' extrapolacao (fallback), e NAO se "
          "espera que passe perto do alvo -- isso e' esperado, nao e' bug.")


if __name__ == "__main__":
    main()