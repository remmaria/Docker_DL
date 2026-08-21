#!/usr/bin/env python3
"""
Etapa 2: para cada sujeito do manifesto, gera o esquema de subamostragem
angular por shell (indices de entrada vs. alvo/held-out) para cada nivel
de N direcoes pedido.

Uso:
    python scripts/02_subsample_directions.py \
        --manifest /caminho/work_dir/manifest.csv \
        --out-dir /caminho/work_dir/subsampling \
        --levels 6 10 15 20 30

Saida: <out-dir>/<subject>[_<session>]_scheme.npz
  Cada arquivo .npz contem, para cada shell e nivel, os arrays de indices
  (globais, relativos ao bval/bvec original do sujeito) de entrada e alvo.
  Layout das chaves: "{shell}__{level}__input" e "{shell}__{level}__target".
  Niveis nao aplicaveis (> n direcoes disponiveis na shell) sao omitidos e
  reportados no console.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, build_subsampling_scheme


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[6, 10, 15, 20, 30])
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--seed-idx", type=int, default=0,
                     help="indice local (dentro da shell) usado como semente do farthest-point sampling")
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for e in entries:
        bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
        scheme = build_subsampling_scheme(bvals, bvecs, n_levels=args.levels,
                                           tol=args.shell_tol, seed_idx=args.seed_idx)
        save_dict = {}
        for shell_b, levels in scheme.items():
            for level, d in levels.items():
                if d["input_idx"] is None:
                    print(f"[aviso] {e.subject}: shell {shell_b} tem apenas "
                          f"{d['n_available']} direcoes, nivel {level} pulado")
                    continue
                key = f"{shell_b}__{level}"
                save_dict[f"{key}__input"] = d["input_idx"]
                save_dict[f"{key}__target"] = d["target_idx"]

        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        out_path = out_dir / f"{tag}_scheme.npz"
        np.savez(out_path, **save_dict)
        print(f"{e.subject}: {len(save_dict)//2} combinacoes (shell,nivel) salvas em {out_path}")


if __name__ == "__main__":
    main()
