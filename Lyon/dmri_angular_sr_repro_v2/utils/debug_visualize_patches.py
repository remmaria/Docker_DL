#!/usr/bin/env python3
"""
Debug: salva algumas fatias 2D de patches (entrada + alvo) amostrados pelo
DWIPatchDataset, como PNG com cmap='jet' -- pra inspecionar visualmente o
que de fato entra no treino (ex.: conferir se o mascaramento de fundo esta
funcionando, se os patches caem dentro do cerebro, se as direcoes de
entrada/alvo fazem sentido).

Nao roda o modelo (so entrada/alvo, sem predicao) -- pra ver a predicao
evoluindo epoca a epoca durante o treino de verdade, use
--debug-plot-every em scripts/04_train_rcae.py.

Uso:
    python scripts/debug_visualize_patches.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --shell-b 1000 --n-level 10 \
        --out-dir work_dir/debug_patches \
        --n-samples 4 --split train
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.dataset import DWIPatchDataset
from utils.viz import save_patch_debug_png


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--patch-size", type=int, default=10,
                     help="default 10 -- ver utils/dataset.py (grade deterministica, "
                          "nao mais crop aleatorio)")
    ap.add_argument("--q-out", type=int, default=10,
                     help="numero fixo de direcoes-alvo por exemplo (default 10)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--n-samples", type=int, default=4,
                     help="quantos patches (indices distintos do dataset) visualizar")
    ap.add_argument("--max-dirs", type=int, default=6,
                     help="no maximo quantas direcoes de entrada/alvo mostrar por amostra")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    entries = load_manifest(args.manifest)
    if args.split != "all":
        entries = [e for e in entries if e.split == args.split]

    # training=(split=="train") -- so pra essa amostra visual tambem
    # refletir o split dinamico de q-space quando --split train (ver
    # utils/dataset.py:_dynamic_split); val/test/all usam o split fixo.
    ds = DWIPatchDataset(entries, args.scheme_dir, args.shell_b, args.n_level,
                          patch_size=args.patch_size, q_out=args.q_out,
                          training=(args.split == "train"), seed=args.seed)

    out_dir = Path(args.out_dir)
    print(f"{len(ds.usable)} sujeitos utilizaveis para shell={args.shell_b} "
          f"nivel={args.n_level} split={args.split} ({len(ds)} tiles no total)")

    for i in range(min(args.n_samples, len(ds))):
        subj_idx, _origin = ds.tile_index[i]
        _, tag = ds.usable[subj_idx]
        sample = ds[i]

        out_path = out_dir / f"patch_{i:03d}_{tag}.png"
        save_patch_debug_png(
            out_path, sample["input_vols"], sample["target_vols"],
            max_dirs=args.max_dirs,
            title=f"{tag} | shell={args.shell_b} n={args.n_level}",
        )
        print(f"salvo: {out_path}")


if __name__ == "__main__":
    main()