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

--method (default "fps"): "fps" (farthest-point sampling, metodo historico
deste pipeline) ou "electrostatic" (energia eletrostatica de Jones et al.
1999 + multi-start + busca local 2-opt, porte de select_pair_rep_centroid_v2.py
-- ver utils.gradients.electrostatic_repulsion_sampling). Recomendado usar
"electrostatic" a partir de 2026-09 (ver notas do projeto "Subamostragem
angular e pipeline de treino da rede"): e o MESMO metodo ja usado no
pre-processamento clinico (--sub_qspace do run_pipeline.sh), e alcanca
dispersao angular bem melhor (ex.: ~28,4 graus de angulo minimo vs. ~14,3
graus do teto do proprio protocolo, para N=16 a partir de 64 direcoes,
contra uma dispersao tipicamente pior com "fps"). Usar "fps" para o treino
e "electrostatic" so na validacao clinica introduziria um domain gap
GEOMETRICO (alem do domain gap de SNR ja documentado) -- a rede nunca
teria visto, no treino, entradas tao bem distribuidas quanto as que
aparecem na validacao/producao.
"""
import argparse
import sys
import traceback
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
                     help="indice local (dentro da shell) usado como semente do farthest-point "
                          "sampling -- ignorado quando --method=electrostatic (ver --sampling-seed)")
    ap.add_argument("--method", choices=["fps", "electrostatic"], default="fps",
                     help="metodo de selecao de direcoes -- ver docstring do modulo")
    ap.add_argument("--n-starts", type=int, default=20,
                     help="so para --method=electrostatic: quantas sementes aleatorias tentar "
                          "(20 validado em 2026-09 via scripts/02e_calibrate_electrostatic_nstarts.py "
                          "para shells 500-2000 x N 6-54 -- ja n_starts=10 ficava a <1% da energia "
                          "de n_starts=160 em todas as combinacoes testadas; 20 mantem margem de "
                          "seguranca 2x)")
    ap.add_argument("--max-local-iter", type=int, default=150,
                     help="so para --method=electrostatic: teto de iteracoes da busca local 2-opt")
    ap.add_argument("--sampling-seed", type=int, default=42,
                     help="so para --method=electrostatic: semente do RNG que escolhe as sementes de --n-starts")
    ap.add_argument("--shard-index", type=int, default=0,
                     help="indice (0-based) deste shard, para paralelizar por SUJEITO (ex.: via "
                          "SLURM array) em vez de processar o manifesto inteiro num job so. Use "
                          "junto com --shard-count -- ver slurm/02_baseline_sh.sh. Nao precisa de "
                          "merge depois (cada sujeito grava seu proprio .npz, sem arquivo "
                          "compartilhado entre shards).")
    ap.add_argument("--shard-count", type=int, default=1,
                     help="numero total de shards (default 1 = sem sharding).")
    args = ap.parse_args()

    if not (0 <= args.shard_index < max(args.shard_count, 1)):
        sys.exit(f"--shard-index ({args.shard_index}) fora do intervalo [0, {args.shard_count})")

    entries = load_manifest(args.manifest)
    if args.shard_count > 1:
        entries = entries[args.shard_index::args.shard_count]
        print(f"[shard {args.shard_index}/{args.shard_count}] {len(entries)} sujeitos neste shard",
              flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "electrostatic":
        print(f"[info] method=electrostatic -- n_starts={args.n_starts}, "
              f"max_local_iter={args.max_local_iter}, sampling_seed={args.sampling_seed} "
              f"(bem mais lento que fps por sujeito -- O(n_starts x n_shell^2) por shell/nivel)")

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        try:
            bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
            scheme = build_subsampling_scheme(bvals, bvecs, n_levels=args.levels,
                                               tol=args.shell_tol, seed_idx=args.seed_idx,
                                               method=args.method, n_starts=args.n_starts,
                                               max_local_iter=args.max_local_iter,
                                               seed=args.sampling_seed)
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
                    # Metadado de proveniencia (adicionado 2026-09, ver discussao no
                    # chat sobre nao dar pra saber depois do fato qual --method gerou
                    # um scheme.npz ja pronto -- nem log rotacionado nem o .npz em si
                    # guardavam essa informacao antes). Gravado POR COMBO (nao como
                    # chave solta no topo do arquivo) de proposito: scripts/03_baseline_sh_
                    # interpolation.py descobre os combos de um sujeito via
                    # `k.rsplit("__", 1)[0] for k in scheme.files` e depois faz
                    # `combo.split("__")` esperando EXATAMENTE 2 partes -- uma chave solta
                    # sem esse formato (ex. "__method") quebraria esse split. Usando
                    # "{shell}__{level}__method" em vez disso, o rsplit(...,1) devolve o
                    # mesmo combo base de sempre ("{shell}__{level}"), entao e' 100%
                    # retrocompativel com todo script que ja le scheme.npz (nenhum
                    # combo novo/espurio aparece) -- so quem sabe procurar o campo
                    # "__method" explicitamente o enxerga.
                    save_dict[f"{key}__method"] = np.asarray(args.method)
                    if args.method == "electrostatic":
                        save_dict[f"{key}__n_starts"] = np.asarray(args.n_starts)
                        save_dict[f"{key}__max_local_iter"] = np.asarray(args.max_local_iter)
                        save_dict[f"{key}__sampling_seed"] = np.asarray(args.sampling_seed)

            out_path = out_dir / f"{tag}_scheme.npz"
            np.savez(out_path, **save_dict)
            n_combos = sum(1 for k in save_dict if k.endswith("__target"))
            print(f"{e.subject}: {n_combos} combinacoes (shell,nivel) salvas em {out_path} "
                  f"(method={args.method})")
        except Exception:
            # nao deixa 1 sujeito com problema (bval/bvec corrompido, shell
            # degenerada, etc.) matar o shard inteiro -- ver a mesma licao em
            # scripts/03_baseline_sh_interpolation.py (18 sujeitos sumiram
            # silenciosamente numa rodada --array sem essa protecao).
            print(f"[erro] falha processando {tag} -- pulando este sujeito e "
                  f"continuando com o resto do shard. Traceback completo abaixo:")
            traceback.print_exc()


if __name__ == "__main__":
    main()