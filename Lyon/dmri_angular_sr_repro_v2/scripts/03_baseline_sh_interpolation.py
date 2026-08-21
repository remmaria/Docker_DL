#!/usr/bin/env python3
"""
Etapa 3 (baseline nao-DL): reconstrucao por interpolacao de harmonicos
esfericos (SH), regularizada (Laplace-Beltrami), a partir do subconjunto de
direcoes de entrada definido na etapa 2. Serve de piso de comparacao para o
RCAE -- nao ha treinamento, e ajustado sujeito a sujeito.

Uso:
    python scripts/03_baseline_sh_interpolation.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --out-dir work_dir/baseline_recon \
        --split test

Por padrao processa TODOS os (shell, nivel) presentes no esquema de cada
sujeito -- o que gera de uma vez o baseline inteiro de todos os
experimentos, com o pico de disco correspondente. Para gerar so um combo
por vez (recomendado quando ha muitos experimentos e espaco em disco e
limitado -- ver slurm/02b_baseline_reconstruct.sh), passe --shell-b e
--n-level juntos.

Requer nibabel e dipy (opcional para mascara -- usamos threshold simples de
b0 se nao houver mascara explicita). Procura, ao lado do dwi, um arquivo
"<stem><--mask-suffix>" (default "_mask3d.nii.gz", ex.:
"bgpdwis_PA_geomcorr_mask3d.nii.gz"); se ausente, gera mascara por
threshold no b0 medio (Otsu simplificado por percentil).
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, load_dwi, split_shells
from utils.sh_basis import fit_sh, predict_sh
from utils.masking import load_or_build_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--lambda-reg", type=float, default=0.006)
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--shell-b", type=float, default=None,
                     help="se combinado com --n-level, processa so esse combo (nao todos)")
    ap.add_argument("--n-level", type=int, default=None,
                     help="se combinado com --shell-b, processa so esse combo (nao todos)")
    ap.add_argument("--shard-index", type=int, default=0,
                     help="indice (0-based) deste shard, para paralelizar por SUJEITO dentro "
                          "do mesmo combo shell/n_level (ex.: via SLURM array) em vez de "
                          "processar todos os sujeitos do split num job so. Use junto com "
                          "--shard-count -- ver slurm/02b_baseline_reconstruct.sh. Nao precisa "
                          "de merge depois (cada sujeito grava na sua propria pasta, sem "
                          "arquivo de saida compartilhado entre shards).")
    ap.add_argument("--shard-count", type=int, default=1,
                     help="numero total de shards (default 1 = sem sharding).")
    args = ap.parse_args()

    if (args.shell_b is None) != (args.n_level is None):
        sys.exit("--shell-b e --n-level precisam ser passados juntos (ou nenhum dos dois)")
    if not (0 <= args.shard_index < max(args.shard_count, 1)):
        sys.exit(f"--shard-index ({args.shard_index}) fora do intervalo [0, {args.shard_count})")
    only_combo = None
    if args.shell_b is not None:
        only_combo = f"{args.shell_b}__{args.n_level}"

    import nibabel as nib

    entries = load_manifest(args.manifest)
    if args.split != "all":
        entries = [e for e in entries if e.split == args.split]
    if args.shard_count > 1:
        entries = entries[args.shard_index::args.shard_count]
        print(f"[shard {args.shard_index}/{args.shard_count}] {len(entries)} sujeitos "
              f"neste shard", flush=True)

    out_dir = Path(args.out_dir)
    scheme_dir = Path(args.scheme_dir)

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        try:
            _process_subject(e, tag, scheme_dir, out_dir, only_combo, args, nib)
        except Exception:
            # NAO deixa 1 sujeito com problema (dado corrompido, shell com
            # poucas direcoes, mascara vazia, etc.) matar o processo inteiro
            # -- sem isso, um erro no sujeito N fazia os sujeitos N+1..fim
            # da lista desse shard nunca serem tentados, e como o traceback
            # nao menciona o NOME de quem sumiu (o crash e em outro
            # sujeito), esses ficam "desaparecidos" sem pista nenhuma no
            # log. Ver historico: foi assim que 18 sujeitos sumiram
            # silenciosamente numa rodada de --array=1-100 sharded.
            print(f"[erro] falha processando {tag} -- pulando este sujeito e "
                  f"continuando com o resto do shard. Traceback completo abaixo:")
            traceback.print_exc()


def _process_subject(e, tag, scheme_dir, out_dir, only_combo, args, nib):
    scheme_path = scheme_dir / f"{tag}_scheme.npz"
    if not scheme_path.exists():
        print(f"[aviso] sem esquema de subamostragem para {tag}, pulando")
        return
    scheme = np.load(scheme_path)

    # descobre quais (shell, nivel) existem para esse sujeito a partir das chaves
    # salvas -- checado ANTES de carregar o DWI 4D inteiro (load_dwi/mascara), pra
    # nao gastar I/O e RAM com um sujeito que vai ser pulado de qualquer jeito.
    combos = sorted({k.rsplit("__", 1)[0] for k in scheme.files})
    if only_combo is not None:
        combos = [c for c in combos if c == only_combo]
        if not combos:
            # ANTES isso retornava sem nenhum print (comentado "pula sem
            # aviso", de proposito -- a logica original considerava isso
            # "nao e erro"). O problema: sem NENHUMA linha no log, um
            # sujeito que simplesmente nao tem essa shell (ex.: so tem
            # 700/2000, nao 1000) fica indistinguivel de uma task que
            # travou/morreu sem deixar rastro -- foi exatamente essa
            # ambiguidade que the fez parecer um bug serio (task "sumindo"
            # sem erro) quando na verdade era so esse sujeito nao ter a
            # shell pedida. Agora fica explicito no log, sem custar
            # I/O extra (a checagem e antes do load_dwi).
            print(f"[aviso] {tag}: sem combo shell={only_combo.split('__')[0]}/"
                  f"n={only_combo.split('__')[1]} no esquema (esse sujeito nao tem "
                  f"essa shell, ou tem direcoes insuficientes nela) -- pulando. "
                  f"Combos disponiveis para este sujeito: {sorted({k.rsplit('__', 1)[0] for k in scheme.files})}")
            return

    bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
    data, affine, header = load_dwi(e.dwi_path)

    shells = split_shells(bvals, tol=args.shell_tol)
    b0_idx = shells.get(0, np.array([], dtype=int))
    if b0_idx.size == 0:
        print(f"[erro] {tag}: nenhum volume b0 encontrado, pulando")
        return
    b0_mean = data[..., b0_idx].mean(axis=-1)

    mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

    b0_safe = np.where(b0_mean > 0, b0_mean, 1.0)

    saved_masks = set()  # evita regravar a mesma mascara a cada nivel (ela nao muda)
    for combo in combos:
        shell_str, level_str = combo.split("__")
        shell_b = float(shell_str)
        level = int(level_str)
        input_idx = scheme[f"{combo}__input"]
        target_idx = scheme[f"{combo}__target"]

        input_signal = data[..., input_idx] / b0_safe[..., None]
        input_bvecs = bvecs[input_idx]
        target_bvecs = bvecs[target_idx]

        voxels = input_signal[mask]  # (n_voxels_mask, n_input)
        coef, _, l_max_used = fit_sh(voxels, input_bvecs, l_max=None,
                                      lambda_reg=args.lambda_reg)
        pred_voxels = predict_sh(coef, target_bvecs, l_max_used)  # (n_voxels_mask, n_target)

        pred_signal = np.zeros(mask.shape + (target_idx.shape[0],), dtype=np.float32)
        pred_signal[mask] = pred_voxels
        pred_dwi = pred_signal * b0_safe[..., None]  # volta pra escala de intensidade original

        shell_out = out_dir / tag / f"shell{int(shell_b)}"
        sub_out = shell_out / f"n{level}"
        sub_out.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred_dwi.astype(np.float32), affine), sub_out / "recon_target.nii.gz")
        np.save(sub_out / "target_idx.npy", target_idx)
        # mascara e identica pra todo nivel dessa (sujeito, shell) -- grava uma vez
        # em shell_out (nivel acima) em vez de duplicar em cada pasta n{level}
        if shell_out not in saved_masks:
            np.save(shell_out / "mask.npy", mask)
            saved_masks.add(shell_out)
        print(f"{tag} shell={int(shell_b)} n={level}: reconstruido "
              f"{target_idx.shape[0]} direcoes held-out (l_max={l_max_used}) -> {sub_out}")


if __name__ == "__main__":
    main()