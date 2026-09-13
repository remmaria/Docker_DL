#!/usr/bin/env python3
"""
Etapa 14 (diagnostico, ver addendum 2026-09-03): investiga, SEM RETREINAR
NADA, se a agregacao por MEDIA (mean-pooling, `ImplicitAngularModel3D.encode`
em model/implicit_angular.py) e' de fato um gargalo do modelo `implicit` --
ou seja, se vale a pena investir tempo trocando-a por uma agregacao
aprendida (ex.: atencao entre as n_level direcoes de entrada) antes de saber
se o checkpoint disponivel (tipicamente ainda PRECOCE, poucas epocas) ja
mostra sinal disso.

So faz FORWARD PASSES com um checkpoint ja treinado (nenhum gradiente,
nenhum otimizador, nenhuma epoca nova) -- e' deliberadamente uma
"coisinha pequena" para rodar entre os treinos longos (implicit/pairflow)
sem competir por GPU/tempo com eles: um punhado de patches de validacao,
alguns forwards extras por patch (n_level+1), termina em segundos.

DUAS METRICAS, calculadas por patch e resumidas no final:

1) "colapso de agregacao" (o quanto a media joga fora): para cada uma das
   n_level direcoes de entrada, calcula a saida do PerDirectionEncoder3D
   ANTES da media (`feat_i`, ver model/implicit_angular.py:encode) e mede o
   desvio dela em relacao a media entre as n_level direcoes
   (`||feat_i - mean_j(feat_j)|| / ||mean_j(feat_j)||`, norma L2 sobre
   canal+espaco, uma unica media/norma por patch/direcao). Se esse desvio
   for pequeno (as n_level saidas do encoder ja convergem pra algo
   parecido), a media nao esta jogando fora muita informacao -- trocar por
   atencao teria pouco o que ganhar. Se for GRANDE (direcoes produzem
   features bem diferentes entre si e a media as apaga numa unica
   representacao), e evidencia de que uma agregacao aprendida (que possa
   PESAR direcoes diferente, em vez de sempre 1/n_level) tem espaco real
   pra ajudar.

2) "sensibilidade leave-one-out" (o quanto FALTAR uma direcao muda a
   predicao final): para cada direcao i, recalcula o "estado" agregado
   excluindo so' ela (media sobre as n_level-1 restantes), decodifica pras
   MESMAS direcoes-alvo do patch, e mede a diferenca media/maxima (dentro
   da mascara de cerebro do patch) em relacao a predicao com todas as
   n_level direcoes. Isso testa diretamente se a rede trata as n_level
   direcoes como intercambiaveis (impacto de remover qualquer uma delas
   seria parecido, ~1/n_level do efeito total -- esperado sob mean-pooling
   "saudavel") ou se ALGUMAS direcoes tem impacto desproporcional (o que
   sugere que a rede JA aprendeu, implicitamente, que certas direcoes
   valem mais -- so nao tem como EXPRESSAR isso na agregacao atual, que e'
   sempre uniforme). Reporta o coeficiente de variacao (std/mean) do
   impacto entre as n_level direcoes por alvo: proximo de 0 = direcoes
   ~equivalentes (mean-pooling ja parece adequado); alto (>0.5, regra de
   bolso) = ha' heterogeneidade real que uma agregacao aprendida poderia
   explorar.

LEITURA (nao e' um teste estatistico formal, e' um diagnostico rapido pra
decidir se vale abrir a frente de atencao):
  - As duas metricas BAIXAS -> mean-pooling provavelmente nao e' o
    gargalo principal deste checkpoint (o problema mais provavel esta em
    outro lugar -- ex.: o proprio checkpoint ainda precoce, poucas epocas,
    ver addendum). Nao precipitar a implementacao de atencao.
  - As duas metricas ALTAS -> ha' evidencia concreta de heterogeneidade
    entre direcoes que a agregacao atual nao consegue expressar -- reforca
    a hipotese de que atencao pode ajudar, MESMO com um checkpoint ainda
    precoce (o comportamento ja estaria aparecendo cedo).
  - Resultado misto (ex.: colapso baixo mas sensibilidade alta, ou
    vice-versa) -- documentar e nao superinterpretar com uma amostra
    pequena de patches/sujeito unico; rodar em mais patches/sujeitos antes
    de decidir.

Uso:
    python scripts/14_diagnose_implicit_pooling.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --checkpoint work_dir/implicit_checkpoints/shell1000_n16/best.pt \
        --shell-b 1000 --n-level 16 \
        --n-patches 8 \
        --out work_dir/diagnostics/implicit_pooling_diagnosis.csv

Requer PyTorch (+ GPU opcional, roda tranquilo em CPU pra poucos patches).
Nao executado neste ambiente de desenvolvimento (sem torch instalado) --
verificado por py_compile e por um teste isolado (numpy puro) da aritmetica
de "media com uma direcao excluida" e "desvio em relacao a media", que sao
as unicas contas novas deste script (o resto e' so' orquestracao de
chamadas ja existentes: DWIPatchDataset, build_implicit_model).
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.dataset import DWIPatchDataset
from model.implicit_angular import build_implicit_model, sh_positional_encoding


def _leave_one_out_means(feat: torch.Tensor) -> torch.Tensor:
    """feat: (n_level, C, D, H, W). Retorna (n_level, C, D, H, W): para cada
    i, a media das OUTRAS n_level-1 direcoes (exclui i). Calculado via soma
    total menos o termo i, dividido por (n_level-1) -- O(n_level) memoria,
    sem repetir a soma inteira n_level vezes."""
    n_level = feat.shape[0]
    if n_level < 2:
        raise ValueError("leave-one-out precisa de n_level >= 2")
    total = feat.sum(dim=0, keepdim=True)  # (1, C, D, H, W)
    return (total - feat) / (n_level - 1)  # (n_level, C, D, H, W), broadcast do total


def _relative_deviation(feat: torch.Tensor) -> np.ndarray:
    """feat: (n_level, C, D, H, W). Retorna array (n_level,): para cada
    direcao i, ||feat_i - mean_j feat_j|| / ||mean_j feat_j|| (normas L2
    sobre canal+espaco, escalares por direcao)."""
    mean_feat = feat.mean(dim=0)  # (C, D, H, W)
    mean_norm = float(torch.linalg.vector_norm(mean_feat).item())
    if mean_norm == 0.0:
        return np.full(feat.shape[0], float("nan"))
    devs = []
    for i in range(feat.shape[0]):
        d = torch.linalg.vector_norm(feat[i] - mean_feat).item()
        devs.append(d / mean_norm)
    return np.asarray(devs, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--n-patches", type=int, default=8,
                     help="quantos patches de VALIDACAO amostrar no total (espacados "
                          "uniformemente DENTRO de cada sujeito escolhido, ver --max-subjects -- "
                          "reproducivel, nao aleatorio). Default 8: rapido, o suficiente pra uma "
                          "leitura inicial; suba se quiser mais confianca antes de decidir.")
    ap.add_argument("--max-subjects", type=int, default=2,
                     help="ADITIVO -- limita a quantos sujeitos DISTINTOS os --n-patches sao "
                          "distribuidos (default 2, patches espalhados igualmente entre eles). "
                          "Motivacao: cada sujeito novo tocado obriga _load_subject a carregar o "
                          "volume 4D INTEIRO do disco (ver utils/dataset.py) -- em storage de "
                          "cluster isso pode levar dezenas de segundos a minutos POR sujeito. "
                          "Amostrar --n-patches espalhados pelo dataset inteiro (comportamento "
                          "antigo) tende a tocar um sujeito DIFERENTE por patch, multiplicando o "
                          "tempo de espera por --n-patches sem necessidade -- este diagnostico so "
                          "precisa de patches o bastante pra ter uma leitura, nao de cobertura do "
                          "dataset inteiro. Suba isto (custando mais tempo) so se quiser checar "
                          "se o veredito muda entre sujeitos diferentes.")
    ap.add_argument("--seed", type=int, default=0,
                     help="so' usado pelo DWIPatchDataset internamente (split fixo de validacao "
                          "nao depende da seed) -- mantido por consistencia com os outros scripts.")
    ap.add_argument("--out", default=None,
                     help="se dado, salva a tabela completa (por patch x por direcao) em CSV.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    l_max = ckpt_args.get("l_max")
    base_ch = ckpt_args.get("base_ch", 16)
    norm_type = ckpt_args.get("norm_type", "instance")
    model = build_implicit_model(n_level=args.n_level, l_max=l_max, base_ch=base_ch,
                                  norm_type=norm_type).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint carregado (epoca {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss')}, "
          f"l_max={model.l_max}, base_ch={base_ch}, norm_type={norm_type})", flush=True)

    entries = load_manifest(args.manifest)
    val_entries = [e for e in entries if e.split == "val"]
    # max_cached_subjects = max_subjects (+1 de folga) -- garante que todos os
    # sujeitos amostrados cabem no cache ao mesmo tempo (sem isso, se
    # --max-subjects for maior que o default de 2 do dataset, sujeitos
    # anteriores seriam despejados e RECARREGADOS do disco quando o loop
    # voltasse a eles -- mesmo mecanismo de redundancia da secao 27 do
    # addendum, aqui evitado de proposito ja que a ordem de patches abaixo
    # e' agrupada por sujeito e nunca deveria precisar recarregar).
    val_ds = DWIPatchDataset(val_entries, args.scheme_dir, args.shell_b, args.n_level,
                              patch_size=args.patch_size, training=False,
                              mask_suffix=args.mask_suffix,
                              min_tile_coverage=args.min_tile_coverage,
                              seed=args.seed,
                              max_cached_subjects=max(args.max_subjects + 1, 2))
    n_total = len(val_ds)

    # agrupa as posicoes do dataset por sujeito (si, ver dataset.tile_index) --
    # escolhe ate --max-subjects sujeitos (espacados uniformemente entre os
    # disponiveis) e distribui --n-patches entre eles, com patches espacados
    # DENTRO de cada sujeito escolhido. Isso limita quantos volumes 4D
    # INTEIROS precisam ser lidos do disco (o gargalo real, ver
    # --max-subjects acima) a --max-subjects, nao a --n-patches.
    subj_positions = defaultdict(list)
    for flat_idx, (si, _origin) in enumerate(val_ds.tile_index):
        subj_positions[si].append(flat_idx)
    subj_ids = sorted(subj_positions)
    n_subjects_avail = len(subj_ids)
    n_subjects = min(args.max_subjects, n_subjects_avail)
    chosen_subj_pos = np.linspace(0, n_subjects_avail - 1, n_subjects, dtype=int)
    chosen_subjects = [subj_ids[i] for i in chosen_subj_pos]

    n_patches = min(args.n_patches, n_total)
    per_subj = max(1, n_patches // n_subjects)
    patch_positions = []
    for si in chosen_subjects:
        positions_si = subj_positions[si]
        k = min(per_subj, len(positions_si))
        idxs = np.linspace(0, len(positions_si) - 1, k, dtype=int)
        patch_positions.extend(positions_si[i] for i in idxs)
    patch_positions = patch_positions[:n_patches]

    subject_tags_chosen = [val_ds.usable[si][1] for si in chosen_subjects]
    print(f"[dataset] val: {n_total} patches / {n_subjects_avail} sujeitos disponiveis, "
          f"amostrando {len(patch_positions)} patches de {n_subjects} sujeito(s) "
          f"({subject_tags_chosen}) -- ate {n_subjects} carregamentos de volume 4D no total, "
          f"nao {len(patch_positions)} (ver --max-subjects).", flush=True)

    rows = []
    collapse_all = []   # um valor por (patch, direcao)
    sensitivity_cv_all = []  # um valor por (patch, alvo) -- coef. de variacao entre direcoes

    with torch.no_grad():
        for k, pos in enumerate(patch_positions):
            t0 = time.time()
            print(f"[patch {k + 1}/{len(patch_positions)}] processando idx={pos}...",
                  end=" ", flush=True)
            item = val_ds[int(pos)]
            load_s = time.time() - t0
            input_vols = item["input_vols"].unsqueeze(0).to(device)      # (1, n_level, 1, ps,ps,ps)
            input_bvecs = item["input_bvecs"].unsqueeze(0).to(device)    # (1, n_level, 3)
            target_bvecs = item["target_bvecs"].unsqueeze(0).to(device)  # (1, N_out, 3)
            subject_tag = item["subject_tag"]
            print(f"sujeito {subject_tag} (carregado/lido do cache em {load_s:.1f}s)",
                  flush=True)

            b, n_level = input_vols.shape[0], input_vols.shape[1]
            spatial = input_vols.shape[-3:]
            sh_flat = sh_positional_encoding(
                input_bvecs.reshape(b * n_level, 3), model.l_max)
            vols_flat = input_vols.reshape(b * n_level, 1, *spatial)
            feat_flat = model.per_dir_encoder(vols_flat, sh_flat)
            feat = feat_flat.reshape(n_level, model.base_ch, *spatial)  # b=1, dropa o eixo

            # metrica 1: desvio de cada direcao em relacao a media (colapso de agregacao)
            devs = _relative_deviation(feat)
            collapse_all.extend(devs.tolist())

            # estado completo (todas as n_level direcoes) -> predicao de referencia
            state_full = model.trunk(feat.mean(dim=0, keepdim=True))  # (1, base_ch, ps,ps,ps)
            pred_full = model.decode(state_full, target_bvecs)         # (1, N_out, 1, ps,ps,ps)

            # estados leave-one-out (uma media por direcao excluida) -> N_out predicoes cada
            loo_means = _leave_one_out_means(feat)                     # (n_level, C, ps,ps,ps)
            state_loo = model.trunk(loo_means)                         # (n_level, base_ch, ps,ps,ps)
            n_out = target_bvecs.shape[1]
            target_bvecs_rep = target_bvecs.expand(n_level, n_out, 3)
            pred_loo = model.decode(state_loo, target_bvecs_rep)       # (n_level, N_out, 1, ps,ps,ps)

            # mascara de cerebro do patch (reconstruida a partir do input, que ja vem
            # zerado fora da mascara em __getitem__ -- ver utils/dataset.py) -- usa
            # qualquer canal de entrada != 0 como proxy da mascara (mesmo patch pra
            # todas as direcoes, ja que a mascara nao muda por direcao).
            mask_patch = (input_vols[0].abs().sum(dim=0) > 0)  # (1, ps,ps,ps)
            n_mask_vox = int(mask_patch.sum().item())
            if n_mask_vox == 0:
                print(f"[aviso] patch idx={pos} (sujeito {subject_tag}) sem voxel de mascara -- "
                      f"pulando sensibilidade leave-one-out deste patch.", flush=True)
                continue

            diff = (pred_loo - pred_full[0].unsqueeze(0)).abs()  # (n_level, N_out, 1, ps,ps,ps)
            mask_b = mask_patch.unsqueeze(0).unsqueeze(0).expand_as(diff)
            print(f"    ... forward pass (encoder + {n_level} leave-one-out) levou "
                  f"{time.time() - t0 - load_s:.1f}s", flush=True)
            for t in range(n_out):
                per_dir_impact = []
                for i in range(n_level):
                    vals = diff[i, t][mask_b[i, t]]
                    per_dir_impact.append(float(vals.mean().item()) if vals.numel() else 0.0)
                per_dir_impact = np.asarray(per_dir_impact, dtype=np.float64)
                mean_impact = per_dir_impact.mean()
                cv = float(per_dir_impact.std() / mean_impact) if mean_impact > 0 else float("nan")
                sensitivity_cv_all.append(cv)
                rows.append({
                    "patch_idx": int(pos),
                    "subject_tag": subject_tag,
                    "target_idx_in_patch": t,
                    "colapso_dev_medio": float(devs.mean()),
                    "colapso_dev_max": float(devs.max()),
                    "sensibilidade_cv": cv,
                    "sensibilidade_impacto_medio": float(mean_impact),
                    "sensibilidade_impacto_max": float(per_dir_impact.max()),
                })

    if not rows:
        sys.exit("Nenhum patch valido processado (todos sem voxel de mascara?) -- confira "
                  "--scheme-dir/--manifest/--shell-b/--n-level.")

    collapse_arr = np.asarray(collapse_all, dtype=np.float64)
    cv_arr = np.asarray([r["sensibilidade_cv"] for r in rows if np.isfinite(r["sensibilidade_cv"])])

    print(f"\n=== Resumo ({len(rows)} (patch, alvo) avaliados, "
          f"{len(patch_positions)} patches, checkpoint epoca {ckpt.get('epoch')}) ===\n")
    print(f"1) Colapso de agregacao (desvio relativo de cada direcao vs. a media, "
          f"antes de SpatialTrunk3D):")
    print(f"   media={collapse_arr.mean():.4f}  mediana={np.median(collapse_arr):.4f}  "
          f"p90={np.percentile(collapse_arr, 90):.4f}  max={collapse_arr.max():.4f}")
    print(f"\n2) Sensibilidade leave-one-out (coef. de variacao do impacto de remover "
          f"cada direcao, por alvo):")
    if cv_arr.size:
        print(f"   media={cv_arr.mean():.4f}  mediana={np.median(cv_arr):.4f}  "
              f"p90={np.percentile(cv_arr, 90):.4f}  max={cv_arr.max():.4f}")
    else:
        print("   (sem alvos validos para calcular -- impacto medio zero em todos)")

    print("\nLeitura (regra de bolso, ver docstring do modulo pra criterio completo):")
    collapse_high = collapse_arr.mean() > 0.3
    cv_high = cv_arr.size and cv_arr.mean() > 0.5
    if collapse_high and cv_high:
        print("  -> AMBAS as metricas altas: evidencia de heterogeneidade real entre direcoes "
              "que a media uniforme nao consegue expressar. Reforca a hipotese de que uma "
              "agregacao aprendida (atencao) pode ajudar, mesmo com este checkpoint precoce.")
    elif not collapse_high and not cv_high:
        print("  -> AMBAS as metricas baixas: mean-pooling nao parece ser o gargalo principal "
              "deste checkpoint. Nao precipitar a implementacao de atencao -- o problema mais "
              "provavel esta em outro lugar (ex.: numero de epocas de treino ainda baixo).")
    else:
        print("  -> Resultado MISTO -- nao superinterpretar com uma amostra pequena. Considere "
              "rodar com --n-patches maior e/ou em outro(s) sujeito(s) antes de decidir.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patch_idx", "subject_tag", "target_idx_in_patch", "colapso_dev_medio",
                        "colapso_dev_max", "sensibilidade_cv", "sensibilidade_impacto_medio",
                        "sensibilidade_impacto_max"])
            for r in rows:
                w.writerow([r["patch_idx"], r["subject_tag"], r["target_idx_in_patch"],
                            r["colapso_dev_medio"], r["colapso_dev_max"], r["sensibilidade_cv"],
                            r["sensibilidade_impacto_medio"], r["sensibilidade_impacto_max"]])
        print(f"\nTabela completa salva em {out_path}", flush=True)


if __name__ == "__main__":
    main()