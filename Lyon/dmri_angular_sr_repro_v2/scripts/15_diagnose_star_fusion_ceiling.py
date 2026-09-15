#!/usr/bin/env python3
"""
Etapa 15 (DIAGNOSTICO, nao treina nada): mede o TETO da fusao no ensemble em
estrela -- ou seja, quanto do gap que ainda existe ate o RCAE poderia, em
principio, ser fechado melhorando a CABECA DE FUSAO, e quanto so pode ser
fechado melhorando as PREDICOES CANDIDATAS em si.

MOTIVACAO (addendum 2026-09-14, "e sobre o rrinstar e pairflow?"): a
confirmacao empirica do item 2 (atencao entre direcoes no `implicit`, ganho
de 2,2-2,7x em velocidade de convergencia) levantou a pergunta de o que
transfere para as linhas star. Ha duas leituras possiveis, e elas apontam
para trabalhos MUITO diferentes:

  (A) "o que funcionou foi ATENCAO entre elementos de um conjunto" -> entao
      o analogo direto e o item 1 (CrossCandidateAttention3D, ja
      implementado), que deixa os M candidatos se verem antes do logit de
      fusao.
  (B) "o que funcionou foi dar mais INFORMACAO ao preditor" (cada direcao
      passou a ver as outras 15 antes do pooling) -> entao o analogo e dar
      contexto angular global a linha de fluxo, cujo `FlowNet3D` ve
      exatamente 2 das n_level direcoes medidas e nunca as outras.

A diferenca pratica e grande: (A) ja esta implementado e e' barato de testar;
(B) exige mudar `utils/rrin_dataset.py` para devolver todas as n_level
direcoes de entrada do patch, mais um encoder novo nos dois modelos star.

ESTE SCRIPT DECIDE ENTRE AS DUAS SEM TREINAR NADA, a partir de um checkpoint
ja existente. A observacao-chave e que a fusao do ensemble em estrela e uma
COMBINACAO CONVEXA por voxel (softmax sobre os M candidatos, ver
`model/rrin3d_star.py:RRIN3DStar.forward`). Isso impoe um teto exato e
calculavel:

  - Se o alvo cai DENTRO do intervalo [min_m pred_m, max_m pred_m] daquele
    voxel, existe uma combinacao convexa que acerta o alvo EXATAMENTE (erro
    0) -- basta a rede aprender os pesos certos.
  - Se o alvo cai FORA desse intervalo, NENHUMA combinacao convexa consegue
    alcanca-lo: o melhor erro possivel e a distancia ate a ponta mais
    proxima do intervalo.

Ou seja, `mae_oracle_convex` (abaixo) e um LIMITE INFERIOR EXATO para
qualquer melhoria concebivel na cabeca de fusao -- incluindo o item 1,
incluindo qualquer cabeca futura. A leitura e direta:

  - `mae_fusao - mae_oracle_convex` = a folga que AINDA EXISTE para a cabeca
    de fusao. Se for pequena, a cabeca ja esta perto do seu proprio teto e
    o item 1 tem pouco a oferecer.
  - `mae_oracle_convex` vs. o alvo de ~0,026 do RCAE = o veredito estrutural.
    Se o proprio oracle ja estiver ACIMA de 0,026, entao nenhuma melhoria de
    fusao, por melhor que seja, leva esta linha ao nivel do RCAE -- os
    candidatos e que precisam melhorar (leitura (B), contexto angular
    global).

ATENCAO A UMA ARMADILHA que este script evita: `min_m |pred_m - alvo|` (o
"melhor candidato") NAO e um limite inferior valido para combinacao convexa
-- uma media de dois candidatos que erram para lados opostos pode ser melhor
que qualquer um deles isoladamente. Esse numero e reportado tambem
(`mae_oracle_select`), mas como referencia de SELECAO DURA (argmax), nao
como teto da fusao. O teto correto e o `mae_oracle_convex`.

Serve as DUAS linhas star (RRIN3DStar e PairFlowStar) -- `--model` escolhe
qual, os dois tem a mesma assinatura de forward e o mesmo contrato de
`return_pairs=True`.

Uso:
    python scripts/15_diagnose_star_fusion_ceiling.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling_m8res15 \
        --checkpoint work_dir/rrin_star_checkpoints/shell1000_n16_star8/best.pt \
        --shell-b 1000 --n-level 16 --model rrin_star \
        --max-batches 60 --out-csv fusion_ceiling.csv

Requer PyTorch + o checkpoint. Nao executado neste ambiente de
desenvolvimento (torch indisponivel) -- ver secao de verificacao no addendum
2026-09-14.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.rrin_dataset import RRINTripletDataset
from model.rrin3d_star import build_star_model
from model.pairflow_star import build_pairflow_star_model


def fusion_metrics(pred, target, ensemble_mask, out_fusion, quality=None):
    """Calcula, POR VOXEL, as quantidades que interessam.

    pred: (B, M, 1, D, H, W) -- predicao de cada candidato.
    target: (B, 1, D, H, W).
    ensemble_mask: (B, M) bool -- True = candidato REAL (nao padding).
    out_fusion: (B, 1, D, H, W) -- a fusao que o modelo de fato produziu.
    quality: (B, M, 2) ou None -- [residual_deg/90, gap_deg/90]; usado so
        para o baseline REALIZAVEL `by_gap`.

    Separa deliberadamente dois tipos de numero, porque confundi-los leva a
    conclusoes erradas:

    ORACULOS (usam o alvo para escolher, logo NAO sao alcancaveis por
    nenhuma rede -- servem so como limite superior de OPORTUNIDADE):
      `convex`  teto da combinacao convexa (o limite correto da fusao)
      `select`  melhor selecao dura (NAO e teto valido -- ver docstring)

    REALIZAVEIS (nao olham o alvo; uma rede poderia, em principio,
    reproduzi-los -- servem como piso que a fusao aprendida PRECISA bater
    para estar justificando a propria existencia):
      `fusion`    a fusao aprendida
      `mean`      media uniforme entre candidatos reais
      `by_gap`    escolher sempre o candidato de menor gap_deg (regra
                  geometrica de ZERO parametros)
      `per_cand`  erro medio de UM candidato isolado -- serve para saber se
                  a media entre eles esta ajudando: se `mean` ~= `per_cand`,
                  os erros sao CORRELACIONADOS (os 8 erram juntos) e nao ha
                  ganho de ensemble nenhum; se `mean` << `per_cand`, os
                  erros sao independentes e a media ja esta fazendo o
                  trabalho.
    """
    b, m = pred.shape[0], pred.shape[1]
    mask = ensemble_mask.view(b, m, 1, 1, 1, 1)

    # --- teto da COMBINACAO CONVEXA (o numero que importa) -----------------
    # Candidatos de padding nao podem entrar nem no min nem no max: enche
    # com +inf para o min e -inf para o max, assim eles nunca vencem.
    big = torch.finfo(pred.dtype).max
    pred_for_min = torch.where(mask, pred, torch.full_like(pred, big))
    pred_for_max = torch.where(mask, pred, torch.full_like(pred, -big))
    lo = pred_for_min.min(dim=1).values                      # (B,1,D,H,W)
    hi = pred_for_max.max(dim=1).values
    # distancia do alvo ao intervalo [lo, hi]; 0 se estiver dentro.
    err_convex = torch.clamp(lo - target, min=0.0) + torch.clamp(target - hi, min=0.0)

    # --- melhor SELECAO dura (referencia, NAO e teto da fusao) -------------
    abs_err = (pred - target.unsqueeze(1)).abs()             # (B,M,1,D,H,W)
    abs_err_masked = torch.where(mask, abs_err, torch.full_like(abs_err, big))
    err_select = abs_err_masked.min(dim=1).values

    # --- media uniforme entre candidatos reais (baseline de fusao) --------
    n_real = mask.float().sum(dim=1).clamp(min=1.0)          # (B,1,1,1,1)
    pred_sum = (pred * mask.float()).sum(dim=1)
    err_mean = (pred_sum / n_real - target).abs()

    # --- o que o modelo realmente faz --------------------------------------
    err_fusion = (out_fusion - target).abs()

    # --- espalhamento entre candidatos (diagnostico auxiliar) --------------
    # Se os M candidatos concordam quase sempre, a fusao tem pouco o que
    # escolher -- outro sintoma de que o gargalo esta nos candidatos.
    spread = (hi - lo)

    # --- fracao de voxels em que o alvo esta DENTRO do intervalo ----------
    # Explica diretamente por que `convex` e alto ou baixo: se essa fracao e
    # ~1, o teto e quase 0 por construcao (existe combinacao exata em quase
    # todo voxel) e o problema nao e "os candidatos nao alcancam o alvo".
    inside = ((target >= lo) & (target <= hi)).to(pred.dtype)

    # --- erro medio de UM candidato isolado (realizavel) ------------------
    per_cand = (abs_err * mask.float()).sum(dim=1) / n_real

    # --- baseline REALIZAVEL: sempre o candidato de menor gap_deg ---------
    # Nao usa o alvo -- e uma regra geometrica pura, de zero parametros. Se a
    # fusao aprendida nao bater isso, toda a maquinaria de fusao nao esta
    # entregando nada sobre "use o par geometricamente melhor".
    if quality is not None:
        gap = quality[..., 1]                                  # (B,M)
        gap_masked = torch.where(ensemble_mask, gap, torch.full_like(gap, big))
        best_gap_idx = gap_masked.argmin(dim=1)                # (B,)
        sel_pred = pred[torch.arange(b, device=pred.device), best_gap_idx]
        err_by_gap = (sel_pred - target).abs()
    else:
        err_by_gap = torch.full_like(err_fusion, float("nan"))

    return {"fusion": err_fusion, "convex": err_convex, "select": err_select,
            "mean": err_mean, "spread": spread, "inside": inside,
            "per_cand": per_cand, "by_gap": err_by_gap}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--model", choices=["rrin_star", "pairflow_star"], default="rrin_star",
                     help="qual linha star o checkpoint e -- as duas tem a mesma assinatura de "
                          "forward, mas construtores diferentes.")
    ap.add_argument("--split", default="val",
                     help="split do manifesto a usar (default 'val' -- o MESMO que gera o "
                          "val_loss com que estes numeros vao ser comparados).")
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=60,
                     help="quantos batches de validacao percorrer (default 60). E um "
                          "diagnostico -- nao precisa da validacao inteira; 60 batches de 8 ja "
                          "dao ~480 alvos, suficiente para as medias estabilizarem.")
    ap.add_argument("--no-only-valid", action="store_true",
                     help="mesma semantica de scripts/04e_train_rrin_star.py. Para comparar com "
                          "o val_loss de um treino, use o MESMO valor usado nele (default: "
                          "only_valid=True, que e o default do treino).")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-csv", default=None,
                     help="se dado, grava as medias por batch (util pra conferir estabilidade).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    ensemble_m = ckpt_args.get("ensemble_m")
    if ensemble_m is None:
        sys.exit(f"checkpoint {args.checkpoint} nao tem 'ensemble_m' em args -- confira se e "
                  f"mesmo um checkpoint de uma das linhas star (etapa 4e/4i).")
    weight_quality_cond = ckpt_args.get("weight_quality_cond", False)
    use_quality_cond = ckpt_args.get("use_quality_cond", False)
    refine_cond = ckpt_args.get("refine_cond", False)
    need_quality = weight_quality_cond or use_quality_cond or refine_cond

    common = dict(base_ch=ckpt_args.get("base_ch", 16),
                   max_disp=ckpt_args.get("max_disp", 0.5),
                   norm_type=ckpt_args.get("norm_type", "instance"),
                   weight_quality_cond=weight_quality_cond,
                   cross_candidate_attention=ckpt_args.get("cross_candidate_attention", False),
                   cross_candidate_attn_heads=ckpt_args.get("cross_candidate_attn_heads", 4),
                   refine_base_ch=ckpt_args.get("refine_base_ch", None),
                   refine_depth=ckpt_args.get("refine_depth", 2),
                   refine_cond=refine_cond)
    if args.model == "rrin_star":
        model = build_star_model(use_quality_cond=use_quality_cond, **common).to(device)
    else:
        model = build_pairflow_star_model(freeze_flow=ckpt_args.get("freeze_flow", False),
                                           **common).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[ckpt] {args.checkpoint}", flush=True)
    print(f"[ckpt] modelo={args.model} epoca={ckpt.get('epoch')} "
          f"val_loss_registrado={ckpt.get('val_loss')} ensemble_m={ensemble_m} "
          f"base_ch={ckpt_args.get('base_ch', 16)} "
          f"cross_candidate_attention={ckpt_args.get('cross_candidate_attention', False)}",
          flush=True)

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]
    if not entries:
        sys.exit(f"nenhum sujeito no split {args.split!r} do manifesto.")
    only_valid = not args.no_only_valid
    ds = RRINTripletDataset(entries, args.triplets_dir, args.shell_b, args.n_level,
                             patch_size=args.patch_size, training=False,
                             mask_suffix=args.mask_suffix, only_valid=only_valid,
                             min_tile_coverage=args.min_tile_coverage,
                             seed=args.seed, max_cached_subjects=args.max_cached_subjects,
                             ensemble_m=ensemble_m)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)
    print(f"[dados] {len(ds.usable)} sujeitos utilizaveis, {len(ds)} patches, "
          f"only_valid={only_valid} -- percorrendo ate {args.max_batches} batches", flush=True)

    # acumuladores: soma e contagem, para media EXATA no fim (nao media de
    # medias por batch, que pesaria batches menores igual aos cheios).
    keys = ("fusion", "mean", "by_gap", "per_cand", "select", "convex", "spread", "inside")
    sums_all = {k: 0.0 for k in keys}
    sums_msk = {k: 0.0 for k in keys}
    n_all = 0
    n_msk = 0
    n_real_sum, n_real_count = 0.0, 0
    rows = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.max_batches:
                break
            vol_a = batch["vol_a_ens"].to(device)
            vol_b = batch["vol_b_ens"].to(device)
            target = batch["target"].to(device)
            bvec_a = batch["bvec_a_ens"].to(device)
            bvec_b = batch["bvec_b_ens"].to(device)
            bvec_t = batch["bvec_t_ens"].to(device)
            t_frac = batch["t_frac_ens"].to(device)
            ensemble_mask = batch["ensemble_mask"].to(device)
            # `quality` e SEMPRE lida (o dataset sempre fornece) porque o
            # baseline realizavel `by_gap` precisa dela -- mas so e PASSADA ao
            # modelo se o checkpoint foi treinado esperando-a, senao o forward
            # mudaria em relacao ao treino.
            quality_all = batch["quality_ens"].to(device)

            out, extra = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, ensemble_mask,
                                quality=quality_all if need_quality else None,
                                return_pairs=True)
            met = fusion_metrics(extra["pred"], target, ensemble_mask, out,
                                  quality=quality_all)

            # (1) sobre TODOS os voxels do patch -- mesmo denominador do
            #     val_loss do treino (que inclui o fundo), entao diretamente
            #     comparavel com os ~0,029 das curvas.
            # (2) sobre os voxels com alvo != 0 -- proxy da mascara de
            #     cerebro (o dataset zera o sinal fora da mascara, ver
            #     utils/rrin_dataset.py). E' um PROXY: um voxel de tecido com
            #     sinal exatamente 0 seria excluido por engano, o que e raro
            #     mas nao impossivel. Reportado como referencia secundaria.
            inside = (target != 0)
            n_inside = int(inside.sum().item())
            row = {"batch": i}
            for k in keys:
                v = met[k]
                sums_all[k] += float(v.sum().item())
                sums_msk[k] += float(v[inside].sum().item()) if n_inside else 0.0
                row[k] = float(v.mean().item())
            n_all += int(target.numel())
            n_msk += n_inside
            n_real_sum += float(ensemble_mask.float().sum(dim=1).sum().item())
            n_real_count += int(ensemble_mask.shape[0])
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"  ... {i + 1} batches", flush=True)

    if n_all == 0:
        sys.exit("nenhum batch processado -- confira --max-batches/--split.")

    avg_all = {k: sums_all[k] / n_all for k in keys}
    avg_msk = {k: (sums_msk[k] / n_msk if n_msk else float("nan")) for k in keys}
    print(f"\n[resumo] {len(rows)} batches, {n_all} voxels no total "
          f"({n_msk} dentro da mascara aproximada, {100.0 * n_msk / n_all:.1f}%), "
          f"media de {n_real_sum / max(n_real_count, 1):.2f}/{ensemble_m} candidatos reais "
          f"por alvo")

    def _report(avg, titulo, denom):
        print(f"\n=== {titulo} ===")
        print("  REALIZAVEIS (nao olham o alvo -- a rede pode, em principio, igualar):")
        print(f"    {'fusao aprendida (o modelo hoje)':<44} {avg['fusion']:.6f}")
        print(f"    {'media uniforme entre candidatos':<44} {avg['mean']:.6f}")
        print(f"    {'sempre o candidato de menor gap_deg':<44} {avg['by_gap']:.6f}")
        print(f"    {'UM candidato isolado (media sobre os M)':<44} {avg['per_cand']:.6f}")
        print("  ORACULOS (usam o alvo -- NAO alcancaveis, so limite de oportunidade):")
        print(f"    {'melhor selecao dura (argmax)':<44} {avg['select']:.6f}")
        print(f"    {'TETO da combinacao convexa':<44} {avg['convex']:.6f}")
        print("  AUXILIARES:")
        print(f"    {'espalhamento entre candidatos (hi-lo)':<44} {avg['spread']:.6f}")
        print(f"    {'fracao de voxels com alvo DENTRO de [lo,hi]':<44} {avg['inside']:.4f}")

        # o que o ensemble esta entregando sobre nao ter ensemble nenhum
        ganho_ens = avg["per_cand"] - avg["fusion"]
        ganho_cabeca = avg["mean"] - avg["fusion"]
        print(f"\n  Ganho do ENSEMBLE sobre 1 candidato isolado: {ganho_ens:+.6f} "
              f"({100.0 * ganho_ens / avg['per_cand']:+.1f}%)")
        print(f"  Ganho da CABECA sobre a media uniforme:      {ganho_cabeca:+.6f} "
              f"({100.0 * ganho_cabeca / avg['mean']:+.1f}%)")
        print(f"  Ganho da cabeca sobre a regra de menor gap:  "
              f"{avg['by_gap'] - avg['fusion']:+.6f}")
        folga = avg["fusion"] - avg["convex"]
        print(f"  Oportunidade TEORICA restante (ate o teto):  {folga:.6f} "
              f"({100.0 * folga / avg['fusion']:.1f}% do erro atual)")
        if denom is not None:
            gap_ref = avg["fusion"] - denom
            if gap_ref > 0:
                print(f"  (para referencia: faltam {gap_ref:.6f} ate o alvo de {denom:.3f})")

    _report(avg_all, "TODOS os voxels do patch (comparavel ao val_loss do treino)", 0.026)
    _report(avg_msk, "So dentro da mascara aproximada (alvo != 0)", None)

    print("\n=== LEITURA ===")
    a = avg_all
    folga = a["fusion"] - a["convex"]
    ganho_cabeca_rel = (a["mean"] - a["fusion"]) / a["mean"]
    ganho_ens_rel = (a["per_cand"] - a["fusion"]) / a["per_cand"]

    if a["convex"] > 0.026:
        print("  [ESTRUTURAL] O TETO da combinacao convexa ja esta ACIMA do alvo de 0,026:")
        print("  em muitos voxels o alvo cai FORA do intervalo coberto pelos M candidatos, e")
        print("  NENHUMA cabeca de fusao alcanca o que nao esta no intervalo. O gargalo sao as")
        print("  PREDICOES CANDIDATAS -> contexto angular global (leitura (B)).")
    else:
        print(f"  [NAO E ESTRUTURAL] O alvo cai dentro do intervalo dos candidatos em "
              f"{100.0 * a['inside']:.1f}% dos voxels,")
        print(f"  e o teto da fusao ({a['convex']:.6f}) esta bem abaixo do alvo de 0,026. Ou seja: os")
        print("  candidatos JA cercam a resposta. O problema nao e alcance, e escolha.")

    print()
    if ganho_ens_rel < 0.05:
        print(f"  [ATENCAO] O ensemble inteiro esta entregando so {100.0 * ganho_ens_rel:.1f}% sobre UM")
        print("  candidato isolado. Com os erros dos M candidatos independentes, a media de M")
        print("  ja deveria render bem mais -- entao os candidatos estao errando JUNTOS, para o")
        print("  mesmo lado (vies compartilhado). Um vies compartilhado NAO e corrigivel por")
        print("  combinacao convexa: o centro do intervalo esta deslocado, e a fusao so pode")
        print("  andar dentro do intervalo. Isso empurra de volta para melhorar os CANDIDATOS.")
    else:
        print(f"  O ensemble entrega {100.0 * ganho_ens_rel:.1f}% sobre um candidato isolado -- a media entre")
        print("  candidatos ja esta fazendo trabalho real (erros ao menos parcialmente independentes).")

    print()
    if ganho_cabeca_rel < 0.02:
        print(f"  [ATENCAO] A cabeca de fusao APRENDIDA esta so {100.0 * ganho_cabeca_rel:.2f}% melhor que uma media")
        print("  uniforme. Depois de um treino inteiro, ela praticamente nao descobriu em quem")
        print("  confiar. Duas explicacoes possiveis, e elas pedem acoes diferentes:")
        print("    (i)  a cabeca e CEGA/pequena demais: 1.777 parametros com base_ch=16, e sem")
        print("         --weight-quality-cond ela nem enxerga gap_deg/residual_deg. Testavel e")
        print("         barato: CROSS_CANDIDATE_ATTENTION=1 WEIGHT_QUALITY_COND=1 (+BASE_CH).")
        print("    (ii) o erro residual e imprevisivel a partir da entrada -- nenhuma cabeca")
        print("         resolve, e a saida e melhorar os candidatos.")
        print("  Compare com a linha 'sempre o candidato de menor gap_deg' acima: se essa regra")
        print("  de ZERO parametros ja empata ou ganha da cabeca aprendida, e sinal forte de (i).")
    else:
        print(f"  A cabeca de fusao esta {100.0 * ganho_cabeca_rel:.1f}% melhor que a media uniforme -- ela aprendeu")
        print("  algo real sobre em quem confiar, e dar mais capacidade/informacao a ela (item 1,")
        print("  WEIGHT_QUALITY_COND) tende a render mais.")

    print("\n  RESSALVA IMPORTANTE sobre o teto: `TETO da combinacao convexa` e `melhor selecao")
    print("  dura` sao ORACULOS -- escolhem os pesos JA SABENDO o alvo. Nenhuma rede pode")
    print("  alcanca-los, porque no uso real o alvo e justamente o que se quer predizer. Eles")
    print("  medem OPORTUNIDADE (quanto de informacao existe no conjunto de candidatos), nao")
    print("  desempenho atingivel. O teto baixo NAO significa 'da pra chegar la com uma cabeca")
    print("  melhor'; significa 'a informacao esta la, resta saber se e extraivel da ENTRADA'.")
    print("  Os numeros REALIZAVEIS (media uniforme, menor gap_deg) e que dizem o que uma regra")
    print("  sem acesso ao alvo consegue de fato.")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["batch"] + list(keys))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[saida] medias por batch em {args.out_csv}")


if __name__ == "__main__":
    main()