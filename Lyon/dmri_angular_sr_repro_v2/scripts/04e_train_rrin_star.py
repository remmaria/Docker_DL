#!/usr/bin/env python3
"""
Etapa 4e ("ensemble em estrela", ver protocolo secao 14.5 item 1 e addendum
2026-08-27): treina a RRIN3DStar (model/rrin3d_star.py) para um
(shell, n_level) especifico, usando os feixes de M pares diversos ja
construidos por scripts/02b_build_rrin_triplets.py --ensemble-m M.

Ao contrario de scripts/04b_train_rrin.py (RRIN3D/RRIN3DLayered, UM par de
entrada por chamada do modelo), aqui cada item do dataset devolve ATE
--ensemble-m pares DIFERENTES para o MESMO alvo (ver
utils/rrin_dataset.py:RRINTripletDataset.ensemble_m e
utils/gradients.py:find_star_ensemble_batch) -- a RRIN3DStar roda o mesmo
pipeline de fluxo+warp+refino (pesos compartilhados) em cada par e funde as
predicoes por um softmax aprendido POR VOXEL (ver docstring de
model/rrin3d_star.py:RRIN3DStar para a motivacao completa).

Espelha bastante scripts/04b_train_rrin.py (mesmo manifesto/split, mesma
normalizacao por percentil, mesmo layout de checkpoint out_dir/<shell>_<n>/
{best,last}.pt, mesmo resume automatico) -- **DELIBERADAMENTE SEM** o termo
de loss angular/SH (--angular-loss-weight de 04b_train_rrin.py) nesta
primeira versao: combinar o feixe-para-loss-SH (sh_q_out, trincas
DIFERENTES do mesmo sujeito) com o feixe-para-ensemble-em-estrela (M pares
do MESMO alvo) exigiria um dataset com DOIS feixes simultaneos e nao foi
implementado ainda -- TODO real (mesmo padrao ja usado em
scripts/04d_train_hfd.py, que tambem entregou sua primeira versao sem a
loss angular/SH portada). Sem esse termo, a loss aqui e so MAE de sinal,
identica ao RRIN3D "cego" de sempre.

Uso:
    python scripts/04e_train_rrin_star.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --out-dir work_dir/rrin_star_checkpoints \
        --ensemble-m 3 \
        --epochs 150 --batch-size 8 --patch-size 10 --lr 1e-3

Requer PyTorch + GPU. Nao executado neste ambiente de desenvolvimento --
revisado manualmente, testado por compilacao de sintaxe e por um teste
sintetico (numpy puro, monkeypatch de I/O) do dataset/lógica de batelamento
-- ver utils/rrin_dataset.py e model/rrin3d_star.py.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.rrin_dataset import RRINTripletDataset
from utils.dataset import SubjectGroupedSampler, worker_init_fn
from model.rrin3d_star import build_star_model


def _masked_residual_l2(residual, ensemble_mask):
    """Penalidade L2 do residual do RefineNet3D, media SO sobre as posicoes
    REAIS do feixe (ensemble_mask=True) -- posicoes de padding tem volumes
    zerados e nao devem contar (ver addendum 2026-09-13, item 3 da resposta
    sobre RCAE vs. familia de fluxo: encolher o residual em direcao ao
    blend/warp bruto, mesmo espirito do --zero-init-refine-output, so que
    como penalidade continua durante o treino em vez de so na inicializacao).
    residual: (B,M,1,D,H,W). ensemble_mask: (B,M) bool."""
    mask_f = ensemble_mask.view(*ensemble_mask.shape, 1, 1, 1, 1).float()
    voxels_per_pair = residual.shape[2] * residual.shape[3] * residual.shape[4] * residual.shape[5]
    n_valid_elems = mask_f.sum() * voxels_per_pair
    return (residual.pow(2) * mask_f).sum() / n_valid_elems.clamp(min=1.0)


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int,
              need_quality: bool = False, batch_log_f=None,
              residual_l2_weight: float = 0.0) -> tuple:
    model.train(mode=train)
    total_loss = 0.0
    total_loss_mae = 0.0
    total_loss_residual = 0.0
    n_batches = 0
    n_samples = 0
    total_wait_s = 0.0
    total_compute_s = 0.0
    split = "train" if train else "val"
    prev_end = time.time()
    for batch in loader:
        t_received = time.time()
        wait_s = t_received - prev_end

        vol_a = batch["vol_a_ens"].to(device)          # (B,M,1,ps,ps,ps)
        vol_b = batch["vol_b_ens"].to(device)
        target = batch["target"].to(device)            # (B,1,ps,ps,ps) -- MESMO alvo p/ todo o feixe
        bvec_a = batch["bvec_a_ens"].to(device)         # (B,M,3)
        bvec_b = batch["bvec_b_ens"].to(device)
        bvec_t = batch["bvec_t_ens"].to(device)
        t_frac = batch["t_frac_ens"].to(device)         # (B,M)
        ensemble_mask = batch["ensemble_mask"].to(device)  # (B,M) bool
        # need_quality = use_quality_cond OU weight_quality_cond (ver model/rrin3d_star.py) --
        # o MESMO tensor `quality_ens` alimenta as duas condicoes, independentemente de qual(is)
        # esta(o) ligada(s); o modelo decide internamente qual sub-modulo de fato usa.
        quality = batch["quality_ens"].to(device) if need_quality else None

        with torch.set_grad_enabled(train):
            if residual_l2_weight > 0.0:
                pred, extra = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, ensemble_mask,
                                     quality=quality, return_pairs=True)
                loss_residual = _masked_residual_l2(extra["residual"], ensemble_mask)
            else:
                pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, ensemble_mask,
                             quality=quality)
                loss_residual = torch.zeros((), device=device)
            # MAE, mesma escolha de 04b_train_rrin.py/04_train_rcae.py -- sem
            # mascara aqui porque "target" ja tem shape fixo (a fusao ja
            # aconteceu dentro do modelo, o alvo em si e um UNICO volume).
            loss_mae = (pred - target).abs().mean()
            loss = loss_mae + residual_l2_weight * loss_residual
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        t_compute_end = time.time()
        compute_s = t_compute_end - t_received

        total_loss += loss.item()
        total_loss_mae += loss_mae.item()
        total_loss_residual += loss_residual.item()
        n_batches += 1
        n_samples += vol_a.shape[0]
        total_wait_s += wait_s
        total_compute_s += compute_s

        if batch_log_f is not None:
            tags_str = ";".join(batch["subject_tag"])
            n_real_mean = ensemble_mask.float().sum(dim=1).mean().item()
            batch_log_f.write(f"{epoch},{split},{n_batches},{loss.item():.6f},"
                               f"{wait_s:.3f},{compute_s:.3f},{tags_str},{n_real_mean:.2f}\n")
            batch_log_f.flush()

        prev_end = time.time()

    if n_batches > 0:
        total_s = total_wait_s + total_compute_s
        throughput = n_samples / total_s if total_s > 0 else float("nan")
        pct_wait = 100 * total_wait_s / total_s if total_s > 0 else float("nan")
        print(f"[{split}] epoca {epoch} resumo: {n_batches} batches, {n_samples} patches | "
              f"wait total {total_wait_s:.1f}s ({pct_wait:.0f}%) | compute total "
              f"{total_compute_s:.1f}s | {throughput:.2f} patches/s", flush=True)

    n = max(1, n_batches)
    return total_loss / n, total_loss_mae / n, total_loss_residual / n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="pasta com os <tag>_rrin_triplets.npz da etapa 2b -- PRECISA ter "
                          "sido gerada com --ensemble-m >= --ensemble-m deste script")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--ensemble-m", type=int, default=3,
                     help="M do ensemble em estrela (ver utils/gradients.py:"
                          "find_star_ensemble_batch e model/rrin3d_star.py) -- quantos pares "
                          "diversos por alvo o dataset devolve e o modelo funde. Default 3 "
                          "(mesmo default recomendado em scripts/02b_build_rrin_triplets.py "
                          "--ensemble-m).")
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--max-disp", type=float, default=0.5)
    ap.add_argument("--use-quality-cond", action="store_true",
                     help="mesmo espirito de --use-quality-cond em 04b_train_rrin.py, aplicado "
                          "a cada par do feixe (ver model/rrin3d_star.py) -- condiciona o "
                          "FlowNet3D (a estimativa de fluxo em si). DESACOPLADO de "
                          "--weight-quality-cond (condiciona a fusao, nao o fluxo) -- as duas "
                          "podem ser ligadas independentemente.")
    ap.add_argument("--weight-quality-cond", action="store_true",
                     help="NOVO (discussao pos-secao 20.9 do addendum): alimenta residual_deg/"
                          "gap_deg de cada par diretamente na PairWeightHead3D (a cabeca que "
                          "decide o peso de fusao entre os M pares), em vez de deixar a cabeca "
                          "inferir confiabilidade so pelo conteudo de imagem -- ver docstring "
                          "de model/rrin3d_star.py:PairWeightHead3D. Independente de "
                          "--use-quality-cond (que condiciona o FlowNet3D, nao a fusao).")
    ap.add_argument("--residual-l2-weight", type=float, default=0.0,
                     help="ADITIVO, default 0.0 = desligado (addendum 2026-09-13, hipotese sobre "
                          "por que o RCAE bate a familia de fluxo: RCAE tem um prior de "
                          "suavidade forte -- ver troca 'nmse/FA melhor, energia SH de ordem "
                          "alta pior' documentada no addendum 2026-08-27 secoes 20.16/20.17). "
                          "Penaliza a MAGNITUDE do residuo do RefineNet3D (media so sobre as "
                          "posicoes REAIS do feixe, ver _masked_residual_l2), empurrando a "
                          "predicao a ficar perto do blend/warp bruto a menos que haja evidencia "
                          "forte pra se afastar dele -- mesmo espirito de --zero-init-refine-"
                          "output (ja existente no PairFlowStar), so que como penalidade continua "
                          "durante o treino, nao so na inicializacao. NAO muda o shape dos pesos "
                          "-- nao e bloqueante para resume, so muda o objetivo de treino (mesmo "
                          "padrao de --angular-loss-weight noutras linhas). Ganha sufixo "
                          "_resl2<valor> no run_tag.")
    ap.add_argument("--refine-base-ch", type=int, default=None,
                     help="NOVO (2026-09-14, ver addendum de capacidade): largura da RefineNet3D "
                          "DESACOPLADA da largura do resto do modelo. Default None = segue "
                          "--base-ch (comportamento historico). Motivacao: com --base-ch 16 a "
                          "RefineNet3D tem 8.737 parametros (4,7% do modelo) e e a UNICA parte "
                          "nao presa a premissa de correspondencia espacial entre direcoes "
                          "(secao 14.6 do protocolo) -- e' nela que qualquer erro dessa premissa "
                          "teria que ser corrigido livremente. MUDA O SHAPE DOS PESOS -- "
                          "bloqueante para resume. Ganha sufixo _rbc<N> no run_tag.")
    ap.add_argument("--refine-depth", type=int, default=2,
                     help="NOVO: numero de camadas ocultas da RefineNet3D (default 2 = topologia "
                          "historica, byte-a-byte). Comparacao de referencia: o decoder do RCAE "
                          "(o modelo que hoje vence) tem 10 convolucoes; esta rede tem 2 ocultas "
                          "+ 1 de saida. MUDA O SHAPE DOS PESOS -- bloqueante para resume. Ganha "
                          "sufixo _rd<N> no run_tag.")
    ap.add_argument("--refine-cond", action="store_true",
                     help="NOVO: injeta bvec_a/bvec_b/bvec_t/t_frac/quality (12 canais constantes, "
                          "ver model/rrin3d_star.py:REFINE_COND_CH) na entrada da RefineNet3D. "
                          "Hoje ela NAO recebe nenhuma informacao de qual direcao esta predizendo "
                          "nem de quao confiavel e a geometria do par -- enquanto o RCAE reinjeta "
                          "o bvec do alvo em TODOS os estagios do decoder. Liga `need_quality` "
                          "automaticamente. MUDA O SHAPE DOS PESOS -- bloqueante para resume. "
                          "Ganha sufixo _rcond no run_tag.")
    ap.add_argument("--cross-candidate-attention", action="store_true",
                     help="NOVO (item 1 do addendum 2026-09-13, ver model/rrin3d_star.py:"
                          "CrossCandidateAttention3D): insere self-attention entre os M "
                          "candidatos do feixe (por voxel) ANTES do logit de confianca da "
                          "PairWeightHead3D -- hoje cada candidato e julgado de forma totalmente "
                          "INDEPENDENTE dos outros M-1 antes da normalizacao final (softmax "
                          "mascarado). ADITIVO (default desligado), MUDA O SHAPE DOS PESOS -- "
                          "bloqueante para --resume-checkpoint se mudar entre chamadas. Ganha "
                          "sufixo _cattn no run_tag.")
    ap.add_argument("--cross-candidate-attn-heads", type=int, default=4,
                     help="numero de cabecas de atencao de CrossCandidateAttention3D (so tem "
                          "efeito com --cross-candidate-attention) -- precisa dividir --base-ch "
                          "sem resto (ValueError cedo na construcao do modelo, senao). Default 4, "
                          "mesmo default de --cross-attn-heads em scripts/04f_train_implicit.py.")
    ap.add_argument("--norm-type", choices=["instance", "batch"], default="instance",
                     help="ver docstring de _norm3d em model/rrin3d.py -- mesmo comportamento/"
                          "restricoes de scripts/04b_train_rrin.py (batch exige treino do zero).")
    ap.add_argument("--no-only-valid", action="store_true",
                     help="mesmo espirito de --no-only-valid em 04b_train_rrin.py -- treina/"
                          "valida tambem com alvos cujo par-UNICO e invalido (o feixe do "
                          "ensemble pode ainda assim ter pares uteis dentro do teto; ver "
                          "utils/rrin_dataset.py). Default (so validos) mantem a mesma cautela "
                          "ja documentada la sobre explosao numerica em alvos sem par valido.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2)
    ap.add_argument("--val-num-workers", type=int, default=None)
    ap.add_argument("--val-max-cached-subjects", type=int, default=1)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-id", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--resume-checkpoint", default=None)
    ap.add_argument("--reset-lr", action="store_true",
                     help="Retoma os PESOS do checkpoint mas recria otimizador e scheduler do zero, no --lr desta chamada (reinicio a quente). Sem esta flag, o resume restaura optimizer_state (que carrega os momentos do Adam E o LR corrente) e scheduler_state, entao o --lr da linha de comando e' IGNORADO na pratica. Uso tipico (addendum 2026-09-14d): o LR foi escolhido no regime de modelo pequeno e ficou grande demais depois de escalar a largura. Mantem best_val (o best.pt so' e' trocado por melhora real) e zera epochs_no_improve.")
    args = ap.parse_args()

    if args.ensemble_m < 1:
        raise ValueError(f"--ensemble-m deve ser >= 1 (recebido {args.ensemble_m})")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, "| job_id:", args.job_id or "(nao informado)")

    entries = load_manifest(args.manifest)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    only_valid = not args.no_only_valid
    print(f"[resumo] only_valid={only_valid}, ensemble_m={args.ensemble_m}")
    train_ds = RRINTripletDataset(train_entries, args.triplets_dir, args.shell_b, args.n_level,
                                   patch_size=args.patch_size, training=True,
                                   mask_suffix=args.mask_suffix, only_valid=only_valid,
                                   min_tile_coverage=args.min_tile_coverage,
                                   seed=args.seed, max_cached_subjects=args.max_cached_subjects,
                                   ensemble_m=args.ensemble_m)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = RRINTripletDataset(val_entries, args.triplets_dir, args.shell_b, args.n_level,
                                 patch_size=args.patch_size, training=False,
                                 mask_suffix=args.mask_suffix, only_valid=only_valid,
                                 min_tile_coverage=args.min_tile_coverage,
                                 seed=args.seed + 1, max_cached_subjects=args.val_max_cached_subjects,
                                 ensemble_m=args.ensemble_m)

    persistent_train = args.num_workers > 0
    persistent_val = val_num_workers > 0
    train_sampler = SubjectGroupedSampler(train_ds, seed=args.seed,
                                           num_workers=args.num_workers, batch_size=args.batch_size)
    if args.num_workers > 1:
        print(f"[dataloader] SubjectGroupedSampler particionado por worker "
              f"(num_workers={args.num_workers}, batch_size={args.batch_size}) -- "
              f"elimina releitura redundante de sujeito entre workers na mesma epoca "
              f"(ver utils.dataset.SubjectGroupedSampler.__init__, addendum 2026-09-03)", flush=True)
    winit = worker_init_fn if args.num_workers > 0 else None
    winit_val = worker_init_fn if val_num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=args.num_workers, drop_last=True,
                               persistent_workers=persistent_train, worker_init_fn=winit)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=val_num_workers,
                             persistent_workers=persistent_val, worker_init_fn=winit_val)

    print(f"[resumo] treino: {len(train_ds.usable)} sujeitos utilizaveis "
          f"({len(train_ds)} patches, {len(train_loader)} batches/epoca)")
    print(f"[resumo] val:    {len(val_ds.usable)} sujeitos utilizaveis "
          f"({len(val_ds)} patches, {len(val_loader)} batches/epoca)", flush=True)

    # --refine-cond precisa do tensor `quality` no forward, entao entra no
    # need_quality junto com as duas flags de qualidade ja existentes.
    need_quality = args.use_quality_cond or args.weight_quality_cond or args.refine_cond
    model = build_star_model(base_ch=args.base_ch, max_disp=args.max_disp,
                              use_quality_cond=args.use_quality_cond,
                              norm_type=args.norm_type,
                              weight_quality_cond=args.weight_quality_cond,
                              cross_candidate_attention=args.cross_candidate_attention,
                              cross_candidate_attn_heads=args.cross_candidate_attn_heads,
                              refine_base_ch=args.refine_base_ch,
                              refine_depth=args.refine_depth,
                              refine_cond=args.refine_cond).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_refine = sum(p.numel() for p in model.refine_net.parameters())
    n_flow = sum(p.numel() for p in model.flow_net.parameters())
    print(f"[resumo] RRIN3DStar: {n_params} parametros (base_ch={args.base_ch}, "
          f"ensemble_m={args.ensemble_m}, use_quality_cond={args.use_quality_cond}, "
          f"weight_quality_cond={args.weight_quality_cond}, norm_type={args.norm_type}, "
          f"cross_candidate_attention={args.cross_candidate_attention}, "
          f"cross_candidate_attn_heads={args.cross_candidate_attn_heads}, "
          f"refine_base_ch={model.refine_base_ch}, refine_depth={args.refine_depth}, "
          f"refine_cond={args.refine_cond})")
    # quebra por submodulo -- referencia util pra comparar com o RCAE (~6,8M
    # parametros, ver addendum de capacidade 2026-09-14): historicamente esta
    # linha rodou com ~185k no total, dos quais so' ~8,7k na refine_net.
    print(f"[resumo] parametros por submodulo: flow_net={n_flow}, refine_net={n_refine}, "
          f"weight_head={sum(p.numel() for p in model.weight_head.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=5)

    print("[sanity] testando 1 batch de treino + 1 de validacao antes do loop de epocas...",
          flush=True)

    def _sanity_step(loader, split_name, do_backward):
        t0 = time.time()
        batch = next(iter(loader))
        vol_a = batch["vol_a_ens"].to(device)
        vol_b = batch["vol_b_ens"].to(device)
        target = batch["target"].to(device)
        bvec_a = batch["bvec_a_ens"].to(device)
        bvec_b = batch["bvec_b_ens"].to(device)
        bvec_t = batch["bvec_t_ens"].to(device)
        t_frac = batch["t_frac_ens"].to(device)
        ensemble_mask = batch["ensemble_mask"].to(device)
        quality = batch["quality_ens"].to(device) if need_quality else None
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            if args.residual_l2_weight > 0.0:
                pred, extra = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, ensemble_mask,
                                     quality=quality, return_pairs=True)
                loss_residual = _masked_residual_l2(extra["residual"], ensemble_mask)
            else:
                pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, ensemble_mask,
                             quality=quality)
                loss_residual = torch.zeros((), device=device)
            loss = (pred - target).abs().mean() + args.residual_l2_weight * loss_residual
            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        n_real_mean = ensemble_mask.float().sum(dim=1).mean().item()
        print(f"[sanity] {split_name} OK ({time.time() - t0:.1f}s, loss={loss.item():.6f}, "
              f"pares_reais_media={n_real_mean:.2f}/{args.ensemble_m}, "
              f"sujeitos={sorted(set(batch['subject_tag']))})", flush=True)

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- comecando o loop de epocas de verdade", flush=True)

    # run_tag: mesma disciplina de scripts/04b_train_rrin.py -- toda variante
    # que muda o comportamento treinavel precisa de um sufixo proprio, senao
    # duas rodadas diferentes colidem silenciosamente no mesmo checkpoint
    # (classe de bug ja corrigida varias vezes nesta linha, ver protocolo
    # secao 12/14.1 e addendum 2026-08-27 secao 3).
    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}_star{args.ensemble_m}"
    if args.use_quality_cond:
        run_tag += "_qc"
    if args.weight_quality_cond:
        run_tag += "_wqc"
    if not only_valid:
        run_tag += "_inclinv"
    if args.norm_type == "batch":
        run_tag += "_bn"
    if args.residual_l2_weight > 0.0:
        run_tag += f"_resl2{args.residual_l2_weight:g}"
    if args.base_ch != 16:
        run_tag += f"_bc{args.base_ch}"
    if args.refine_base_ch is not None and args.refine_base_ch != args.base_ch:
        run_tag += f"_rbc{args.refine_base_ch}"
    if args.refine_depth != 2:
        run_tag += f"_rd{args.refine_depth}"
    if args.refine_cond:
        run_tag += "_rcond"
    if args.cross_candidate_attention:
        run_tag += "_cattn"
        if args.cross_candidate_attn_heads != 4:
            run_tag += f"{args.cross_candidate_attn_heads}h"
    if abs(args.lr - 1e-3) > 1e-12:
        run_tag += f"_lr{args.lr:g}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt -- caminho fixo, "
          f"usado pela etapa 5f)")
    print(f"[resumo] logs deste run em: {run_dir}")

    start_epoch = 1
    best_val = float("inf")
    epochs_no_improve = 0
    resume_ckpt_path = None
    if not args.no_resume:
        if args.resume_checkpoint:
            resume_ckpt_path = Path(args.resume_checkpoint)
        elif (out_dir / "last.pt").exists():
            resume_ckpt_path = out_dir / "last.pt"

    if resume_ckpt_path is not None:
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"--resume-checkpoint {resume_ckpt_path} nao existe")
        print(f"[resume] carregando checkpoint existente: {resume_ckpt_path}", flush=True)
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        old_args = ckpt.get("args", {})
        # base_ch saiu desta lista de AVISO em 2026-09-14 -- virou bloqueante
        # logo abaixo (muda shape de peso; avisar e deixar morrer depois dentro
        # do load_state_dict era pior que falhar cedo com mensagem clara).
        for key in ("shell_b", "n_level", "patch_size", "max_disp", "use_quality_cond",
                    "weight_quality_cond", "ensemble_m", "norm_type", "lr"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional.",
                      flush=True)
        old_norm_type = old_args.get("norm_type", "instance")
        if old_norm_type != args.norm_type:
            raise ValueError(
                f"--norm-type={args.norm_type} nao bate com o checkpoint ({old_norm_type}) -- "
                f"norm_type nao e retomavel entre variantes. Use --no-resume ou um --out-dir/"
                f"--norm-type novos para treinar a variante '{args.norm_type}' do zero.")
        # base_ch/refine_* mudam o SHAPE dos pesos -- bloqueantes (antes da
        # correcao de 2026-09-14, base_ch so' gerava aviso e o treino morria
        # depois com um erro cripitico de shape dentro de load_state_dict).
        for key, default in (("base_ch", 16), ("refine_depth", 2)):
            old_val = old_args.get(key, default)
            new_val = vars(args).get(key)
            if old_val != new_val:
                raise ValueError(
                    f"--{key.replace('_','-')}={new_val} nao bate com o checkpoint ({old_val}) -- "
                    f"muda o shape dos pesos, nao e retomavel entre variantes. Use --no-resume ou "
                    f"um --out-dir novo.")
        old_refine_bc = old_args.get("refine_base_ch", None)
        old_refine_bc_eff = old_refine_bc if old_refine_bc is not None else old_args.get("base_ch", 16)
        new_refine_bc_eff = args.refine_base_ch if args.refine_base_ch is not None else args.base_ch
        if old_refine_bc_eff != new_refine_bc_eff:
            raise ValueError(
                f"--refine-base-ch efetivo ({new_refine_bc_eff}) nao bate com o checkpoint "
                f"({old_refine_bc_eff}) -- muda o shape dos pesos da RefineNet3D, nao e "
                f"retomavel. Use --no-resume ou um --out-dir novo.")
        old_refine_cond = old_args.get("refine_cond", False)
        if old_refine_cond != args.refine_cond:
            raise ValueError(
                f"--refine-cond={args.refine_cond} nao bate com o checkpoint ({old_refine_cond}) "
                f"-- muda o numero de canais de entrada da RefineNet3D, nao e retomavel. Use "
                f"--no-resume ou um --out-dir novo.")
        old_cross_candidate_attn = old_args.get("cross_candidate_attention", False)
        if old_cross_candidate_attn != args.cross_candidate_attention:
            raise ValueError(
                f"--cross-candidate-attention={args.cross_candidate_attention} nao bate com o "
                f"checkpoint ({old_cross_candidate_attn}) -- muda o shape dos pesos da "
                f"PairWeightHead3D (ver model/rrin3d_star.py:CrossCandidateAttention3D), nao e "
                f"retomavel entre variantes. Use --no-resume ou um --out-dir novo.")
        if args.cross_candidate_attention:
            old_cattn_heads = old_args.get("cross_candidate_attn_heads", 4)
            if old_cattn_heads != args.cross_candidate_attn_heads:
                raise ValueError(
                    f"--cross-candidate-attn-heads={args.cross_candidate_attn_heads} nao bate com "
                    f"o checkpoint ({old_cattn_heads}) -- muda o shape dos pesos de "
                    f"nn.MultiheadAttention, nao e retomavel entre variantes. Use --no-resume ou "
                    f"um --out-dir novo.")
        model.load_state_dict(ckpt["model_state"])
        if args.reset_lr:
            # Reinicio a quente: pesos sim, estado de otimizacao nao. O
            # optimizer_state do Adam carrega os momentos E o LR corrente; o
            # scheduler_state carrega o contador de plateau. Restaurar os dois
            # e' justamente o que faz o --lr desta chamada nao ter efeito.
            _old_lr = ckpt.get("args", {}).get("lr")
            print(f"[reset-lr] descartando optimizer_state/scheduler_state do checkpoint "
                  f"-- otimizador e scheduler recriados com lr={args.lr:.2e}"
                  + (f" (o checkpoint vinha de lr={_old_lr})" if _old_lr is not None else ""),
                  flush=True)
        else:
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        if args.reset_lr and epochs_no_improve:
            print(f"[reset-lr] zerando epochs_no_improve ({epochs_no_improve} -> 0) -- o "
                  f"modelo precisa de algumas epocas pra assentar no LR novo. "
                  f"best_val={best_val:.6f} e MANTIDO, entao o best.pt so' e' substituido "
                  f"por uma melhora real.", flush=True)
            epochs_no_improve = 0
        print(f"[resume] retomando da epoca {start_epoch} (best_val={best_val:.6f}, "
              f"epochs_no_improve={epochs_no_improve})", flush=True)
        if start_epoch > args.epochs:
            print(f"[resume] epoca de retomada ({start_epoch}) ja passa de --epochs "
                  f"({args.epochs}) -- nada a fazer.", flush=True)
    else:
        print("[resume] nenhum checkpoint anterior encontrado (ou --no-resume) -- "
              "comecando do zero.", flush=True)
        if args.reset_lr:
            # Nao e' erro, mas quase sempre significa que o --resume-checkpoint
            # nao chegou ou que veio um --no-resume junto por engano -- e nesse
            # caso as epocas que a flag deveria aproveitar somem em silencio.
            print("[reset-lr][aviso] --reset-lr passado, mas nao ha checkpoint pra retomar: "
                  "nao ha estado de otimizacao a descartar e o treino comeca do zero no "
                  f"--lr={args.lr:.2e}. Se a intencao era o reinicio A QUENTE, confira o "
                  "--resume-checkpoint (e nao passe --no-resume junto).", flush=True)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_loss_mae,train_loss_residual,"
                 "val_loss,val_loss_mae,val_loss_residual,lr\n")
    batch_log_path = run_dir / "batch_log.csv"
    batch_log_f = open(batch_log_path, "w")
    batch_log_f.write("epoch,split,batch,loss,wait_s,compute_s,subject_tags,n_pares_reais_media\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss, train_loss_mae, train_loss_residual = run_epoch(
                model, train_loader, optimizer, device, train=True,
                epoch=epoch, need_quality=need_quality, batch_log_f=batch_log_f,
                residual_l2_weight=args.residual_l2_weight)
            val_loss, val_loss_mae, val_loss_residual = run_epoch(
                model, val_loader, optimizer, device, train=False,
                epoch=epoch, need_quality=need_quality, batch_log_f=batch_log_f,
                residual_l2_weight=args.residual_l2_weight)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{train_loss_mae:.6f},{train_loss_residual:.6f},"
                        f"{val_loss:.6f},{val_loss_mae:.6f},{val_loss_residual:.6f},{current_lr:.2e}\n")
            if args.residual_l2_weight > 0.0:
                print(f"epoch {epoch:03d} | train {train_loss:.6f} (mae {train_loss_mae:.6f}, "
                      f"resid {train_loss_residual:.6f}) | val {val_loss:.6f} (mae "
                      f"{val_loss_mae:.6f}, resid {val_loss_residual:.6f}) | lr {current_lr:.2e}")
            else:
                print(f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
                      f"lr {current_lr:.2e}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "args": vars(args),
                    "epoch": epoch, "val_loss": val_loss, "best_val": best_val,
                    "epochs_no_improve": epochs_no_improve,
                }, out_dir / "best.pt")
                shutil.copy2(out_dir / "best.pt", run_dir / "best.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping na epoca {epoch} (sem melhora ha {args.patience} epocas)")
                    break

            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "args": vars(args), "epoch": epoch,
                        "val_loss": val_loss, "best_val": best_val,
                        "epochs_no_improve": epochs_no_improve}, out_dir / "last.pt")
            shutil.copy2(out_dir / "last.pt", run_dir / "last.pt")
    finally:
        batch_log_f.close()

    print("Treino concluido. Melhor val_loss:", best_val, "-> checkpoint em", out_dir / "best.pt")
    print(f"Copia permanente deste run em: {run_dir / 'best.pt'} (job_id={run_id})")


if __name__ == "__main__":
    main()