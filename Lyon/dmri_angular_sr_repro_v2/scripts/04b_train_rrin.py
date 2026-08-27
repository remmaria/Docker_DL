#!/usr/bin/env python3
"""
Etapa 4b (linha original da tese, retomada como diagnostico quantitativo --
ver protocolo, secao 10.1): treina a RRIN3D (model/rrin3d.py) para um
(shell, nivel de subamostragem) especifico, usando as trincas ja
construidas por scripts/02b_build_rrin_triplets.py.

Espelha bastante scripts/04_train_rcae.py (mesmo manifesto/split, mesma
normalizacao por percentil, mesmo layout de checkpoint out_dir/<shell>_<n>/
{best,last}.pt, mesmo resume automatico), mas SIMPLIFICADO de proposito
(sem os PNGs de debug por patch fixo -- ver docstring de model/rrin3d.py
para o porque desta rede ser mantida enxuta).

**Termo de loss angular/SH (--angular-loss-weight, ver protocolo secao
9/14.5/15):** portado de scripts/04_train_rcae.py (utils/sh_angular_loss.py,
modulo compartilhado). Diferenca estrutural importante em relacao ao RCAE:
cada item do RRIN preve UMA UNICA direcao-alvo por chamada do modelo (nao
uma sequencia N_in->N_out como o RCAE), entao nao da pra montar a base SH
com as direcoes de UM item so. A solucao (ver utils/rrin_dataset.py,
`sh_q_out`): quando --angular-loss-weight > 0, cada item do dataset TAMBEM
devolve um "feixe" de ate `--sh-loss-q-out` trincas adicionais do MESMO
sujeito/patch; este script empilha essas trincas (rodando o RRIN uma vez
por trinca do feixe, em um unico forward batched) e so ENTAO monta o
tensor (B, sh_q_out, ...) que compute_sh_angular_loss espera -- a mesma
funcao usada pelo RCAE, sem nenhuma modificacao.

Uso:
    python scripts/04b_train_rrin.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 10 \
        --out-dir work_dir/rrin_checkpoints \
        --epochs 100 --batch-size 8 --patch-size 10 --lr 1e-4

Requer PyTorch + GPU. Nao executado neste ambiente de desenvolvimento.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.rrin_dataset import RRINTripletDataset
from utils.dataset import SubjectGroupedSampler, worker_init_fn
from utils.sh_basis import max_order_for_n_directions
from utils.sh_angular_loss import n_coeffs_even, compute_sh_angular_loss
from model.rrin3d import build_rrin_model


def _sh_bundle_forward(model, batch, device, use_quality_cond: bool):
    """Roda o modelo sobre o feixe `*_sh` do batch (ver utils/rrin_dataset.py,
    `sh_q_out`) e devolve (pred_sh, target_sh, bvec_t_sh, sh_mask) no
    formato (B, K, ...) que utils.sh_angular_loss.compute_sh_angular_loss
    espera. As K trincas do feixe sao empilhadas no eixo de batch (B*K)
    para UM UNICO forward do modelo (nao um loop Python de K chamadas) --
    RRIN3D/RRIN3DLayered ja processam cada item do batch de forma
    independente, entao isso e equivalente e bem mais rapido."""
    vol_a_sh = batch["vol_a_sh"].to(device)      # (B, K, 1, ps, ps, ps)
    vol_b_sh = batch["vol_b_sh"].to(device)
    target_sh = batch["target_sh"].to(device)
    bvec_a_sh = batch["bvec_a_sh"].to(device)    # (B, K, 3)
    bvec_b_sh = batch["bvec_b_sh"].to(device)
    bvec_t_sh = batch["bvec_t_sh"].to(device)
    t_frac_sh = batch["t_frac_sh"].to(device)    # (B, K)
    sh_mask = batch["sh_mask"].to(device)        # (B, K) bool

    B, K = vol_a_sh.shape[0], vol_a_sh.shape[1]
    vol_a_flat = vol_a_sh.reshape(B * K, *vol_a_sh.shape[2:])
    vol_b_flat = vol_b_sh.reshape(B * K, *vol_b_sh.shape[2:])
    bvec_a_flat = bvec_a_sh.reshape(B * K, 3)
    bvec_b_flat = bvec_b_sh.reshape(B * K, 3)
    bvec_t_flat = bvec_t_sh.reshape(B * K, 3)
    t_frac_flat = t_frac_sh.reshape(B * K)
    quality_flat = None
    if use_quality_cond:
        quality_flat = batch["quality_sh"].to(device).reshape(B * K, 2)

    pred_flat = model(vol_a_flat, vol_b_flat, bvec_a_flat, bvec_b_flat, bvec_t_flat,
                       t_frac_flat, quality=quality_flat)
    pred_sh = pred_flat.reshape(B, K, *pred_flat.shape[1:])
    return pred_sh, target_sh, bvec_t_sh, sh_mask


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int,
              use_quality_cond: bool = False, batch_log_f=None,
              angular_loss_weight: float = 0.0, sh_loss_high_order_min: int = 4,
              sh_loss_lmax_cap: int = 8) -> float:
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    n_samples = 0
    total_wait_s = 0.0
    total_compute_s = 0.0
    split = "train" if train else "val"
    prev_end = time.time()
    for batch in loader:
        t_received = time.time()
        wait_s = t_received - prev_end

        vol_a = batch["vol_a"].to(device)
        vol_b = batch["vol_b"].to(device)
        target = batch["target"].to(device)
        bvec_a = batch["bvec_a"].to(device)
        bvec_b = batch["bvec_b"].to(device)
        bvec_t = batch["bvec_t"].to(device)
        t_frac = batch["t_frac"].to(device)
        quality = batch["quality"].to(device) if use_quality_cond else None

        with torch.set_grad_enabled(train):
            pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, quality=quality)
            # MAE (mesma escolha do RCAE, ver run_epoch em 04_train_rcae.py) --
            # sem mascara aqui porque todo item ja tem shape fixo (1 par + 1
            # alvo, ver utils/rrin_dataset.py), nao ha padding de collate.
            loss_signal = (pred - target).abs().mean()
            # termo angular/SH opcional (ver protocolo secao 9/14.5/15 e
            # docstring do modulo acima) -- desativado por padrao
            # (angular_loss_weight=0.0), loss identica a antes.
            if angular_loss_weight > 0:
                pred_sh, target_sh, bvec_t_sh, sh_mask = _sh_bundle_forward(
                    model, batch, device, use_quality_cond)
                loss_angular = compute_sh_angular_loss(
                    pred_sh, target_sh, bvec_t_sh, sh_mask,
                    l_max_cap=sh_loss_lmax_cap, high_order_min=sh_loss_high_order_min)
                loss = loss_signal + angular_loss_weight * loss_angular
            else:
                loss_angular = None
                loss = loss_signal
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        t_compute_end = time.time()
        compute_s = t_compute_end - t_received

        total_loss += loss.item()
        n_batches += 1
        n_samples += vol_a.shape[0]
        total_wait_s += wait_s
        total_compute_s += compute_s

        if batch_log_f is not None:
            tags_str = ";".join(batch["subject_tag"])
            # loss_signal/loss_angular no fim (nao mudam a posicao das colunas
            # antigas) -- loss_angular fica vazia quando o termo esta
            # desativado, mesma convencao de 04_train_rcae.py.
            loss_angular_str = f"{loss_angular.item():.6f}" if loss_angular is not None else ""
            batch_log_f.write(f"{epoch},{split},{n_batches},{loss.item():.6f},"
                               f"{wait_s:.3f},{compute_s:.3f},{tags_str},"
                               f"{loss_signal.item():.6f},{loss_angular_str}\n")
            batch_log_f.flush()

        prev_end = time.time()

    if n_batches > 0:
        total_s = total_wait_s + total_compute_s
        throughput = n_samples / total_s if total_s > 0 else float("nan")
        pct_wait = 100 * total_wait_s / total_s if total_s > 0 else float("nan")
        print(f"[{split}] epoca {epoch} resumo: {n_batches} batches, {n_samples} patches | "
              f"wait total {total_wait_s:.1f}s ({pct_wait:.0f}%) | compute total "
              f"{total_compute_s:.1f}s | {throughput:.2f} patches/s", flush=True)

    return total_loss / max(1, n_batches)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="pasta com os <tag>_rrin_triplets.npz da etapa 2b")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--patch-size", type=int, default=10,
                     help="mesmo default do RCAE (10) -- ver utils/dataset.py")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-ch", type=int, default=16,
                     help="canais base da RRIN3D (ver model/rrin3d.py) -- rede "
                          "deliberadamente enxuta, ver docstring do modulo")
    ap.add_argument("--max-disp", type=float, default=0.5,
                     help="deslocamento maximo do campo de fluxo, em unidades "
                          "normalizadas (-1..1 cobre o patch inteiro por eixo)")
    ap.add_argument("--use-quality-cond", action="store_true",
                     help="condiciona a RRIN3D em residual_deg/gap_deg da trinca (ver "
                          "protocolo secao 10.1 e docstring de model.rrin3d.RRIN3D) -- em vez "
                          "de so filtrar trincas ruins fora do treino, deixa a rede aprender a "
                          "confiar menos no fluxo quando a geometria nao sustenta a suposicao "
                          "de interpolacao. Default (desativado) mantem o teste 'cego' mais "
                          "proximo de VFI de video de verdade -- ative pra rodar a variante "
                          "'consciente da qualidade' e comparar as duas.")
    ap.add_argument("--num-layers", type=int, default=1,
                     help="numero K de camadas de fluxo independentes (ver "
                          "model/rrin3d.py:RRIN3DLayered e protocolo secao 13, 'Toward a "
                          "layered-flow extension for crossing fibers'). Default 1 usa a "
                          "arquitetura ORIGINAL (model.rrin3d.RRIN3D -- um unico fluxo "
                          "bidirecional + 1 mapa de visibilidade escalar), mantendo "
                          "compatibilidade total com os checkpoints ja treinados. K>=2 usa "
                          "RRIN3DLayered: cada camada tem seu proprio par de fluxo (a->t, "
                          "b->t) e sua propria visibilidade, e as K camadas sao combinadas "
                          "por um softmax POR VOXEL (nao e um K que 'muda' por voxel -- a "
                          "arquitetura sempre calcula as K camadas em todo lugar; o que muda "
                          "por voxel e o PESO de cada camada nesse softmax, aprendido sem "
                          "nenhuma supervisao externa). Motivacao: um unico mapa de "
                          "visibilidade so decide 'confia mais em a ou em b' (bom pra oclusao "
                          "simples, tipo VFI de video), mas nao representa um voxel que e uma "
                          "MISTURA de duas populacoes de fibra (crossing) -- cada camada pode "
                          "se especializar numa populacao. Comece SEM nenhuma loss auxiliar "
                          "(so a loss de reconstrucao já existente) e compare K=1 (baseline) "
                          "vs K=2 vs K=3 via aggregate_valid/aggregate_invalid "
                          "(scripts/06_evaluate_reconstruction.py); so vale a pena acrescentar "
                          "supervisao tipo CSD/peak-count (sempre calculada da aquisicao "
                          "COMPLETA do sujeito, nunca do n_level subamostrado -- ver protocolo, "
                          "risco de indeterminacao/circularidade) se as camadas colapsarem "
                          "(pi quase identico em toda a imagem, sem estrutura espacial "
                          "reconhecivel quando comparado a regioes conhecidas de crossing).")
    ap.add_argument("--angular-loss-weight", type=float, default=0.0,
                     help="lambda do termo de loss opcional no dominio angular/SH (ver "
                          "protocolo secao 9/14.5/15, portado de scripts/04_train_rcae.py via "
                          "utils/sh_angular_loss.py). Default 0.0 = desativado, loss identica a "
                          "antes (so MAE de sinal). Com peso > 0, o dataset TAMBEM monta um "
                          "feixe de --sh-loss-q-out trincas do mesmo sujeito/patch por item "
                          "(ver utils/rrin_dataset.py, custo extra de compute proporcional a "
                          "--sh-loss-q-out) para poder ajustar uma base SH e penalizar erro nos "
                          "coeficientes de ordem alta (l>=--sh-loss-high-order-min) -- os "
                          "mesmos que capturam crossing e que a MAE de sinal bruto dilui entre "
                          "todos os voxels (maioria fibra unica). Grava em run_tag com sufixo "
                          "_sh (nao colide com as variantes sem loss angular).")
    ap.add_argument("--sh-loss-high-order-min", type=int, default=4,
                     help="ordem SH minima (par) considerada 'alta' pelo termo angular -- "
                          "so tem efeito se --angular-loss-weight > 0.")
    ap.add_argument("--sh-loss-lmax-cap", type=int, default=8,
                     help="teto de ordem SH usado no ajuste, mesmo com --sh-loss-q-out grande "
                          "o suficiente para sustentar mais -- so tem efeito se "
                          "--angular-loss-weight > 0.")
    ap.add_argument("--sh-loss-q-out", type=int, default=16,
                     help="tamanho do feixe de trincas extra por item usado SO para o termo de "
                          "loss angular/SH (ver utils/rrin_dataset.py, RRINTripletDataset.sh_q_out) "
                          "-- e o equivalente RRIN do --q-out do RCAE (numero de direcoes-alvo "
                          "simultaneas usadas pra ajustar a base SH). Default 16 sustenta ate "
                          "l_max=4 (precisa >=15 direcoes); l=6 precisa >=28, l=8 precisa >=45 "
                          "(ver n_coeffs_even). Tambem limitado pelo numero de trincas VALIDAS "
                          "que o sujeito realmente tem para este (shell,n_level) -- ver protocolo "
                          "secao 10.3 sobre a fracao valid cair em n_level baixo; itens sem "
                          "trincas suficientes pra sustentar --sh-loss-high-order-min sao "
                          "pulados no termo angular (a loss de sinal continua normal pra eles). "
                          "So tem efeito se --angular-loss-weight > 0.")
    ap.add_argument("--norm-type", choices=["instance", "batch"], default="instance",
                     help="tipo de normalizacao usada em todas as camadas conv de "
                          "model/rrin3d.py (ver docstring de _norm3d la). 'instance' "
                          "(default) e o comportamento ORIGINAL, compativel com todos os "
                          "checkpoints ja treinados -- mas calcula estatisticas por PATCH, o "
                          "que causa um artefato de 'costura' visivel entre patches na "
                          "reconstrucao com sliding-window (ver protocolo e "
                          "slurm/05b_reconstruct_rrin.sh, STRIDE/PATCH_SIZE -- confirmado "
                          "empiricamente que STRIDE menor/mais overlap atenua bastante o "
                          "artefato, mas isso e um paliativo). 'batch' troca por BatchNorm3d, "
                          "que em eval() usa estatisticas FIXAS (running_mean/running_var "
                          "acumuladas em todo o treino) em vez de recalcular por patch -- "
                          "resolve a causa raiz, nao so o sintoma. Custo: precisa treinar do "
                          "ZERO (nao da pra retomar um checkpoint 'instance' com "
                          "--norm-type batch, os parametros/estatisticas nao sao "
                          "compativeis) -- use --no-resume ou um --out-dir novo. Cria um run "
                          "SEPARADO (sufixo _bn no run_tag), nao sobrescreve os checkpoints "
                          "'instance' existentes.")
    ap.add_argument("--no-only-valid", action="store_true",
                     help="treina/valida tambem com trincas INVALIDAS (residuo alto ou alvo "
                          "extrapolado, ver utils/rrin_dataset.py:RRINTripletDataset e "
                          "protocolo secao 10.1/10.2). Default (so validas) e o motivo "
                          "confirmado de rrin/rrin_qc produzirem valores fisicamente "
                          "implausiveis (explosao numerica, NMSE ~1e9-1e11) nos alvos "
                          "invalidos na reconstrucao -- a rede nunca viu geometria parecida no "
                          "treino. Ativar esta flag NAO deve fazer a rede aprender fluxo onde "
                          "ele nao existe geometricamente (isso e uma limitacao de informacao, "
                          "nao de capacidade), mas deve evitar a explosao numerica trocando "
                          "'falha catastrofica' por 'degrada mal, de forma controlada' -- teste "
                          "e compare o campo aggregate_invalid de scripts/06_evaluate_reconstruction.py "
                          "antes/depois pra confirmar.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2)
    ap.add_argument("--val-num-workers", type=int, default=None)
    ap.add_argument("--val-max-cached-subjects", type=int, default=1)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-id", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--resume-checkpoint", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, "| job_id:", args.job_id or "(nao informado)")

    entries = load_manifest(args.manifest)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    only_valid = not args.no_only_valid
    print(f"[resumo] only_valid={only_valid} (--no-only-valid {'passado' if args.no_only_valid else 'nao passado'})")
    sh_q_out = args.sh_loss_q_out if args.angular_loss_weight > 0 else 0
    if args.angular_loss_weight > 0:
        max_l = min(max_order_for_n_directions(sh_q_out), args.sh_loss_lmax_cap)
        print(f"[angular-loss] ATIVO: lambda={args.angular_loss_weight}, "
              f"high_order_min={args.sh_loss_high_order_min}, lmax_cap={args.sh_loss_lmax_cap}, "
              f"sh_q_out={sh_q_out} -> ordem maxima alcancavel = l={max_l}", flush=True)
        if max_l < args.sh_loss_high_order_min:
            print(f"[angular-loss][aviso] --sh-loss-q-out {sh_q_out} so sustenta ate l={max_l} "
                  f"(< --sh-loss-high-order-min {args.sh_loss_high_order_min}) -- este termo vai "
                  f"ser pulado em praticamente todo item. Aumente --sh-loss-q-out para ter "
                  f"efeito real, ou reduza --sh-loss-high-order-min.", flush=True)
    else:
        print("[angular-loss] desativado (--angular-loss-weight 0.0, default) -- "
              "loss identica a antes (so MAE de sinal).", flush=True)
    train_ds = RRINTripletDataset(train_entries, args.triplets_dir, args.shell_b, args.n_level,
                                   patch_size=args.patch_size, training=True,
                                   mask_suffix=args.mask_suffix, only_valid=only_valid,
                                   min_tile_coverage=args.min_tile_coverage,
                                   seed=args.seed, max_cached_subjects=args.max_cached_subjects,
                                   sh_q_out=sh_q_out)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = RRINTripletDataset(val_entries, args.triplets_dir, args.shell_b, args.n_level,
                                 patch_size=args.patch_size, training=False,
                                 mask_suffix=args.mask_suffix, only_valid=only_valid,
                                 min_tile_coverage=args.min_tile_coverage,
                                 seed=args.seed + 1, max_cached_subjects=args.val_max_cached_subjects,
                                 sh_q_out=sh_q_out)

    persistent_train = args.num_workers > 0
    persistent_val = val_num_workers > 0
    train_sampler = SubjectGroupedSampler(train_ds, seed=args.seed)
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

    if args.num_layers < 1:
        raise ValueError(f"--num-layers deve ser >= 1 (recebido {args.num_layers})")
    model = build_rrin_model(num_layers=args.num_layers, base_ch=args.base_ch,
                              max_disp=args.max_disp,
                              use_quality_cond=args.use_quality_cond,
                              norm_type=args.norm_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model_name = type(model).__name__
    print(f"[resumo] {model_name}: {n_params} parametros (base_ch={args.base_ch}, "
          f"num_layers={args.num_layers}, use_quality_cond={args.use_quality_cond}, "
          f"norm_type={args.norm_type})")
    if args.norm_type == "batch" and args.batch_size < 4:
        print(f"[aviso] norm_type=batch com --batch-size={args.batch_size} e baixo -- "
              f"BatchNorm3d calcula estatisticas sobre o batch inteiro (mesmo agregando "
              f"tambem a extensao espacial de cada patch, um batch muito pequeno deixa "
              f"essas estatisticas ruidosas). Considere --batch-size>=4 se possivel.",
              flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=5)

    print("[sanity] testando 1 batch de treino + 1 de validacao antes do loop de epocas...",
          flush=True)

    def _sanity_step(loader, split_name, do_backward):
        t0 = time.time()
        batch = next(iter(loader))
        vol_a = batch["vol_a"].to(device)
        vol_b = batch["vol_b"].to(device)
        target = batch["target"].to(device)
        bvec_a = batch["bvec_a"].to(device)
        bvec_b = batch["bvec_b"].to(device)
        bvec_t = batch["bvec_t"].to(device)
        t_frac = batch["t_frac"].to(device)
        quality = batch["quality"].to(device) if args.use_quality_cond else None
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, quality=quality)
            loss = (pred - target).abs().mean()
            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        print(f"[sanity] {split_name} OK ({time.time() - t0:.1f}s, loss={loss.item():.6f}, "
              f"sujeitos={sorted(set(batch['subject_tag']))})", flush=True)
        if args.angular_loss_weight > 0:
            t1 = time.time()
            with torch.set_grad_enabled(do_backward):
                pred_sh, target_sh, bvec_t_sh, sh_mask = _sh_bundle_forward(
                    model, batch, device, args.use_quality_cond)
                loss_ang = compute_sh_angular_loss(
                    pred_sh, target_sh, bvec_t_sh, sh_mask,
                    l_max_cap=args.sh_loss_lmax_cap, high_order_min=args.sh_loss_high_order_min)
            print(f"[sanity] {split_name} (feixe SH) OK ({time.time() - t1:.1f}s, "
                  f"loss_angular={loss_ang.item():.6f}, K={sh_mask.shape[1]}, "
                  f"n_validos_medio={sh_mask.float().sum(dim=1).mean().item():.1f})", flush=True)

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- comecando o loop de epocas de verdade", flush=True)

    # ATENCAO: run_tag precisa refletir use_quality_cond E only_valid -- sem
    # isso, treinar uma variante com o MESMO --out-dir de outra sobrescreveria
    # o mesmo best.pt/last.pt (colisao silenciosa, sem nenhum aviso -- as duas
    # rodadas competindo pelo mesmo checkpoint em vez de ficarem separadas
    # para comparacao).
    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}"
    if args.use_quality_cond:
        run_tag += "_qc"
    if not only_valid:
        run_tag += "_inclinv"
    if args.num_layers > 1:
        run_tag += f"_k{args.num_layers}"
    if args.norm_type == "batch":
        run_tag += "_bn"
    if args.angular_loss_weight > 0:
        run_tag += "_sh"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt -- caminho fixo, "
          f"usado pela etapa 5b)")
    print(f"[resumo] logs deste run em: {run_dir}")

    # resume automatico -- mesmo mecanismo/semantica de 04_train_rcae.py
    # (ver comentarios la e protocolo secao 9, prioridade 3): por padrao
    # retoma de out_dir/last.pt se existir, salvo --no-resume ou
    # --resume-checkpoint explicito.
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
        for key in ("shell_b", "n_level", "patch_size", "base_ch", "max_disp", "use_quality_cond",
                    "num_layers", "norm_type", "angular_loss_weight", "sh_loss_high_order_min",
                    "sh_loss_lmax_cap", "sh_loss_q_out"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional.",
                      flush=True)
        # norm_type nao pode ser retomado entre variantes -- BatchNorm3d e
        # InstanceNorm3d tem parametros/buffers diferentes (running_mean/
        # running_var so existem no primeiro), load_state_dict falharia (ou
        # pior, "sucederia" parcialmente com strict=False se algum dia isso
        # mudar) de forma confusa. Falhar alto e cedo aqui.
        old_norm_type = old_args.get("norm_type", "instance")  # checkpoints antigos nao tinham este campo
        if old_norm_type != args.norm_type:
            raise ValueError(
                f"--norm-type={args.norm_type} nao bate com o checkpoint ({old_norm_type}) -- "
                f"norm_type nao e retomavel entre variantes (parametros/buffers incompativeis). "
                f"Use --no-resume ou um --out-dir/--norm-type novos para treinar a variante "
                f"'{args.norm_type}' do zero.")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(f"[resume] retomando da epoca {start_epoch} (best_val={best_val:.6f}, "
              f"epochs_no_improve={epochs_no_improve})", flush=True)
        if start_epoch > args.epochs:
            print(f"[resume] epoca de retomada ({start_epoch}) ja passa de --epochs "
                  f"({args.epochs}) -- nada a fazer.", flush=True)
    else:
        print("[resume] nenhum checkpoint anterior encontrado (ou --no-resume) -- "
              "comecando do zero.", flush=True)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr\n")
    batch_log_path = run_dir / "batch_log.csv"
    batch_log_f = open(batch_log_path, "w")
    batch_log_f.write("epoch,split,batch,loss,wait_s,compute_s,subject_tags,loss_signal,loss_angular\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                                    epoch=epoch, use_quality_cond=args.use_quality_cond,
                                    batch_log_f=batch_log_f,
                                    angular_loss_weight=args.angular_loss_weight,
                                    sh_loss_high_order_min=args.sh_loss_high_order_min,
                                    sh_loss_lmax_cap=args.sh_loss_lmax_cap)
            val_loss = run_epoch(model, val_loader, optimizer, device, train=False,
                                  epoch=epoch, use_quality_cond=args.use_quality_cond,
                                  batch_log_f=batch_log_f,
                                  angular_loss_weight=args.angular_loss_weight,
                                  sh_loss_high_order_min=args.sh_loss_high_order_min,
                                  sh_loss_lmax_cap=args.sh_loss_lmax_cap)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{current_lr:.2e}\n")
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