#!/usr/bin/env python3
"""
Etapa 4g (linha nova `pairflow_ssl`, Etapa 1/2 -- ver model/pairflow_ssl.py
e addendum secao 20.15): PRE-TREINO AUTO-SUPERVISIONADO do fluxo
bidirecional `PairFlowNet3D` entre pares de direcoes REAIS quaisquer, sem
nenhuma trinca/alvo curado (ver utils/pairflow_ssl_dataset.py -- NAO precisa
que scripts/02b_build_rrin_triplets.py tenha rodado).

Loss (model/pairflow_ssl.py:pairflow_ssl_losses): reconstroi cada extremo
do par a partir do OUTRO via warp (`warp3d(vol_a, flow_ab) ~= vol_b` e
vice-versa) + termo de consistencia direta/inversa (peso
--consistency-weight) + termo de suavidade OPCIONAL (peso
--smooth-weight, default 0.0 -- DESLIGADO de proposito, ver docstring de
`pairflow_ssl_losses`: suavidade e' exatamente o vies que pode apagar
estrutura real de cruzamento, e aqui nao ha nenhum terceiro ponto real no
meio do arco pra corrigir isso durante o pre-treino).

Depois deste treino, o checkpoint (`best.pt`) e' usado como
`--init-checkpoint` de scripts/04h_train_pairflow_finetune.py (Etapa 2,
supervisionada, nas trincas de sempre) -- ver addendum secao 20.15 para a
motivacao completa da ideia em duas etapas.

Espelha a estrutura de scripts/04b_train_rrin.py (resume automatico, layout
de checkpoint out_dir/<run_tag>/{best,last}.pt, batch_log.csv) --
SIMPLIFICADO (sem angular-loss/sh_q_out/quality_cond/num_layers, que nao se
aplicam a esta etapa: aqui nao ha alvo nenhum pra ajustar uma base SH).

Uso:
    python scripts/04g_train_pairflow_ssl.py \
        --manifest work_dir/manifest.csv \
        --shell-b 1000 \
        --out-dir work_dir/pairflow_ssl_checkpoints \
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
from utils.pairflow_ssl_dataset import PairFlowSSLDataset
from utils.dataset import SubjectGroupedSampler, worker_init_fn
from model.pairflow_ssl import build_pairflow_ssl_model, bidirectional_flow, pairflow_ssl_losses


def _gpu_mem_str(device) -> str:
    """String curta com memoria de GPU alocada/reservada (MB), ou '' se
    `device` nao for CUDA -- usada nos prints de diagnostico GPU-vs-CPU
    (ver --print-every/--batch-log-every abaixo). `memory_allocated` e' o
    que os tensores estao de fato usando; `memory_reserved` inclui o cache
    do allocator do PyTorch (normalmente maior, nao significa vazamento)."""
    if device.type != "cuda":
        return ""
    alloc_mb = torch.cuda.memory_allocated(device) / 1e6
    reserv_mb = torch.cuda.memory_reserved(device) / 1e6
    return f" | gpu_mem alloc={alloc_mb:.0f}MB reserved={reserv_mb:.0f}MB"


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int,
              consistency_weight: float, smooth_weight: float, batch_log_f=None,
              batch_log_every: int = 1, print_every: int = 0,
              gap_hist_step_deg: float = 15.0) -> float:
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    n_samples = 0
    total_wait_s = 0.0
    total_compute_s = 0.0
    split = "train" if train else "val"
    prev_end = time.time()

    # Histograma de gap_deg dos pares SORTEADOS nesta epoca (pedido da
    # usuaria apos ver que batches/epoca = tiles espaciais, nao pares -- ver
    # addendum secao 20.15: a cobertura O(N^2) de pares nao aumenta o
    # tamanho da epoca, ela acontece "por baixo", sorteio a sorteio, dentro
    # de PairFlowSSLDataset.__getitem__ -- isso aqui e' so' pra tornar essa
    # cobertura VISIVEL no log, em vez de so' confiar na teoria do dataset.
    # Custo desprezivel: `gap_deg` ja vem calculado no batch, so' agregamos
    # um histograma (poucos bins) em vez de guardar cada valor.
    gap_hist_edges = None
    gap_hist_counts = None
    if gap_hist_step_deg > 0:
        gap_hist_edges = np.arange(0.0, 90.0 + gap_hist_step_deg, gap_hist_step_deg)
        gap_hist_counts = np.zeros(len(gap_hist_edges) - 1, dtype=np.int64)
    for batch in loader:
        t_received = time.time()
        wait_s = t_received - prev_end

        vol_a = batch["vol_a"].to(device)
        vol_b = batch["vol_b"].to(device)
        bvec_a = batch["bvec_a"].to(device)
        bvec_b = batch["bvec_b"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            flow_ab, flow_ba = bidirectional_flow(model, vol_a, vol_b, bvec_a, bvec_b)
            losses = pairflow_ssl_losses(vol_a, vol_b, flow_ab, flow_ba, mask=mask,
                                          consistency_weight=consistency_weight,
                                          smooth_weight=smooth_weight)
            loss = losses["total"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        # sincroniza ANTES de medir compute_s -- sem isso, chamadas CUDA sao
        # assincronas e o "tempo de compute" medido aqui seria so o tempo de
        # ENFILEIRAR os kernels (quase instantaneo), nao de eles rodarem de
        # verdade -- inflaria wait_s artificialmente (o tempo real da GPU
        # apareceria "escondido" dentro do wait_s do PROXIMO batch, ja que so
        # loss.item() abaixo forca uma sincronizacao implicita). So custa
        # tempo de verdade se a GPU realmente estiver ocupada -- e' exatamente
        # o que queremos medir aqui.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_compute_end = time.time()
        compute_s = t_compute_end - t_received

        total_loss += loss.item()
        n_batches += 1
        n_samples += vol_a.shape[0]
        total_wait_s += wait_s
        total_compute_s += compute_s

        if gap_hist_counts is not None:
            gap_np = batch["gap_deg"].numpy()
            counts, _ = np.histogram(gap_np, bins=gap_hist_edges)
            gap_hist_counts += counts

        # --batch-log-every>1 (default 1 = loga todo batch, comportamento
        # ORIGINAL): grava so 1 a cada N linhas no batch_log.csv -- pedido da
        # usuaria, o arquivo ficava grande demais num treino de muitas
        # epocas/batches por epoca. Sempre loga o PRIMEIRO batch de cada
        # epoca (n_batches==1), pra nunca perder o inicio de uma epoca no CSV.
        if batch_log_f is not None and ((n_batches - 1) % batch_log_every == 0):
            tags_str = ";".join(batch["subject_tag"])
            mean_gap = batch["gap_deg"].mean().item()
            batch_log_f.write(
                f"{epoch},{split},{n_batches},{loss.item():.6f},{wait_s:.3f},{compute_s:.3f},"
                f"{tags_str},{losses['recon_ab'].item():.6f},{losses['recon_ba'].item():.6f},"
                f"{losses['consistency'].item():.6f},{float(losses['smooth']):.6f},"
                f"{mean_gap:.2f}\n")
            batch_log_f.flush()

        # --print-every>0: diagnostico GPU-vs-CPU EM TEMPO REAL no stdout
        # (log do sbatch), sem precisar esperar o resumo de fim de epoca nem
        # abrir o batch_log.csv. wait_s alto e compute_s baixo = gargalo de
        # dataloading (CPU/disco) -- suba --num-workers/--max-cached-subjects,
        # nao --batch-size. compute_s alto e' a GPU (ou o host) de fato
        # trabalhando -- ai' sim --batch-size maior tende a ajudar (mais
        # trabalho por lancamento de kernel). gpu_mem ajuda a saber se da'
        # pra subir --batch-size sem estourar OOM.
        if print_every > 0 and n_batches % print_every == 0:
            print(f"[{split}] epoca {epoch} batch {n_batches}: wait={wait_s:.3f}s "
                  f"compute={compute_s:.3f}s{_gpu_mem_str(device)}", flush=True)

        prev_end = time.time()

    if n_batches > 0:
        total_s = total_wait_s + total_compute_s
        throughput = n_samples / total_s if total_s > 0 else float("nan")
        pct_wait = 100 * total_wait_s / total_s if total_s > 0 else float("nan")
        print(f"[{split}] epoca {epoch} resumo: {n_batches} batches, {n_samples} patches | "
              f"wait total {total_wait_s:.1f}s ({pct_wait:.0f}%) | compute total "
              f"{total_compute_s:.1f}s | {throughput:.2f} patches/s", flush=True)

        if gap_hist_counts is not None and gap_hist_counts.sum() > 0:
            total_gap = int(gap_hist_counts.sum())
            parts = []
            for i in range(len(gap_hist_counts)):
                lo, hi = gap_hist_edges[i], gap_hist_edges[i + 1]
                pct = 100.0 * gap_hist_counts[i] / total_gap
                parts.append(f"[{lo:g}-{hi:g})={int(gap_hist_counts[i])}({pct:.0f}%)")
            print(f"[{split}] epoca {epoch} gap_deg hist (graus, {total_gap} pares "
                  "sorteados): " + " ".join(parts), flush=True)

    return total_loss / max(1, n_batches)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--min-pair-gap-deg", type=float, default=5.0,
                     help="descarta pares angularmente MUITO proximos (solucao de fluxo~0 "
                          "trivial, ver docstring de utils/pairflow_ssl_dataset.py). Default 5.")
    ap.add_argument("--max-pair-gap-deg", type=float, default=None,
                     help="teto opcional de gap angular entre o par sorteado (default None = "
                          "sem teto, DE PROPOSITO -- o ponto central da ideia e' treinar tambem "
                          "com pares distantes, ver addendum secao 20.15).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--max-disp", type=float, default=0.5)
    ap.add_argument("--norm-type", choices=["instance", "batch"], default="instance")
    ap.add_argument("--consistency-weight", type=float, default=0.1,
                     help="peso do termo de consistencia direta/inversa (ver docstring de "
                          "model.pairflow_ssl.pairflow_ssl_losses). Default 0.1.")
    ap.add_argument("--smooth-weight", type=float, default=0.0,
                     help="peso do termo de suavidade (TV) do fluxo -- DESLIGADO por padrao de "
                          "proposito (ver docstring de pairflow_ssl_losses: risco de apagar "
                          "estrutura real de cruzamento). So ative com peso pequeno e "
                          "conscientemente.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2)
    ap.add_argument("--val-num-workers", type=int, default=None)
    ap.add_argument("--val-max-cached-subjects", type=int, default=1)
    ap.add_argument("--batch-log-every", type=int, default=10,
                     help="grava so 1 a cada N linhas no batch_log.csv (default 10 -- antes "
                          "cada batch virava uma linha, o arquivo crescia rapido demais em "
                          "treinos longos). --batch-log-every 1 volta ao comportamento "
                          "original (loga todo batch). O primeiro batch de cada epoca e' "
                          "SEMPRE logado, mesmo com N>1.")
    ap.add_argument("--print-every", type=int, default=20,
                     help="imprime no stdout, a cada N batches, o tempo de espera (wait_s -- "
                          "dataloader/CPU) vs. tempo de compute (compute_s -- GPU, ja com "
                          "torch.cuda.synchronize) do batch, mais memoria de GPU alocada -- "
                          "diagnostico rapido de gargalo SEM esperar o resumo de fim de epoca "
                          "nem abrir o batch_log.csv. wait_s alto = gargalo de dataloading "
                          "(--num-workers/--max-cached-subjects ajudam, --batch-size nao); "
                          "compute_s alto = GPU de fato ocupada (--batch-size maior tende a "
                          "ajudar). 0 = desligado.")
    ap.add_argument("--gap-hist-step-deg", type=float, default=15.0,
                     help="tamanho (em graus) de cada faixa do histograma de gap_deg dos pares "
                          "sorteados nesta epoca, impresso no resumo de fim de epoca -- torna "
                          "visivel no log que a cobertura O(N^2) de pares (proximos E distantes, "
                          "ver docstring de utils/pairflow_ssl_dataset.py) esta de fato "
                          "acontecendo, mesmo com batches/epoca fixado pelo numero de tiles "
                          "espaciais (nao pelo numero de pares). 0 = desligado.")
    ap.add_argument("--freeze-subject-order", action="store_true",
                     help="EXPERIMENTAL (2026-09-02), default DESLIGADO -- comportamento default "
                          "voltou a ser o antigo (ordem dos sujeitos reembaralhada a cada epoca, "
                          "SubjectGroupedSampler(freeze_order=False)). Passar esta flag ativa o "
                          "diagnostico de congelar a ordem (SubjectGroupedSampler(freeze_order=True), "
                          "ver docstring la) -- so pra testar/depurar gargalo de dataloading "
                          "(ver addendum secao 20.15/20.17-bis), revertido a pedido explicito da "
                          "usuaria em 2026-09-02 por nao ter resolvido o gargalo real e ter "
                          "coincidido com uma rodada travada.")
    ap.add_argument("--log-worker-loads", action="store_true",
                     help="diagnostico (2026-09-02): imprime worker_id/subject_tag a cada carga "
                          "REAL de disco (cache MISS) em PairFlowSSLDataset._load_subject -- "
                          "procure no log um MESMO subject com MULTIPLOS worker= diferentes na "
                          "mesma epoca, confirma que o DataLoader esta despachando batches desse "
                          "sujeito pra workers diferentes (round-robin por batch, nao por bloco "
                          "de sujeito -- ver docstring de utils.dataset.SubjectGroupedSampler), "
                          "forcando releitura redundante do mesmo volume em cada um. Gera MUITAS "
                          "linhas de log -- ligue so pra diagnostico pontual, nao deixe ligado "
                          "num treino longo de producao.")
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

    train_ds = PairFlowSSLDataset(train_entries, args.shell_b, patch_size=args.patch_size,
                                   training=True, mask_suffix=args.mask_suffix,
                                   shell_tol=args.shell_tol,
                                   min_tile_coverage=args.min_tile_coverage,
                                   min_pair_gap_deg=args.min_pair_gap_deg,
                                   max_pair_gap_deg=args.max_pair_gap_deg,
                                   seed=args.seed, max_cached_subjects=args.max_cached_subjects,
                                   log_worker_loads=args.log_worker_loads)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = PairFlowSSLDataset(val_entries, args.shell_b, patch_size=args.patch_size,
                                 training=False, mask_suffix=args.mask_suffix,
                                 shell_tol=args.shell_tol,
                                 min_tile_coverage=args.min_tile_coverage,
                                 min_pair_gap_deg=args.min_pair_gap_deg,
                                 max_pair_gap_deg=args.max_pair_gap_deg,
                                 seed=args.seed + 1, max_cached_subjects=args.val_max_cached_subjects,
                                 log_worker_loads=args.log_worker_loads)

    persistent_train = args.num_workers > 0
    persistent_val = val_num_workers > 0
    train_sampler = SubjectGroupedSampler(train_ds, seed=args.seed,
                                           freeze_order=args.freeze_subject_order)
    if args.freeze_subject_order:
        print("[dataloader] ordem dos sujeitos CONGELADA entre epocas (--freeze-subject-order "
              "explicito -- ver utils.dataset.SubjectGroupedSampler.__init__)",
              flush=True)
    winit = worker_init_fn if args.num_workers > 0 else None
    winit_val = worker_init_fn if val_num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=args.num_workers, drop_last=True,
                               persistent_workers=persistent_train, worker_init_fn=winit)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=val_num_workers,
                             persistent_workers=persistent_val, worker_init_fn=winit_val)

    print(f"[resumo] treino: {len(train_ds.usable)} sujeitos ({len(train_ds)} tiles, "
          f"{len(train_loader)} batches/epoca)")
    print(f"[resumo] val:    {len(val_ds.usable)} sujeitos ({len(val_ds)} tiles, "
          f"{len(val_loader)} batches/epoca)", flush=True)

    model = build_pairflow_ssl_model(base_ch=args.base_ch, max_disp=args.max_disp,
                                      norm_type=args.norm_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[resumo] PairFlowNet3D: {n_params} parametros (base_ch={args.base_ch}, "
          f"norm_type={args.norm_type}, consistency_weight={args.consistency_weight}, "
          f"smooth_weight={args.smooth_weight})")
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
        bvec_a = batch["bvec_a"].to(device)
        bvec_b = batch["bvec_b"].to(device)
        mask = batch["mask"].to(device)
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            flow_ab, flow_ba = bidirectional_flow(model, vol_a, vol_b, bvec_a, bvec_b)
            losses = pairflow_ssl_losses(vol_a, vol_b, flow_ab, flow_ba, mask=mask,
                                          consistency_weight=args.consistency_weight,
                                          smooth_weight=args.smooth_weight)
            loss = losses["total"]
            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        print(f"[sanity] {split_name} OK ({time.time() - t0:.1f}s, loss={loss.item():.6f}, "
              f"recon_ab={losses['recon_ab'].item():.6f}, recon_ba={losses['recon_ba'].item():.6f}, "
              f"consistency={losses['consistency'].item():.6f}, "
              f"sujeitos={sorted(set(batch['subject_tag']))})", flush=True)

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- comecando o loop de epocas de verdade", flush=True)

    run_tag = f"shell{int(args.shell_b)}"
    if args.norm_type == "batch":
        run_tag += "_bn"
    if abs(args.consistency_weight - 0.1) > 1e-12:
        run_tag += f"_cw{args.consistency_weight:g}"
    if args.smooth_weight > 0:
        run_tag += f"_sw{args.smooth_weight:g}"
    if args.min_pair_gap_deg != 5.0:
        run_tag += f"_mingap{args.min_pair_gap_deg:g}"
    if args.max_pair_gap_deg is not None:
        run_tag += f"_maxgap{args.max_pair_gap_deg:g}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt)")
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
        for key in ("shell_b", "patch_size", "base_ch", "max_disp", "norm_type", "lr"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional.",
                      flush=True)
        old_norm_type = old_args.get("norm_type", "instance")
        if old_norm_type != args.norm_type:
            raise ValueError(
                f"--norm-type={args.norm_type} nao bate com o checkpoint ({old_norm_type}) -- "
                f"use --no-resume ou um --out-dir novo para treinar do zero.")
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
    else:
        print("[resume] nenhum checkpoint anterior encontrado (ou --no-resume) -- "
              "comecando do zero.", flush=True)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr\n")
    batch_log_path = run_dir / "batch_log.csv"
    batch_log_f = open(batch_log_path, "w")
    batch_log_f.write("epoch,split,batch,loss,wait_s,compute_s,subject_tags,"
                       "recon_ab,recon_ba,consistency,smooth,mean_gap_deg\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                                    epoch=epoch, consistency_weight=args.consistency_weight,
                                    smooth_weight=args.smooth_weight, batch_log_f=batch_log_f,
                                    batch_log_every=args.batch_log_every,
                                    print_every=args.print_every,
                                    gap_hist_step_deg=args.gap_hist_step_deg)
            val_loss = run_epoch(model, val_loader, optimizer, device, train=False,
                                  epoch=epoch, consistency_weight=args.consistency_weight,
                                  smooth_weight=args.smooth_weight, batch_log_f=batch_log_f,
                                  batch_log_every=args.batch_log_every,
                                  print_every=args.print_every,
                                  gap_hist_step_deg=args.gap_hist_step_deg)
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

    print("Pre-treino auto-supervisionado concluido. Melhor val_loss:", best_val,
          "-> checkpoint em", out_dir / "best.pt")
    print(f"Copia permanente deste run em: {run_dir / 'best.pt'} (job_id={run_id})")
    print("Proximo passo: scripts/04h_train_pairflow_finetune.py --init-checkpoint "
          f"{out_dir / 'best.pt'} ...")


if __name__ == "__main__":
    main()