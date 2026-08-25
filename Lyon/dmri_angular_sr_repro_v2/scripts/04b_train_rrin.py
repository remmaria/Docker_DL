#!/usr/bin/env python3
"""
Etapa 4b (linha original da tese, retomada como diagnostico quantitativo --
ver protocolo, secao 10.1): treina a RRIN3D (model/rrin3d.py) para um
(shell, nivel de subamostragem) especifico, usando as trincas ja
construidas por scripts/02b_build_rrin_triplets.py.

Espelha bastante scripts/04_train_rcae.py (mesmo manifesto/split, mesma
normalizacao por percentil, mesmo layout de checkpoint out_dir/<shell>_<n>/
{best,last}.pt, mesmo resume automatico), mas SIMPLIFICADO de proposito
(sem os PNGs de debug por patch fixo, sem o termo de loss angular/SH --
ver docstring de model/rrin3d.py para o porque desta rede ser mantida
enxuta). Se depois de rodar a comparacao inicial isso fizer falta, da pra
portar essas partes de 04_train_rcae.py sem mudar o resto.

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
from model.rrin3d import RRIN3D


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int,
              use_quality_cond: bool = False, batch_log_f=None) -> float:
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
            loss = (pred - target).abs().mean()
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
            batch_log_f.write(f"{epoch},{split},{n_batches},{loss.item():.6f},"
                               f"{wait_s:.3f},{compute_s:.3f},{tags_str}\n")
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

    train_ds = RRINTripletDataset(train_entries, args.triplets_dir, args.shell_b, args.n_level,
                                   patch_size=args.patch_size, training=True,
                                   mask_suffix=args.mask_suffix,
                                   min_tile_coverage=args.min_tile_coverage,
                                   seed=args.seed, max_cached_subjects=args.max_cached_subjects)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = RRINTripletDataset(val_entries, args.triplets_dir, args.shell_b, args.n_level,
                                 patch_size=args.patch_size, training=False,
                                 mask_suffix=args.mask_suffix,
                                 min_tile_coverage=args.min_tile_coverage,
                                 seed=args.seed + 1, max_cached_subjects=args.val_max_cached_subjects)

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

    model = RRIN3D(base_ch=args.base_ch, max_disp=args.max_disp,
                    use_quality_cond=args.use_quality_cond).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[resumo] RRIN3D: {n_params} parametros (base_ch={args.base_ch}, "
          f"use_quality_cond={args.use_quality_cond})")
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

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- comecando o loop de epocas de verdade", flush=True)

    # ATENCAO: run_tag precisa refletir use_quality_cond -- sem isso, treinar
    # a variante "consciente da qualidade" (--use-quality-cond) com o MESMO
    # --out-dir da variante "cega" sobrescreveria o mesmo best.pt/last.pt
    # (colisao silenciosa, sem nenhum aviso -- as duas rodadas competindo
    # pelo mesmo checkpoint em vez de ficarem separadas para comparacao).
    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}"
    if args.use_quality_cond:
        run_tag += "_qc"
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
        for key in ("shell_b", "n_level", "patch_size", "base_ch", "max_disp", "use_quality_cond"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional.",
                      flush=True)
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
    batch_log_f.write("epoch,split,batch,loss,wait_s,compute_s,subject_tags\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                                    epoch=epoch, use_quality_cond=args.use_quality_cond,
                                    batch_log_f=batch_log_f)
            val_loss = run_epoch(model, val_loader, optimizer, device, train=False,
                                  epoch=epoch, use_quality_cond=args.use_quality_cond,
                                  batch_log_f=batch_log_f)
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