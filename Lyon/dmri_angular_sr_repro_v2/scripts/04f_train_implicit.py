#!/usr/bin/env python3
"""
Etapa 4f: treina o modelo de representacao angular IMPLICITA (NeRF/LIIF-
style, ver model/implicit_angular.py e addendum secao 20.11) para um
(shell, nivel de subamostragem) especifico. Repita a chamada para cada
combinacao que quiser cobrir.

Linha NOVA e INDEPENDENTE do RCAE (scripts/04_train_rcae.py) e da familia
RRIN/AMT/HFD/estrela (scripts/04b/04c/04d/04e) -- nao ha correspondencia
par-a-par nenhuma aqui: o dataset (utils/dataset.py:DWIPatchDataset, o MESMO
usado pelo RCAE -- e uma utilidade GENERICA de carregamento de patch, nao
codigo especifico do RCAE, ver docstring do modulo) entrega TODAS as
n_level direcoes medidas de uma vez (nao pares), e o modelo consulta
qualquer direcao-alvo continua (ver model/implicit_angular.py).

Uso:
    python scripts/04f_train_implicit.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --out-dir work_dir/implicit_checkpoints \
        --epochs 100 --batch-size 4 --patch-size 10 --lr 1e-3

Requer PyTorch + GPU. Nao executado neste ambiente de desenvolvimento (sem
torch instalado); revisar/ajustar hiperparametros no cluster.
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
from utils.dataset import (
    DWIPatchDataset, collate_variable_targets, SubjectGroupedSampler, worker_init_fn,
)
from utils.viz import save_patch_debug_png
from model.implicit_angular import build_implicit_model


def _tensor_stats(x: torch.Tensor, outlier_threshold: float = None) -> tuple:
    """Identico a scripts/04_train_rcae.py:_tensor_stats -- pequeno o
    bastante para duplicar aqui em vez de importar entre scripts numerados
    (convencao do projeto: scripts/0X_*.py nao se importam entre si, so
    utils/ e compartilhado livremente)."""
    if x.numel() == 0:
        stats = (float("nan"), float("nan"), float("nan"), float("nan"))
        return stats + (0,) if outlier_threshold is not None else stats
    stats = (x.mean().item(), x.std().item(), x.min().item(), x.max().item())
    if outlier_threshold is not None:
        n_outliers = int((x.abs() > outlier_threshold).sum().item())
        return stats + (n_outliers,)
    return stats


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int, batch_log_f=None,
              debug_state=None, outlier_threshold: float = 3.0, batch_log_every: int = 5):
    """MESMO espirito de scripts/04_train_rcae.py:run_epoch (mask MAE sobre
    target_mask, log por batch amostrado, snapshot de debug opcional) --
    adaptado para model.encode()/model.decode() (metodos, nao submodulos
    .encoder/.decoder como no RCAE, ver model/implicit_angular.py)."""
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
        subject_tags = batch["subject_tags"]
        input_vols = batch["input_vols"].to(device)
        input_bvecs = batch["input_bvecs"].to(device)
        target_vols = batch["target_vols"].to(device)
        target_bvecs = batch["target_bvecs"].to(device)
        target_mask = batch["target_mask"].to(device)
        mask = target_mask[:, :, None, None, None, None].expand_as(target_vols).float()

        with torch.set_grad_enabled(train):
            # encode/decode chamados separado (em vez de model(...) direto)
            # -- MESMO calculo, ImplicitAngularModel3D.forward faz
            # exatamente isso por dentro, sem custo extra -- so pra ficar
            # com "state" a mao pro snapshot de debug (mesmo padrao de
            # scripts/04_train_rcae.py).
            state = model.encode(input_vols, input_bvecs)
            pred = model.decode(state, target_bvecs)
            err = (pred - target_vols).abs()  # MAE, mesmo motivo do RCAE (ver 04_train_rcae.py)
            loss = (err * mask).sum() / mask.sum().clamp(min=1.0)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        t_compute_end = time.time()
        compute_s = t_compute_end - t_received

        total_loss += loss.item()
        n_batches += 1
        n_samples += input_vols.shape[0]
        total_wait_s += wait_s
        total_compute_s += compute_s

        if batch_log_f is not None and (n_batches == 1 or n_batches % max(1, batch_log_every) == 0):
            in_mean, in_std, in_min, in_max, in_n_out = _tensor_stats(
                input_vols, outlier_threshold=outlier_threshold)
            valid_target = target_vols[mask.bool()]
            tg_mean, tg_std, tg_min, tg_max, tg_n_out = _tensor_stats(
                valid_target, outlier_threshold=outlier_threshold)
            tags_str = ";".join(subject_tags)
            batch_log_f.write(
                f"{epoch},{split},{n_batches},{loss.item():.6f},"
                f"{in_mean:.4f},{in_std:.4f},{in_min:.4f},{in_max:.4f},{in_n_out},"
                f"{tg_mean:.4f},{tg_std:.4f},{tg_min:.4f},{tg_max:.4f},{tg_n_out},"
                f"{wait_s:.3f},{compute_s:.3f},{tags_str}\n"
            )
            batch_log_f.flush()

        if debug_state is not None and train:
            debug_state["step"] += 1
            step = debug_state["step"]
            every = debug_state["every"]
            if every > 0 and (step == 1 or step % every == 0):
                png_path = debug_state["dir"] / f"step_{step:06d}_epoch{epoch:04d}_batch{n_batches:04d}.png"
                subj0 = subject_tags[0] if subject_tags else "?"
                save_patch_debug_png(
                    png_path, input_vols[0], target_vols[0], pred_vols=pred[0].detach(),
                    context=state[0].detach(), max_dirs=debug_state["max_dirs"],
                    title=f"step {step} | epoca {epoch} batch {n_batches} | {subj0} | loss {loss.item():.6f}",
                )
                print(f"[debug] snapshot (batch) salvo em {png_path}", flush=True)

        prev_end = time.time()

    if n_batches > 0:
        total_s = total_wait_s + total_compute_s
        throughput = n_samples / total_s if total_s > 0 else float("nan")
        pct_wait = 100 * total_wait_s / total_s if total_s > 0 else float("nan")
        print(f"[{split}] epoca {epoch} resumo: {n_batches} batches, {n_samples} patches | "
              f"wait total {total_wait_s:.1f}s ({pct_wait:.0f}%) | compute total "
              f"{total_compute_s:.1f}s | {throughput:.2f} patches/s", flush=True)

    return total_loss / max(1, n_batches)


def plot_fixed_debug_patch(model, fixed_batch, device, plot_dir, epoch, val_loss=None,
                            shell_b=None, n_level=None, max_dirs=6):
    """MESMO papel de scripts/04_train_rcae.py:plot_fixed_debug_patch --
    roda o patch fixo de validacao (eval, no_grad) e salva o snapshot."""
    model.eval()
    with torch.no_grad():
        state = model.encode(fixed_batch["input_vols"].to(device),
                              fixed_batch["input_bvecs"].to(device))
        target_bvecs = fixed_batch["target_bvecs"].to(device)
        pred = model.decode(state, target_bvecs)
    loss_str = f" | val_loss {val_loss:.6f}" if val_loss is not None else " | baseline (sem treino)"
    png_path = plot_dir / f"epoch_{epoch:04d}.png"
    save_patch_debug_png(
        png_path, fixed_batch["input_vols"][0], fixed_batch["target_vols"][0],
        pred_vols=pred[0], context=state[0], max_dirs=max_dirs,
        title=f"shell={shell_b} n={n_level} | epoca {epoch}{loss_str}",
    )
    print(f"[debug] snapshot da predicao (patch fixo) salvo em {png_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--patch-size", type=int, default=10,
                     help="tamanho do patch cubico (default 10, mesmo default do RCAE/RRIN).")
    ap.add_argument("--q-out", type=int, default=10,
                     help="numero fixo de direcoes-alvo por exemplo em VALIDACAO (default 10 -- "
                          "ver utils/dataset.py:DWIPatchDataset. Em TREINO o split e "
                          "re-amostrado a cada exemplo, ver --training/_dynamic_split; este "
                          "flag so limita o N_out MAXIMO por exemplo). Ao contrario do RCAE "
                          "(paper fixa N_out=10 por ser o valor usado no artigo), aqui N_out "
                          "e livre (o decoder implicito aceita qualquer N_out, ver "
                          "model/implicit_angular.py) -- 10 e so um default razoavel de custo.")
    ap.add_argument("--l-max", type=int, default=None,
                     help="ordem par maxima da base SH usada para codificar direcoes de "
                          "entrada e alvo (ver model/implicit_angular.py:sh_positional_encoding). "
                          "Default None = automatico, utils.sh_basis.max_order_for_n_directions"
                          "(n_level) -- amarra a resolucao angular da representacao a quantas "
                          "direcoes sao realmente medidas, mesma convencao do baseline_sh. FIXO "
                          "para o checkpoint inteiro (muda o numero de canais da rede -- nao da "
                          "pra mudar em --resume-checkpoint, ver checagem abaixo).")
    ap.add_argument("--base-ch", type=int, default=16,
                     help="largura (numero de canais) dos blocos conv de PerDirectionEncoder3D/"
                          "SpatialTrunk3D/ImplicitDecoderHead3D (default 16, mesmo default de "
                          "FlowNet3D em model/rrin3d.py).")
    ap.add_argument("--norm-type", choices=["instance", "batch"], default="instance",
                     help="'instance' (default, ver model/rrin3d.py:_norm3d para a discussao "
                          "completa do artefato de costura entre patches na reconstrucao por "
                          "sliding-window) ou 'batch' (resolve a costura, exige treinar do "
                          "ZERO). Mesma semantica de --norm-type em scripts/04b_train_rrin.py.")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2)
    ap.add_argument("--val-num-workers", type=int, default=None)
    ap.add_argument("--val-max-cached-subjects", type=int, default=1)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-id", default="")
    ap.add_argument("--debug-plot-every", type=int, default=0)
    ap.add_argument("--debug-plot-every-batches", type=int, default=0)
    ap.add_argument("--debug-max-dirs", type=int, default=0)
    ap.add_argument("--outlier-threshold", type=float, default=3.0)
    ap.add_argument("--batch-log-every", type=int, default=5)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--resume-checkpoint", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, "| job_id:", args.job_id or "(nao informado)")
    debug_max_dirs = args.debug_max_dirs if args.debug_max_dirs > 0 else max(args.n_level, args.q_out)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    entries = load_manifest(args.manifest)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    train_ds = DWIPatchDataset(train_entries, args.scheme_dir, args.shell_b, args.n_level,
                                patch_size=args.patch_size, q_out=args.q_out, training=True,
                                mask_suffix=args.mask_suffix,
                                min_tile_coverage=args.min_tile_coverage,
                                seed=args.seed, max_cached_subjects=args.max_cached_subjects)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = DWIPatchDataset(val_entries, args.scheme_dir, args.shell_b, args.n_level,
                              patch_size=args.patch_size, q_out=args.q_out, training=False,
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
                               collate_fn=collate_variable_targets,
                               persistent_workers=persistent_train, worker_init_fn=winit)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=val_num_workers,
                             collate_fn=collate_variable_targets,
                             persistent_workers=persistent_val, worker_init_fn=winit_val)

    print(f"[resumo] treino: {len(train_ds.usable)} sujeitos utilizaveis "
          f"({len(train_ds)} patches, {len(train_loader)} batches/epoca)")
    print(f"[resumo] val:    {len(val_ds.usable)} sujeitos utilizaveis "
          f"({len(val_ds)} patches, {len(val_loader)} batches/epoca)")

    debug_fixed_batch = None
    if args.debug_plot_every > 0:
        best_idx = int(np.argmax(val_ds.tile_coverage))
        best_si, best_origin = val_ds.tile_index[best_idx]
        best_entry, best_tag = val_ds.usable[best_si]
        print(f"[resumo] patch fixo de debug: sujeito={best_tag} origem={best_origin} "
              f"cobertura_mascara={val_ds.tile_coverage[best_idx]:.3f}", flush=True)
        debug_fixed_batch = collate_variable_targets([val_ds[best_idx]])

    model = build_implicit_model(n_level=args.n_level, l_max=args.l_max, base_ch=args.base_ch,
                                  norm_type=args.norm_type).to(device)
    print(f"[modelo] n_level={args.n_level} l_max={model.l_max} (sh_dim={model.sh_dim}) "
          f"base_ch={args.base_ch} norm_type={args.norm_type} -- "
          f"{sum(p.numel() for p in model.parameters())} parametros")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=5)

    print("[sanity] testando 1 batch de treino + 1 de validacao antes do loop de epocas...",
          flush=True)

    def _sanity_step(loader, split_name, do_backward):
        t0 = time.time()
        batch = next(iter(loader))
        input_vols = batch["input_vols"].to(device)
        input_bvecs = batch["input_bvecs"].to(device)
        target_vols = batch["target_vols"].to(device)
        target_bvecs = batch["target_bvecs"].to(device)
        target_mask = batch["target_mask"].to(device)
        mask = target_mask[:, :, None, None, None, None].expand_as(target_vols).float()
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            pred = model(input_vols, input_bvecs, target_bvecs)
            err = (pred - target_vols).abs()
            loss = (err * mask).sum() / mask.sum().clamp(min=1.0)
            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        print(f"[sanity] {split_name} OK ({time.time() - t0:.1f}s, loss={loss.item():.6f}, "
              f"sujeitos={sorted(set(batch['subject_tags']))})", flush=True)

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- comecando o loop de epocas de verdade", flush=True)

    # run_tag: sufixos analogos aos ja usados em scripts/04b_train_rrin.py/
    # scripts/04_train_rcae.py (_qc/_inclinv/_bn/_sh) -- aqui so l_max/
    # base_ch/norm_type mudam o SHAPE dos pesos (bloqueante para resume, ver
    # checagem abaixo), entao cada combinacao ganha seu proprio out_dir.
    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}"
    if args.l_max is not None:
        run_tag += f"_lmax{args.l_max}"
    if args.base_ch != 16:
        run_tag += f"_ch{args.base_ch}"
    if args.norm_type == "batch":
        run_tag += "_bn"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_plot_dir = run_dir / "debug_patches"
    debug_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt -- caminho fixo, "
          f"usado pela etapa 5i)")
    print(f"[resumo] logs/debug deste run em: {run_dir}")

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
            raise FileNotFoundError(
                f"--resume-checkpoint {resume_ckpt_path} nao existe (confira o caminho, ou "
                f"use --no-resume pra comecar do zero sem retomar de checkpoint nenhum)")
        print(f"[resume] carregando checkpoint existente: {resume_ckpt_path}", flush=True)
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        old_args = ckpt.get("args", {})
        # l_max/base_ch/norm_type SAO bloqueantes (mudam shape dos pesos,
        # nao so a loss) -- mesma logica de bloqueio de decoder_type em
        # scripts/04_train_rcae.py / norm_type em scripts/04b_train_rrin.py.
        old_l_max = old_args.get("l_max")
        effective_old_l_max = old_l_max  # None so se o checkpoint tambem usou automatico
        if effective_old_l_max is None:
            print("[resume][aviso] checkpoint antigo nao registrou l_max explicito (ou usou "
                  "automatico) -- confie no proprio load_state_dict para pegar qualquer "
                  "incompatibilidade de shape.", flush=True)
        elif effective_old_l_max != args.l_max:
            raise ValueError(
                f"--l-max mudou entre o checkpoint ({old_l_max}) e esta chamada ({args.l_max}) "
                f"-- muda o numero de canais de PerDirectionEncoder3D/ImplicitDecoderHead3D "
                f"(sh_dim_for_lmax), incompativel para resume. Use --no-resume para treinar "
                f"a variante nova do zero.")
        old_base_ch = old_args.get("base_ch", 16)
        if old_base_ch != args.base_ch:
            raise ValueError(
                f"--base-ch mudou entre o checkpoint ({old_base_ch}) e esta chamada "
                f"({args.base_ch}) -- muda o shape de TODOS os pesos. Use --no-resume.")
        old_norm_type = old_args.get("norm_type", "instance")
        if old_norm_type != args.norm_type:
            raise ValueError(
                f"--norm-type mudou entre o checkpoint ({old_norm_type}) e esta chamada "
                f"({args.norm_type}) -- InstanceNorm3d e BatchNorm3d tem parametros "
                f"incompativeis (mesmo motivo de model/rrin3d.py). Use --no-resume.")
        for key in ("shell_b", "n_level", "patch_size", "q_out", "lr"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional.",
                      flush=True)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("[resume][aviso] checkpoint antigo sem optimizer_state -- otimizador "
                  "reinicia do zero.", flush=True)
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(f"[resume] retomando da epoca {start_epoch} (best_val={best_val:.6f}, "
              f"epochs_no_improve={epochs_no_improve}) -- treino ia ate a epoca "
              f"{args.epochs}", flush=True)
        if start_epoch > args.epochs:
            print(f"[resume] epoca de retomada ({start_epoch}) ja passa de --epochs "
                  f"({args.epochs}) -- nada a fazer, treino ja estava concluido.", flush=True)
    else:
        print("[resume] nenhum checkpoint anterior encontrado (ou --no-resume passado) -- "
              "comecando do zero.", flush=True)

    if debug_fixed_batch is not None:
        snapshot_epoch = 0 if start_epoch == 1 else start_epoch - 1
        plot_fixed_debug_patch(model, debug_fixed_batch, device, debug_plot_dir,
                                epoch=snapshot_epoch, val_loss=(best_val if start_epoch > 1 else None),
                                shell_b=args.shell_b, n_level=args.n_level,
                                max_dirs=debug_max_dirs)

    debug_state = None
    if args.debug_plot_every_batches > 0:
        debug_state = {"dir": debug_plot_dir, "every": args.debug_plot_every_batches, "step": 0,
                        "max_dirs": debug_max_dirs}

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr\n")

    batch_log_path = run_dir / "batch_log.csv"
    batch_log_f = open(batch_log_path, "w")
    batch_log_f.write(
        "epoch,split,batch,loss,"
        "input_mean,input_std,input_min,input_max,input_n_outliers,"
        "target_mean,target_std,target_min,target_max,target_n_outliers,"
        "wait_s,compute_s,subject_tags\n"
    )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                                    epoch=epoch, batch_log_f=batch_log_f,
                                    debug_state=debug_state, outlier_threshold=args.outlier_threshold,
                                    batch_log_every=args.batch_log_every)
            val_loss = run_epoch(model, val_loader, optimizer, device, train=False,
                                  epoch=epoch, batch_log_f=batch_log_f,
                                  outlier_threshold=args.outlier_threshold,
                                  batch_log_every=args.batch_log_every)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{current_lr:.2e}\n")
            print(f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

            if debug_fixed_batch is not None and (epoch % args.debug_plot_every == 0):
                plot_fixed_debug_patch(model, debug_fixed_batch, device, debug_plot_dir,
                                        epoch=epoch, val_loss=val_loss,
                                        shell_b=args.shell_b, n_level=args.n_level,
                                        max_dirs=debug_max_dirs)

            ckpt_common = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val": best_val,
                "epochs_no_improve": epochs_no_improve,
            }
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                ckpt_common["best_val"] = best_val
                ckpt_common["epochs_no_improve"] = epochs_no_improve
                torch.save(ckpt_common, out_dir / "best.pt")
                shutil.copy2(out_dir / "best.pt", run_dir / "best.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping na epoca {epoch} (sem melhora ha {args.patience} epocas)")
                    break

            ckpt_common["epochs_no_improve"] = epochs_no_improve
            torch.save(ckpt_common, out_dir / "last.pt")
            shutil.copy2(out_dir / "last.pt", run_dir / "last.pt")
    finally:
        batch_log_f.close()

    print("Treino concluido. Melhor val_loss:", best_val, "-> checkpoint em", out_dir / "best.pt")
    print(f"Copia permanente deste run em: {run_dir / 'best.pt'} (job_id={run_id})")
    print("Log por batch salvo em:", batch_log_path)


if __name__ == "__main__":
    main()