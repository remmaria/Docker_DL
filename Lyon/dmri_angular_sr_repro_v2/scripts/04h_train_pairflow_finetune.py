#!/usr/bin/env python3
"""
Etapa 4h (linha nova `pairflow_ssl`, Etapa 2/2 -- ver model/pairflow_ssl.py
e addendum secao 20.15): fine-tuning SUPERVISIONADO da `PairFlowInterp3D`
(fluxo bidirecional + extrapolacao linear pra t + blend/refino) nas MESMAS
trincas de sempre (scripts/02b_build_rrin_triplets.py via
utils/rrin_dataset.py:RRINTripletDataset) -- e' aqui que a extrapolacao por
t e' corrigida contra um alvo REAL pela primeira vez (a Etapa 1,
scripts/04g_train_pairflow_ssl.py, nunca ve nenhum ponto do meio do arco).

`--init-checkpoint` (opcional, mas e' o PONTO CENTRAL da ideia em duas
etapas): carrega os pesos do `flow_net` de um checkpoint da Etapa 1 pra
inicializar `PairFlowInterp3D.flow_net` (o `refine_net`, que so existe na
Etapa 2, comeca do zero sempre). Sem `--init-checkpoint`, este script ainda
funciona (treina `PairFlowInterp3D` do zero, com as MESMAS trincas
curadas que o RRIN3D usa) -- serve de CONTROLE pra isolar quanto do
resultado final vem do pre-treino auto-supervisionado versus so da
arquitetura em si (fluxo bidirecional + extrapolacao linear + refino, sem
NUNCA ter visto o pool grande sem curadoria).

`--freeze-flow` (ver model.pairflow_ssl.PairFlowInterp3D): congela
`flow_net` durante o fine-tuning (so `refine_net` e' treinado) -- isola
quanto do ganho vem so do blend/refino aprendendo a compensar um fluxo
pre-treinado FIXO, versus deixar o proprio fluxo se re-ajustar tambem aos
alvos reais (`--freeze-flow` nao passado = fine-tuning de verdade, fluxo
tambem se move).

Espelha bastante scripts/04b_train_rrin.py (mesmo manifesto/split via
RRINTripletDataset, mesmo resume automatico, mesmo layout de checkpoint) --
SIMPLIFICADO (sem angular-loss/sh_q_out/quality_cond/num_layers/
ensemble_m, que nao se aplicam aqui: `PairFlowInterp3D` nao tem esses
graus de liberdade).

Uso (com pre-treino da etapa 1):
    python scripts/04h_train_pairflow_finetune.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --init-checkpoint work_dir/pairflow_ssl_checkpoints/shell1000/best.pt \
        --out-dir work_dir/pairflow_checkpoints \
        --epochs 100 --batch-size 8 --patch-size 10 --lr 1e-4

Uso (controle, sem pre-treino -- treina do zero nas trincas curadas):
    python scripts/04h_train_pairflow_finetune.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --out-dir work_dir/pairflow_checkpoints \
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
from model.pairflow_ssl import build_pairflow_interp_model


def _gpu_mem_str(device) -> str:
    """Identico a scripts/04g_train_pairflow_ssl.py:_gpu_mem_str -- duplicado
    aqui (sem import cruzado entre scripts numerados, mesma convencao do
    projeto)."""
    if device.type != "cuda":
        return ""
    alloc_mb = torch.cuda.memory_allocated(device) / 1e6
    reserv_mb = torch.cuda.memory_reserved(device) / 1e6
    return f" | gpu_mem alloc={alloc_mb:.0f}MB reserved={reserv_mb:.0f}MB"


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int,
              batch_log_f=None, batch_log_every: int = 1, print_every: int = 0) -> float:
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

        with torch.set_grad_enabled(train):
            pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac)
            loss = (pred - target).abs().mean()  # mesma MAE de sempre (RCAE/RRIN/implicito)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        # ver comentario identico em scripts/04g_train_pairflow_ssl.py:run_epoch
        # -- sincroniza ANTES de medir compute_s, senao o tempo de GPU
        # aparece escondido dentro do wait_s do PROXIMO batch.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_compute_end = time.time()
        compute_s = t_compute_end - t_received

        total_loss += loss.item()
        n_batches += 1
        n_samples += vol_a.shape[0]
        total_wait_s += wait_s
        total_compute_s += compute_s

        # --batch-log-every (default 10, ver docstring do CLI) -- mesma
        # motivacao/mecanismo de scripts/04g_train_pairflow_ssl.py: batch_log.csv
        # ficava grande demais em treinos longos.
        if batch_log_f is not None and ((n_batches - 1) % batch_log_every == 0):
            tags_str = ";".join(batch["subject_tag"])
            batch_log_f.write(f"{epoch},{split},{n_batches},{loss.item():.6f},"
                               f"{wait_s:.3f},{compute_s:.3f},{tags_str}\n")
            batch_log_f.flush()

        # --print-every (default 20) -- mesmo diagnostico GPU-vs-CPU em tempo
        # real de scripts/04g_train_pairflow_ssl.py.
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

    return total_loss / max(1, n_batches)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="pasta com os <tag>_rrin_triplets.npz da etapa 2b -- MESMO esquema "
                          "de trincas usado por scripts/04b_train_rrin.py/04e_train_rrin_star.py.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--max-disp", type=float, default=0.5)
    ap.add_argument("--norm-type", choices=["instance", "batch"], default="instance")
    ap.add_argument("--no-only-valid", action="store_true",
                     help="mesma semantica de --no-only-valid em scripts/04b_train_rrin.py.")
    ap.add_argument("--init-checkpoint", default=None,
                     help="checkpoint da Etapa 1 (scripts/04g_train_pairflow_ssl.py) -- carrega "
                          "SO os pesos de `flow_net` pra inicializar (ver docstring do modulo). "
                          "Default None = treina PairFlowInterp3D do zero (controle -- ver "
                          "docstring do modulo).")
    ap.add_argument("--freeze-flow", action="store_true",
                     help="congela flow_net durante o fine-tuning (so refine_net e' treinado) -- "
                          "ver docstring do modulo e model.pairflow_ssl.PairFlowInterp3D. "
                          "Requer --init-checkpoint (nao faz sentido congelar um fluxo do zero).")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2)
    ap.add_argument("--val-num-workers", type=int, default=None)
    ap.add_argument("--val-max-cached-subjects", type=int, default=1)
    ap.add_argument("--batch-log-every", type=int, default=10,
                     help="grava so 1 a cada N linhas no batch_log.csv (default 10 -- ver "
                          "mesmo flag em scripts/04g_train_pairflow_ssl.py). --batch-log-every "
                          "1 volta ao comportamento original.")
    ap.add_argument("--print-every", type=int, default=20,
                     help="diagnostico GPU-vs-CPU em tempo real no stdout a cada N batches -- "
                          "ver mesmo flag em scripts/04g_train_pairflow_ssl.py. 0 = desligado.")
    ap.add_argument("--freeze-subject-order", action="store_true",
                     help="mesma flag/racional de scripts/04g_train_pairflow_ssl.py -- default "
                          "DESLIGADO (ordem dos sujeitos reembaralhada a cada epoca, "
                          "comportamento antigo). Passe pra ativar o diagnostico experimental "
                          "de congelar a ordem, revertido a pedido explicito da usuaria em "
                          "2026-09-02.")
    ap.add_argument("--log-worker-loads", action="store_true",
                     help="mesma flag/racional de scripts/04g_train_pairflow_ssl.py -- imprime "
                          "worker_id/subject_tag a cada carga real de disco em "
                          "RRINTripletDataset._load_subject. Gera muitas linhas -- so pra "
                          "diagnostico pontual.")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-id", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--resume-checkpoint", default=None)
    args = ap.parse_args()

    if args.freeze_flow and not args.init_checkpoint:
        sys.exit("--freeze-flow requer --init-checkpoint (nao faz sentido congelar fluxo do "
                  "zero, nao-treinado)")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, "| job_id:", args.job_id or "(nao informado)")

    entries = load_manifest(args.manifest)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    only_valid = not args.no_only_valid
    print(f"[resumo] only_valid={only_valid}")
    train_ds = RRINTripletDataset(train_entries, args.triplets_dir, args.shell_b, args.n_level,
                                   patch_size=args.patch_size, training=True,
                                   mask_suffix=args.mask_suffix, only_valid=only_valid,
                                   min_tile_coverage=args.min_tile_coverage,
                                   seed=args.seed, max_cached_subjects=args.max_cached_subjects,
                                   log_worker_loads=args.log_worker_loads)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = RRINTripletDataset(val_entries, args.triplets_dir, args.shell_b, args.n_level,
                                 patch_size=args.patch_size, training=False,
                                 mask_suffix=args.mask_suffix, only_valid=only_valid,
                                 min_tile_coverage=args.min_tile_coverage,
                                 seed=args.seed + 1, max_cached_subjects=args.val_max_cached_subjects,
                                 log_worker_loads=args.log_worker_loads)

    persistent_train = args.num_workers > 0
    persistent_val = val_num_workers > 0
    train_sampler = SubjectGroupedSampler(train_ds, seed=args.seed,
                                           freeze_order=args.freeze_subject_order)
    if args.freeze_subject_order:
        print("[dataloader] ordem dos sujeitos CONGELADA entre epocas (--freeze-subject-order "
              "explicito)", flush=True)
    winit = worker_init_fn if args.num_workers > 0 else None
    winit_val = worker_init_fn if val_num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=args.num_workers, drop_last=True,
                               persistent_workers=persistent_train, worker_init_fn=winit)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=val_num_workers,
                             persistent_workers=persistent_val, worker_init_fn=winit_val)

    print(f"[resumo] treino: {len(train_ds.usable)} sujeitos ({len(train_ds)} patches, "
          f"{len(train_loader)} batches/epoca)")
    print(f"[resumo] val:    {len(val_ds.usable)} sujeitos ({len(val_ds)} patches, "
          f"{len(val_loader)} batches/epoca)", flush=True)

    model = build_pairflow_interp_model(base_ch=args.base_ch, max_disp=args.max_disp,
                                         norm_type=args.norm_type,
                                         freeze_flow=args.freeze_flow).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[resumo] PairFlowInterp3D: {n_params} parametros ({n_trainable} treinaveis, "
          f"freeze_flow={args.freeze_flow}), base_ch={args.base_ch}, norm_type={args.norm_type}")

    if args.init_checkpoint:
        print(f"[init] carregando flow_net do checkpoint da Etapa 1: {args.init_checkpoint}",
              flush=True)
        ssl_ckpt = torch.load(args.init_checkpoint, map_location=device)
        ssl_args = ssl_ckpt.get("args", {})
        if ssl_args.get("base_ch") is not None and ssl_args["base_ch"] != args.base_ch:
            print(f"[init][aviso] --base-ch ({args.base_ch}) difere do checkpoint da Etapa 1 "
                  f"({ssl_args['base_ch']}) -- load_state_dict provavelmente vai falhar por "
                  f"shape incompativel.", flush=True)
        model.flow_net.load_state_dict(ssl_ckpt["model_state"])
        print(f"[init] flow_net inicializado (checkpoint da Etapa 1: epoca "
              f"{ssl_ckpt.get('epoch')}, val_loss {ssl_ckpt.get('val_loss')})", flush=True)
    else:
        print("[init] --init-checkpoint nao passado -- treinando PairFlowInterp3D do ZERO "
              "(controle, ver docstring do modulo)", flush=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr)
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
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac)
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

    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}"
    if not only_valid:
        run_tag += "_inclinv"
    if args.norm_type == "batch":
        run_tag += "_bn"
    if args.init_checkpoint:
        run_tag += "_pretrained"
    if args.freeze_flow:
        run_tag += "_frozen"
    if abs(args.lr - 1e-4) > 1e-12:
        run_tag += f"_lr{args.lr:g}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt -- caminho fixo, usado pela "
          f"etapa 5j)")
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
        for key in ("shell_b", "n_level", "patch_size", "base_ch", "max_disp", "norm_type",
                    "freeze_flow", "lr"):
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
        # ATENCAO: retomar via last.pt SOBRESCREVE o que --init-checkpoint teria carregado
        # (o load_state_dict do resume vem DEPOIS, e' o comportamento correto -- o run ja
        # em andamento ja incorporou o pre-treino na primeira epoca, nao precisa recarregar).
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
              "comecando do zero (ou do --init-checkpoint, se passado).", flush=True)

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
                                    epoch=epoch, batch_log_f=batch_log_f,
                                    batch_log_every=args.batch_log_every,
                                    print_every=args.print_every)
            val_loss = run_epoch(model, val_loader, optimizer, device, train=False,
                                  epoch=epoch, batch_log_f=batch_log_f,
                                  batch_log_every=args.batch_log_every,
                                  print_every=args.print_every)
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

    print("Fine-tuning concluido. Melhor val_loss:", best_val, "-> checkpoint em",
          out_dir / "best.pt")
    print(f"Copia permanente deste run em: {run_dir / 'best.pt'} (job_id={run_id})")


if __name__ == "__main__":
    main()