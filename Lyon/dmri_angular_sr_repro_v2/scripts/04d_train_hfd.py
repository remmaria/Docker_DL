#!/usr/bin/env python3
"""
Etapa 4d (linha HFD3D, ver model/hfd3d.py e protocolo/addendum 2026-08-27
secao 8): treina a HFD3D para um (shell, nivel de subamostragem)
especifico, usando as MESMAS trincas ja construidas por
scripts/02b_build_rrin_triplets.py e o MESMO dataset
(utils/rrin_dataset.py:RRINTripletDataset, sem nenhuma modificacao) que
scripts/04b_train_rrin.py e scripts/04c_train_amt.py ja usam.

DIFERENCA ESTRUTURAL em relacao a 04b/04c (leia antes de rodar): a HFD3D
NAO treina do zero contra o sinal-alvo direto -- ela precisa de uma AMT3D
JA TREINADA como "professora" de fluxo pseudo-verdadeiro (ver docstring de
model/hfd3d.py, secao "TREINO REQUER UM PROFESSOR..."). Por isso este
script exige `--teacher-checkpoint` (um best.pt/last.pt de
scripts/04c_train_amt.py para o MESMO shell/n_level -- nao precisa ser a
mesma variante de --use-quality-cond/--norm-type do aluno, sao
independentes). A professora fica CONGELADA (torch.no_grad(), sem
gradiente, sem otimizador) durante todo o treino do aluno.

CUSTO COMPUTACIONAL MAIOR que RRIN3D/AMT3D (avise a usuaria antes de rodar
em produção): a loss fotometrica (`(pred-target).abs().mean()`) passa pelo
`model.forward(...)`, que por sua vez roda `sample_flow` -- o loop DDIM
INTEIRO de `--num-sample-steps` passos (default 6), cada um chamando o
denoiser -- e o gradiente e retropropagado atraves de toda essa cadeia
(mesmo espirito de BPTT por uma RNN desenrolada `num-sample-steps` vezes).
Cada batch de treino faz, portanto, ~(num_sample_steps + 1) forwards do
denoiser (os `num_sample_steps` do `forward`/loss fotometrica, mais 1 do
`diffusion_loss`), contra 1 unico forward do modelo inteiro no RRIN3D/AMT3D.
Espere um tempo por epoca proporcionalmente maior -- ajuste --time no
slurm/04d_train_hfd.sh se o throughput real for muito menor que o do AMT3D.

Loss combinada (treino de UM ESTAGIO SO, ver model/hfd3d.py "SIMPLIFICACOES
DELIBERADAS"): loss = loss_signal (fotometrica, MAE, MESMA formula de
RRIN3D/AMT3D) + --diffusion-loss-weight * loss_diffusion (MSE de ruido,
ver model.hfd3d.HFD3D.diffusion_loss). Loss angular/SH (--angular-loss-weight)
NAO foi portada para a HFD3D nesta primeira versao (TODO real, nao apenas
nota desatualizada como aconteceu com a AMT3D -- ver model/hfd3d.py, que
nao expoe nenhum gancho de loss angular) -- deprioritizado porque o foco
desta rede e ser um TERCEIRO diagnostico da premissa de fluxo, nao
maximizar desempenho absoluto; pode ser adicionado depois se o resultado
justificar mais investimento aqui.

Uso:
    python scripts/04d_train_hfd.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --teacher-checkpoint work_dir/amt_checkpoints/shell1000_n10/best.pt \
        --shell-b 1000 --n-level 10 \
        --out-dir work_dir/hfd_checkpoints \
        --epochs 100 --batch-size 8 --patch-size 10 --lr 1e-3

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
from model.amt3d import build_amt_model
from model.hfd3d import build_hfd_model


def load_frozen_teacher(teacher_checkpoint: str, device):
    """Carrega uma AMT3D ja treinada (scripts/04c_train_amt.py) a partir do
    seu proprio checkpoint, em modo eval, com gradiente desligado em todos
    os parametros -- serve so como fonte de fluxo pseudo-verdadeiro (ver
    docstring do modulo). Le a arquitetura (base_ch, num_fields,
    corr_radius, use_quality_cond, norm_type) de dentro do checkpoint
    (`ckpt["args"]`), igual scripts/05d_reconstruct_amt.py ja faz -- nao
    precisa (nem deveria) ser passada de novo na linha de comando deste
    script, pra nao correr o risco de instanciar a professora com uma
    arquitetura diferente da que foi de fato treinada."""
    ckpt = torch.load(teacher_checkpoint, map_location=device)
    ckpt_args = ckpt["args"]
    teacher_use_quality_cond = ckpt_args.get("use_quality_cond", False)
    teacher = build_amt_model(base_ch=ckpt_args.get("base_ch", 16),
                               max_disp=ckpt_args.get("max_disp", 0.5),
                               num_fields=ckpt_args.get("num_fields", 3),
                               corr_radius=ckpt_args.get("corr_radius", 3),
                               use_quality_cond=teacher_use_quality_cond,
                               norm_type=ckpt_args.get("norm_type", "instance")).to(device)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[professor] AMT3D carregada de {teacher_checkpoint} "
          f"(epoca {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss')}, "
          f"use_quality_cond={teacher_use_quality_cond}, "
          f"num_fields={ckpt_args.get('num_fields', 3)}, "
          f"corr_radius={ckpt_args.get('corr_radius', 3)})", flush=True)
    return teacher, teacher_use_quality_cond


def _teacher_flow(teacher, teacher_use_quality_cond, batch, device):
    """Roda a professora (AMT3D congelada) num batch de trincas e retorna
    (flow_a, flow_b) pseudo-verdadeiros, ja destacados do grafo (nao ha
    gradiente fluindo de volta pra professora de qualquer forma, dado
    requires_grad_(False) em load_frozen_teacher, mas .detach() aqui deixa
    a intencao explicita no ponto de uso)."""
    vol_a = batch["vol_a"].to(device)
    vol_b = batch["vol_b"].to(device)
    bvec_a = batch["bvec_a"].to(device)
    bvec_b = batch["bvec_b"].to(device)
    bvec_t = batch["bvec_t"].to(device)
    t_frac = batch["t_frac"].to(device)
    quality = batch["quality"].to(device) if teacher_use_quality_cond else None
    with torch.no_grad():
        _, flow_a, flow_b, _ = teacher(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac,
                                        quality=quality, return_flow=True)
    return flow_a.detach(), flow_b.detach()


def run_epoch(model, teacher, teacher_use_quality_cond, loader, optimizer, device, train: bool,
              epoch: int, use_quality_cond: bool = False, diffusion_loss_weight: float = 1.0,
              batch_log_f=None) -> float:
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

        target_flow_a, target_flow_b = _teacher_flow(teacher, teacher_use_quality_cond, batch, device)

        with torch.set_grad_enabled(train):
            pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, quality=quality)
            loss_signal = (pred - target).abs().mean()
            loss_diffusion = model.diffusion_loss(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac,
                                                   target_flow_a, target_flow_b, quality=quality)
            loss = loss_signal + diffusion_loss_weight * loss_diffusion
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
                               f"{wait_s:.3f},{compute_s:.3f},{tags_str},"
                               f"{loss_signal.item():.6f},{loss_diffusion.item():.6f}\n")
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
                     help="pasta com os <tag>_rrin_triplets.npz da etapa 2b (mesma pasta "
                          "usada por scripts/04b_train_rrin.py e scripts/04c_train_amt.py)")
    ap.add_argument("--teacher-checkpoint", required=True,
                     help="checkpoint de uma AMT3D ja treinada (scripts/04c_train_amt.py) para "
                          "o MESMO shell/n_level -- ver load_frozen_teacher() e a nota de "
                          "'TREINO REQUER UM PROFESSOR' em model/hfd3d.py. A arquitetura da "
                          "professora e lida de dentro do proprio checkpoint, nao precisa "
                          "(nem deve) ser repetida aqui.")
    ap.add_argument("--out-dir", required=True,
                     help="raiz dos checkpoints DESTE metodo -- use um diretorio SEPARADO dos "
                          "usados por RRIN/AMT (ex.: $WORK_DIR/hfd_checkpoints), ver "
                          "slurm/04d_train_hfd.sh")
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--max-disp", type=float, default=0.5)
    ap.add_argument("--corr-radius", type=int, default=3,
                     help="mesmo papel/convencao de --corr-radius em scripts/04c_train_amt.py "
                          "(ver model/hfd3d.py:HFD3D) -- ARQUITETURAL (muda o shape da primeira "
                          "camada do denoiser), BLOQUEANTE em resume. Grava sufixo _r<radius> "
                          "quando != 3.")
    ap.add_argument("--num-timesteps", type=int, default=1000,
                     help="passos do schedule de difusao usado no TREINO (ver "
                          "model.hfd3d.HFD3D). Tratado como BLOQUEANTE em resume por "
                          "consistencia do estado do otimizador/scheduler com o regime de "
                          "ruido, embora nao mude nenhum shape de peso. Grava sufixo "
                          "_t<num_timesteps> quando != 1000.")
    ap.add_argument("--num-sample-steps", type=int, default=6,
                     help="passos DDIM na amostragem/inferencia (ver model.hfd3d.HFD3D) -- NAO "
                          "muda nenhum shape/peso, so custo/qualidade da amostragem -- por isso "
                          "e so AVISO (nao bloqueante) em resume, ao contrario de "
                          "--num-timesteps/--corr-radius. Grava sufixo _dstep<K> quando != 6.")
    ap.add_argument("--diffusion-loss-weight", type=float, default=1.0,
                     help="peso da loss de difusao (MSE de ruido) somada a loss fotometrica "
                          "(MAE) -- ver run_epoch(). Grava sufixo _dw<valor> quando != 1.0.")
    ap.add_argument("--use-quality-cond", action="store_true",
                     help="condiciona a HFD3D (aluna) em residual_deg/gap_deg da trinca -- "
                          "INDEPENDENTE de se a professora (--teacher-checkpoint) foi treinada "
                          "com ou sem isso (ver load_frozen_teacher, le da propria professora).")
    ap.add_argument("--norm-type", choices=["instance", "batch"], default="instance")
    ap.add_argument("--no-only-valid", action="store_true")
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

    teacher, teacher_use_quality_cond = load_frozen_teacher(args.teacher_checkpoint, device)

    entries = load_manifest(args.manifest)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    only_valid = not args.no_only_valid
    print(f"[resumo] only_valid={only_valid} (--no-only-valid {'passado' if args.no_only_valid else 'nao passado'})")

    train_ds = RRINTripletDataset(train_entries, args.triplets_dir, args.shell_b, args.n_level,
                                   patch_size=args.patch_size, training=True,
                                   mask_suffix=args.mask_suffix, only_valid=only_valid,
                                   min_tile_coverage=args.min_tile_coverage,
                                   seed=args.seed, max_cached_subjects=args.max_cached_subjects)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = RRINTripletDataset(val_entries, args.triplets_dir, args.shell_b, args.n_level,
                                 patch_size=args.patch_size, training=False,
                                 mask_suffix=args.mask_suffix, only_valid=only_valid,
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

    model = build_hfd_model(base_ch=args.base_ch, max_disp=args.max_disp,
                             corr_radius=args.corr_radius,
                             use_quality_cond=args.use_quality_cond,
                             norm_type=args.norm_type,
                             num_timesteps=args.num_timesteps,
                             num_sample_steps=args.num_sample_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[resumo] HFD3D: {n_params} parametros (base_ch={args.base_ch}, "
          f"corr_radius={args.corr_radius}, num_timesteps={args.num_timesteps}, "
          f"num_sample_steps={args.num_sample_steps}, use_quality_cond={args.use_quality_cond}, "
          f"norm_type={args.norm_type})")
    print(f"[aviso] custo por batch ~= {args.num_sample_steps + 1}x um forward do denoiser "
          f"(loop DDIM completo na loss fotometrica + 1 passo na loss de difusao) -- ver "
          f"docstring deste script. Ajuste --time no slurm se necessario.", flush=True)
    if args.norm_type == "batch" and args.batch_size < 4:
        print(f"[aviso] norm_type=batch com --batch-size={args.batch_size} e baixo -- "
              f"BatchNorm3d calcula estatisticas sobre o batch inteiro.", flush=True)

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
        target_flow_a, target_flow_b = _teacher_flow(teacher, teacher_use_quality_cond, batch, device)
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, quality=quality)
            loss_signal = (pred - target).abs().mean()
            loss_diffusion = model.diffusion_loss(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac,
                                                   target_flow_a, target_flow_b, quality=quality)
            loss = loss_signal + args.diffusion_loss_weight * loss_diffusion
            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        print(f"[sanity] {split_name} OK ({time.time() - t0:.1f}s, loss={loss.item():.6f}, "
              f"loss_signal={loss_signal.item():.6f}, loss_diffusion={loss_diffusion.item():.6f}, "
              f"sujeitos={sorted(set(batch['subject_tag']))})", flush=True)

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- comecando o loop de epocas de verdade", flush=True)

    # run_tag: mesma disciplina de sufixos de scripts/04b_train_rrin.py /
    # scripts/04c_train_amt.py.
    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}"
    if args.use_quality_cond:
        run_tag += "_qc"
    if not only_valid:
        run_tag += "_inclinv"
    if args.corr_radius != 3:
        run_tag += f"_r{args.corr_radius}"
    if args.num_timesteps != 1000:
        run_tag += f"_t{args.num_timesteps}"
    if args.num_sample_steps != 6:
        run_tag += f"_dstep{args.num_sample_steps}"
    if abs(args.diffusion_loss_weight - 1.0) > 1e-12:
        run_tag += f"_dw{args.diffusion_loss_weight:g}"
    if args.norm_type == "batch":
        run_tag += "_bn"
    if abs(args.lr - 1e-3) > 1e-12:
        run_tag += f"_lr{args.lr:g}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt -- caminho fixo, "
          f"usado pela etapa 5e)")
    print(f"[resumo] logs deste run em: {run_dir}")
    print(f"[resumo] professora (congelada): {args.teacher_checkpoint}")

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
        # WARN-only: nao mudam shape de peso.
        for key in ("shell_b", "n_level", "patch_size", "base_ch", "max_disp", "use_quality_cond",
                    "num_sample_steps", "diffusion_loss_weight", "lr", "teacher_checkpoint"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional.",
                      flush=True)
        # BLOQUEANTE: corr_radius, num_timesteps e norm_type mudam shape de
        # peso ou o buffer de schedule (ver docstrings dos argumentos acima).
        old_corr_radius = old_args.get("corr_radius", 3)
        old_num_timesteps = old_args.get("num_timesteps", 1000)
        old_norm_type = old_args.get("norm_type", "instance")
        if old_corr_radius != args.corr_radius:
            raise ValueError(
                f"--corr-radius={args.corr_radius} nao bate com o checkpoint ({old_corr_radius}) -- "
                f"corr_radius muda o numero de canais de entrada do denoiser, nao e retomavel "
                f"entre variantes. Use --no-resume ou um --out-dir novo.")
        if old_num_timesteps != args.num_timesteps:
            raise ValueError(
                f"--num-timesteps={args.num_timesteps} nao bate com o checkpoint "
                f"({old_num_timesteps}) -- muda o buffer alpha_bars salvo no checkpoint, nao e "
                f"retomavel entre variantes. Use --no-resume ou um --out-dir novo.")
        if old_norm_type != args.norm_type:
            raise ValueError(
                f"--norm-type={args.norm_type} nao bate com o checkpoint ({old_norm_type}) -- "
                f"nao e retomavel entre variantes. Use --no-resume ou um --out-dir/--norm-type novos.")
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
    batch_log_f.write("epoch,split,batch,loss,wait_s,compute_s,subject_tags,loss_signal,loss_diffusion\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = run_epoch(model, teacher, teacher_use_quality_cond, train_loader,
                                    optimizer, device, train=True, epoch=epoch,
                                    use_quality_cond=args.use_quality_cond,
                                    diffusion_loss_weight=args.diffusion_loss_weight,
                                    batch_log_f=batch_log_f)
            val_loss = run_epoch(model, teacher, teacher_use_quality_cond, val_loader,
                                  optimizer, device, train=False, epoch=epoch,
                                  use_quality_cond=args.use_quality_cond,
                                  diffusion_loss_weight=args.diffusion_loss_weight,
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