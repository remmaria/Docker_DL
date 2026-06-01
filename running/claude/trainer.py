"""
trainer.py
----------
Training loop para o SIREN encoder + decoder com masked q-space modeling.

Features:
  - Mixed precision (AMP) para economia de memória
  - Gradient clipping (importante para SIREN)
  - LR scheduler com warmup
  - Checkpointing automático
  - Logging via WandB (com fallback silencioso se chave não disponível)
  - Early stopping baseado em validation loss
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from siren import SIRENEncoder, SIRENDecoder
from losses import QSpaceLoss
from dataset import MaskedQSpaceDataset, collate_variable_dwi

# Força flush em todo print — essencial para logs aparecerem no SLURM
import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

# ---------------------------------------------------------------------------
# WandB — inicializa apenas se a chave estiver disponível
# ---------------------------------------------------------------------------

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def init_wandb(config: dict, output_dir: Path, project: str = "qshine") -> bool:
    """
    Inicializa o WandB seguindo o padrão do usuário:
      - Lê WANDB_API_KEY do ambiente
      - Se não tiver chave, roda em modo offline (salva localmente)
      - Retorna True se inicializado com sucesso
    """
    if not HAS_WANDB:
        print("wandb não instalado. pip install wandb")
        return False

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        # Modo offline: logs ficam em wandb/offline-* e podem ser sincronizados depois
        os.environ["WANDB_MODE"] = "offline"
        print("WANDB_API_KEY não encontrada — rodando em modo offline.")
        print("Para sincronizar depois: wandb sync wandb/offline-*")

    wandb.init(
        project=project,
        name=output_dir.name,           # nome do run = nome da pasta (ex: experiment_01)
        config=config,
        dir=str(output_dir),
        resume="allow",                 # permite retomar run após checkpoint
    )
    return True


# ---------------------------------------------------------------------------
# Modelo completo (Encoder + Decoder juntos)
# ---------------------------------------------------------------------------

class QSpaceModel(nn.Module):
    """Wrapper que une Encoder e Decoder. Facilita save/load e inferência."""

    def __init__(
        self,
        in_features: int   = 5,
        query_dim: int     = 4,
        hidden_dim: int    = 256,
        latent_dim: int    = 128,
        n_enc_layers: int  = 5,
        n_dec_layers: int  = 4,
        omega_0: float     = 30.0,
    ):
        super().__init__()
        self.encoder = SIRENEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_enc_layers,
            omega_0=omega_0,
        )
        self.decoder = SIRENDecoder(
            query_dim=query_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_layers=n_dec_layers,
            omega_0=omega_0,
        )

    def forward(
        self,
        x_context: torch.Tensor,
        q_query: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if ctx_mask is not None:
            x_context = x_context.masked_fill(ctx_mask.unsqueeze(-1), 0.0)
        z = self.encoder(x_context)
        S_pred = self.decoder(z, q_query)
        return S_pred, z

    def encode(self, x_context, ctx_mask=None):
        if ctx_mask is not None:
            x_context = x_context.masked_fill(ctx_mask.unsqueeze(-1), 0.0)
        return self.encoder(x_context)

    def decode(self, z, q_query):
        return self.decoder(z, q_query)


# ---------------------------------------------------------------------------
# LR Scheduler com warmup linear + cosine annealing
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self._step        = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            factor = self._step / max(1, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = self.min_lr + 0.5 * (1.0 - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Training loop principal
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: QSpaceModel,
        train_dataset: MaskedQSpaceDataset,
        val_dataset: Optional[MaskedQSpaceDataset],
        config: dict,
        output_dir: str = "runs/experiment_01",
        wandb_project: str = "qshine",
    ):
        self.model        = model
        self.train_dataset = train_dataset
        self.val_dataset  = val_dataset
        self.config       = config
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Salva config em JSON
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {self.device}")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
        )

        # DataLoaders
        # Usa SequentialSampler porque _build_index já embaralha por sujeito.
        # Isso evita cache misses aleatórios: batches consecutivos vêm do
        # mesmo sujeito → 1 leitura de disco serve muitos batches.
        from torch.utils.data import SequentialSampler
        num_workers = config.get("num_workers", 0)

        def worker_init_fn(worker_id):
            import torch.utils.data
            ds = torch.utils.data.get_worker_info().dataset
            if hasattr(ds, "_cache"):
                ds._cache.clear()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            sampler=SequentialSampler(train_dataset),
            num_workers=num_workers,
            collate_fn=collate_variable_dwi,
            pin_memory=self.device.type == "cuda" and num_workers > 0,
            drop_last=True,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.get("batch_size", 32) * 2,
                sampler=SequentialSampler(val_dataset),
                num_workers=num_workers,
                collate_fn=collate_variable_dwi,
                pin_memory=self.device.type == "cuda" and num_workers > 0,
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
            )
        else:
            self.val_loader = None

        print(f"DataLoader: num_workers={num_workers}, "
              f"batch_size={config.get('batch_size', 32)}, "
              f"cache_size={config.get('cache_size', 8)}")

        # Scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps     = config.get("epochs", 100) * steps_per_epoch
        # warmup_epochs adapta ao tamanho do dataset automaticamente
        warmup_epochs   = config.get("warmup_epochs", 2)
        warmup_steps    = warmup_epochs * steps_per_epoch
        self.scheduler  = WarmupCosineScheduler(self.optimizer, warmup_steps, total_steps)
        print(f"Scheduler: warmup={warmup_steps} steps ({warmup_epochs} épocas), "
              f"total={total_steps} steps")

        # Loss
        self.criterion = QSpaceLoss(
            lambda_recon  = config.get("lambda_recon", 1.0),
            lambda_mono   = config.get("lambda_mono", 0.1),
            lambda_smooth = config.get("lambda_smooth", 0.05),
        )

        # Mixed precision
        self.use_amp = config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler  = GradScaler("cuda", enabled=self.use_amp)

        # WandB
        self.use_wandb = init_wandb(config, self.output_dir, project=wandb_project)
        if self.use_wandb:
            wandb.watch(self.model, log="gradients", log_freq=200)

        # Estado
        self.global_step     = 0
        self.best_val_loss   = float("inf")
        self.patience_counter = 0
        self.patience        = config.get("patience", 20)

    # -----------------------------------------------------------------------
    # Logging helper — único ponto de contato com wandb
    # -----------------------------------------------------------------------

    def _log(self, metrics: dict, step: Optional[int] = None):
        """Loga um dict de métricas. step=None → usa epoch implícito do wandb."""
        if self.use_wandb:
            wandb.log(metrics, step=step)

    # -----------------------------------------------------------------------
    # Train epoch
    # -----------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_dataset.resample()

        accum = {"loss": 0., "recon": 0., "mono": 0., "smooth": 0.}
        n_batches = len(self.train_loader)
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            x_ctx    = batch["x_context"].to(self.device)
            ctx_mask = batch["ctx_mask"].to(self.device)
            q_query  = batch["q_query"].to(self.device)
            S_target = batch["S_target"].to(self.device)
            q_mask   = batch["q_mask"].to(self.device)
            b_vals   = batch["bvals_query"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                S_pred, z = self.model(x_ctx, q_query, ctx_mask)
                losses = self.criterion(S_pred, S_target, q_query, q_mask, b_vals)

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            for k in accum:
                accum[k] += losses.get(k, losses["total"]).item()

            self.global_step += 1

            # Log a cada 10% da época, no mínimo 1x
            log_every = max(1, n_batches // 10)
            if batch_idx % log_every == 0:
                lr      = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:03d} | Step {batch_idx:04d}/{n_batches} | "
                    f"Loss: {losses['total'].item():.4f} | "
                    f"Recon: {losses['recon'].item():.4f} | "
                    f"Mono: {losses['mono'].item():.4f} | "
                    f"LR: {lr:.2e} | {elapsed:.1f}s"
                )
                self._log({
                    "train/loss_step":  losses["total"].item(),
                    "train/recon_step": losses["recon"].item(),
                    "train/mono_step":  losses["mono"].item(),
                    "train/lr":         lr,
                }, step=self.global_step)

        for k in accum:
            accum[k] /= n_batches
        return accum

    # -----------------------------------------------------------------------
    # Validation epoch
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        if self.val_loader is None:
            return {}

        self.model.eval()
        accum = {"loss": 0., "recon": 0., "mono": 0., "smooth": 0.}
        shell_errors: Dict[str, list] = {}

        for batch in self.val_loader:
            x_ctx    = batch["x_context"].to(self.device)
            ctx_mask = batch["ctx_mask"].to(self.device)
            q_query  = batch["q_query"].to(self.device)
            S_target = batch["S_target"].to(self.device)
            q_mask   = batch["q_mask"].to(self.device)
            b_vals   = batch["bvals_query"].to(self.device)

            with autocast("cuda", enabled=self.use_amp):
                S_pred, z = self.model(x_ctx, q_query, ctx_mask)
                losses = self.criterion(S_pred, S_target, q_query, q_mask, b_vals)

            for k in accum:
                accum[k] += losses.get(k, losses["total"]).item()

            # MAE por shell
            valid     = ~q_mask
            S_pred_sq = S_pred.squeeze(-1)
            b_rounded = (b_vals / 100).round() * 100   # torch não aceita round(-2)
            for b_val in [0, 1000, 2000, 3000]:
                shell_mask = (b_rounded == b_val) & valid
                if shell_mask.sum() > 0:
                    err = (S_pred_sq[shell_mask] - S_target[shell_mask]).abs().mean().item()
                    shell_errors.setdefault(f"b{b_val}", []).append(err)

        n = len(self.val_loader)
        for k in accum:
            accum[k] /= n

        for shell, errs in shell_errors.items():
            accum[f"mae_{shell}"] = sum(errs) / len(errs)

        return accum

    # -----------------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        ckpt = {
            "epoch":                epoch,
            "global_step":          self.global_step,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss":             val_loss,
            "config":               self.config,
        }
        torch.save(ckpt, self.output_dir / "last_checkpoint.pt")
        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pt")
            print(f"  ★ Novo melhor modelo salvo (val_loss={val_loss:.4f})")
            if self.use_wandb:
                wandb.run.summary["best_val_loss"] = val_loss
                wandb.run.summary["best_epoch"]    = epoch

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["global_step"]
        print(f"Checkpoint carregado: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
        return ckpt["epoch"]

    # -----------------------------------------------------------------------
    # Loop principal
    # -----------------------------------------------------------------------

    def _warmup_cache(self, dataset, n: int = 3):
        """
        Pré-carrega os primeiros n sujeitos.
        Com preload=True isso é no-op (já estão em RAM).
        Com preload=False carrega do disco antes do Step 0.
        """
        import time
        if getattr(dataset, "preload", False):
            return  # já tudo em RAM, nada a fazer
        print(f"Pré-aquecendo cache ({n} sujeitos)...", end=" ")
        t0 = time.time()
        for s_idx in range(min(n, len(dataset.meta))):
            dataset._get_S_norm(s_idx)
        print(f"pronto em {time.time()-t0:.1f}s")

    def fit(self, resume_from: Optional[str] = None):
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1

        epochs = self.config.get("epochs", 100)
        print(f"\nIniciando treino: {epochs} épocas, device={self.device}")
        print(f"Train batches/epoch: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches/epoch: {len(self.val_loader)}")
        print("-" * 60)

        # Pré-aquece cache para evitar freeze no Step 0
        cache_size = self.config.get("cache_size", 5)
        self._warmup_cache(self.train_dataset, n=min(cache_size, 3))

        for epoch in range(start_epoch, epochs):
            t_start       = time.time()
            train_metrics = self.train_epoch(epoch)
            t_train       = time.time() - t_start
            val_metrics   = self.validate(epoch)

            val_loss = val_metrics.get("loss", train_metrics["loss"])

            # Log por época — tudo num único wandb.log para alinhar o eixo x
            epoch_log = {
                "epoch":             epoch,
                "train/loss":        train_metrics["loss"],
                "train/recon":       train_metrics["recon"],
                "train/mono":        train_metrics["mono"],
                "train/smooth":      train_metrics["smooth"],
                "train/epoch_time":  t_train,
            }
            if val_metrics:
                epoch_log["val/loss"]   = val_loss
                epoch_log["val/recon"]  = val_metrics.get("recon", 0)
                for k, v in val_metrics.items():
                    if k.startswith("mae_"):
                        epoch_log[f"val/{k}"] = v
            self._log(epoch_log, step=self.global_step)

            print(
                f"\nEpoch {epoch:03d}/{epochs-1} | "
                f"Train: {train_metrics['loss']:.4f} "
                f"(recon={train_metrics['recon']:.4f}, mono={train_metrics['mono']:.4f}) | "
                f"Val: {val_loss:.4f} | "
                f"Tempo: {t_train:.1f}s"
            )

            # Shell MAEs no print se disponíveis
            shell_str = "  ".join(
                f"{k}={v:.4f}" for k, v in val_metrics.items() if k.startswith("mae_")
            )
            if shell_str:
                print(f"  Shell MAEs → {shell_str}")

            # Checkpoint + early stopping
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss    = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, val_loss, is_best=is_best)

            # ---- Debug PNGs ----
            debug_every = self.config.get("debug_every", 5)
            if epoch % debug_every == 0 or is_best:
                try:
                    from debug_viz import save_debug_images
                    save_debug_images(
                        model=self.model,
                        val_dataset=self.val_dataset,
                        output_dir=str(self.output_dir),
                        epoch=epoch,
                        device=self.device,
                        n_voxels=self.config.get("debug_n_voxels", 6),
                    )
                except Exception as e:
                    print(f"  Debug viz falhou (não crítico): {e}")

            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping após {self.patience} épocas sem melhora.")
                break

        if self.use_wandb:
            wandb.finish()

        print(f"\nTreino finalizado. Melhor val_loss: {self.best_val_loss:.4f}")
        print(f"Modelo salvo em: {self.output_dir / 'best_model.pt'}")