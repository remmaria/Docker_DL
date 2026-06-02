"""
train.py
--------
Script principal de treino.

Uso:
  # Com dados reais:
  python train.py --data_dir /path/to/subjects --output_dir runs/exp01

  # Teste rápido com dados sintéticos (sem nenhum dado real):
  python train.py --synthetic --output_dir runs/exp_synthetic

  # Retomar treino:
  python train.py --data_dir /path/to/subjects --resume runs/exp01/last_checkpoint.pt
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List

from siren import SIRENEncoder, SIRENDecoder
from trainer import QSpaceModel, Trainer
from dataset import MaskedQSpaceDataset
from metrics import evaluate_model, plot_qspace_prediction


# ---------------------------------------------------------------------------
# Gerador de dados sintéticos (para testar sem dados reais)
# ---------------------------------------------------------------------------

def generate_synthetic_dwi(
    n_voxels: int = 1000,
    protocol: str = "single_shell",
    noise_level: float = 0.02,
    save_dir: str = "synthetic_data/sub_synth",
):
    """
    Gera dados DWI sintéticos via modelo de tensor de difusão (DTI).

    S(b, g) = S0 · exp(-b · g^T D g)

    onde D é um tensor de difusão aleatório (fisicamente plausível).
    Isso garante que os dados satisfazem as propriedades físicas que
    a rede deve aprender.

    Protocolos disponíveis:
      - 'single_shell':  b=1000, 30 direções
      - 'multi_shell':   b=0/1000/2000, 6+30+30 direções
      - 'hcp_like':      b=0/1000/2000/3000, 6+90+90+90 direções
    """
    import os

    protocols = {
        "single_shell": {
            "shells": [(0, 6), (1000, 30)],
        },
        "multi_shell": {
            "shells": [(0, 6), (1000, 30), (2000, 30)],
        },
        "hcp_like": {
            "shells": [(0, 6), (1000, 90), (2000, 90), (3000, 90)],
        },
    }

    spec = protocols[protocol]
    rng = np.random.default_rng(42)

    # ---- Gera gradientes (distribuição uniforme na esfera) ----
    bvals_list = []
    bvecs_list = []
    for b_val, n_dirs in spec["shells"]:
        if b_val == 0:
            bvecs_list.append(np.zeros((n_dirs, 3)))
            bvals_list.append(np.full(n_dirs, b_val, dtype=np.float32))
        else:
            # Distribuição uniforme na esfera via método de rejeição
            vecs = rng.normal(size=(n_dirs, 3))
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            bvecs_list.append(vecs)
            bvals_list.append(np.full(n_dirs, b_val, dtype=np.float32))

    bvals = np.concatenate(bvals_list)
    bvecs = np.concatenate(bvecs_list, axis=0)
    N_dwi = len(bvals)

    # ---- Gera tensores de difusão aleatórios ----
    # Eigenvalues típicos de substância branca (em mm²/s × 10^-3)
    # λ1 ∈ [0.8, 1.8] × 10^-3, λ2≈λ3 ∈ [0.2, 0.6] × 10^-3
    lambda1 = rng.uniform(0.8e-3, 1.8e-3, n_voxels)
    lambda23 = rng.uniform(0.2e-3, 0.6e-3, n_voxels)
    fa_target = (lambda1 - lambda23) / (np.sqrt(2) *
        np.sqrt(lambda1**2 + 2*lambda23**2) / np.sqrt(3))

    # Orientações principais aleatórias
    main_dirs = rng.normal(size=(n_voxels, 3))
    main_dirs /= np.linalg.norm(main_dirs, axis=1, keepdims=True)

    # Constrói tensor D para cada voxel e prediz sinal
    S0 = rng.uniform(800, 1200, n_voxels)   # intensidade T2
    dwi = np.zeros((n_voxels, N_dwi), dtype=np.float32)

    for i in range(n_voxels):
        v = main_dirs[i]
        # Eigenvectors: v é o principal, ortogonais aleatórios
        perp = rng.normal(size=3)
        perp -= perp.dot(v) * v
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(v, perp)

        # Tensor D = Σ λ_k * v_k v_k^T
        D = (lambda1[i] * np.outer(v, v)
           + lambda23[i] * np.outer(perp, perp)
           + lambda23[i] * np.outer(perp2, perp2))

        for j, (b, g) in enumerate(zip(bvals, bvecs)):
            if b == 0:
                dwi[i, j] = S0[i]
            else:
                adc = g @ D @ g
                dwi[i, j] = S0[i] * np.exp(-b * adc)

    # Ruído Rician
    noise_std = float(noise_level) * float(S0.mean())
    noise = rng.normal(0, noise_std, dwi.shape)
    dwi = np.maximum(dwi + noise, 0)

    # ---- Salva em formato compatível com load_subject() ----
    os.makedirs(save_dir, exist_ok=True)

    # Reshape para 3D
    side = int(np.ceil(n_voxels ** (1/3)))
    pad_size = side**3 - n_voxels
    dwi_padded = np.pad(dwi, [(0, pad_size), (0, 0)])
    dwi_3d = dwi_padded.reshape(side, side, side, N_dwi).astype(np.float32)

    mask_flat   = np.ones(n_voxels, dtype=np.uint8)
    mask_padded = np.pad(mask_flat, (0, pad_size))
    mask_3d     = mask_padded.reshape(side, side, side).astype(bool)

    # Usa os mesmos nomes dos arquivos reais
    np.save(f"{save_dir}/bgpdwis_PA_geomcorr.npy",          dwi_3d)
    np.save(f"{save_dir}/bgpdwis_PA_geomcorr_mask3d.npy",   mask_3d)
    np.savetxt(f"{save_dir}/bgpdwis_PA_geomcorr.bval", bvals[None, :], fmt="%.1f")
    np.savetxt(f"{save_dir}/bgpdwis_PA_geomcorr.bvec", bvecs.T,        fmt="%.6f")

    print(f"Dados sintéticos gerados: {save_dir}")
    print(f"  Protocolo: {protocol}")
    print(f"  Voxels: {n_voxels}, N_dwi: {N_dwi}")
    print(f"  FA médio: {fa_target.mean():.3f} ± {fa_target.std():.3f}")

    return save_dir


# ---------------------------------------------------------------------------
# Config padrão
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Arquitetura
    "in_features":   5,
    "query_dim":     4,
    "hidden_dim":    256,
    "latent_dim":    128,
    "n_enc_layers":  5,
    "n_dec_layers":  4,
    "omega_0":       30.0,

    # Treino
    "epochs":           100,
    "batch_size":       32,
    "lr":               5e-5,
    "weight_decay":     1e-4,
    "warmup_epochs":    2,      # warmup em épocas, não steps (adapta ao dataset)
    "use_amp":          True,
    "num_workers":      0,
    "patience":         30,

    # Dataset
    "mask_ratio":           0.30,
    "masking_strategy":     "random",
    "voxels_per_subject":   500,
    "b_max":                5000.0,
    "val_fraction":         0.15,
    "cache_size":           5,
    "preload":              True,
    "ram_limit_gb":         0,      # 0 = auto-detecta do SLURM_MEM_PER_NODE
    "n_load_workers":       8,

    # Losses
    "lambda_recon":   1.0,
    "lambda_mono":    0.2,
    "lambda_smooth":  0.2,

    # Debug
    "debug_every":    5,    # salva PNGs a cada N épocas (e sempre que houver novo best)
    "debug_n_voxels": 6,    # número de voxels no painel de debug
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import builtins, functools
    # Força flush em todo print desde o início do main (SLURM bufferiza stdout)
    builtins.print = functools.partial(builtins.print, flush=True)

    parser = argparse.ArgumentParser(description="q-SHINE Stage 1: SIREN + Masked Q-Space")
    parser.add_argument("--data_dir",       type=str, default=None,
                        help="Diretório com subdiretórios de sujeitos")
    parser.add_argument("--output_dir",     type=str, default="runs/experiment_01")
    parser.add_argument("--config",         type=str, default=None,
                        help="JSON com config personalizada (sobrescreve padrões)")
    parser.add_argument("--resume",         type=str, default=None,
                        help="Path para checkpoint para retomar treino")
    parser.add_argument("--synthetic",      action="store_true",
                        help="Gera dados sintéticos para teste rápido")
    parser.add_argument("--n_synth_subjects", type=int, default=5)
    parser.add_argument("--eval_only",      action="store_true",
                        help="Apenas avalia um checkpoint (requer --resume)")
    parser.add_argument("--wandb_project",  type=str, default="qshine",
                        help="Nome do projeto no WandB")
    args = parser.parse_args()

    # Cria output_dir imediatamente — antes de qualquer outra coisa
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Redireciona stdout e stderr para arquivo de log (além do terminal)
    log_path = output_dir / "train.log"
    print(f"{'='*60}")
    print(f"  q-SHINE — Stage 1 iniciando")
    print(f"  output_dir : {output_dir.resolve()}")
    print(f"  log        : {log_path}")
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  torch      : {torch.__version__}")
    print(f"  CUDA       : {torch.version.cuda}")
    print(f"  GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")

    # Escreve também em arquivo para não depender de tail no SLURM
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # ---- Config ----
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))
    print(f"Config: batch={config['batch_size']}, lr={config['lr']}, epochs={config['epochs']}")

    # ---- Dados ----
    if args.synthetic:
        print("=== Modo sintético ===")
        subject_dirs = []
        protocols = ["single_shell", "multi_shell", "hcp_like",
                     "single_shell", "multi_shell"]
        for i in range(args.n_synth_subjects):
            prot = protocols[i % len(protocols)]
            d = generate_synthetic_dwi(
                n_voxels=500,
                protocol=prot,
                save_dir=f"synthetic_data/sub_{i:03d}",
            )
            subject_dirs.append(d)
    else:
        if args.data_dir is None:
            raise ValueError("Forneça --data_dir ou use --synthetic")
        data_path = Path(args.data_dir)
        subject_dirs = sorted([str(p) for p in data_path.iterdir() if p.is_dir()])
        print(f"Encontrados {len(subject_dirs)} sujeitos em {args.data_dir}")

    # ---- Split train/val ----
    n_val = max(1, int(len(subject_dirs) * config["val_fraction"]))
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(subject_dirs))
    val_dirs   = [subject_dirs[i] for i in perm[:n_val]]
    train_dirs = [subject_dirs[i] for i in perm[n_val:]]
    print(f"Train: {len(train_dirs)} sujeitos | Val: {len(val_dirs)} sujeitos")

    # ---- Datasets ----
    # Auto-detecta RAM disponível do SLURM e seta limite seguro
    if config.get("ram_limit_gb", 0) == 0:
        slurm_mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", 0))
        if slurm_mem_mb > 0:
            # Reserva 30GB para SO + torch + índice
            config["ram_limit_gb"] = max(10.0, slurm_mem_mb / 1024 - 30)
            print(f"SLURM mem={slurm_mem_mb/1024:.0f}GB → "
                  f"ram_limit_gb={config['ram_limit_gb']:.0f}GB "
                  f"(com float16 cabe ~{config['ram_limit_gb']*2:.0f}GB de dados brutos)")
        else:
            config["ram_limit_gb"] = 80.0   # fallback conservador
            print(f"SLURM_MEM_PER_NODE não encontrado → ram_limit_gb=80GB")

    print(f"\nCriando datasets...")
    train_dataset = MaskedQSpaceDataset(
        train_dirs,
        mask_ratio=config["mask_ratio"],
        masking_strategy=config["masking_strategy"],
        voxels_per_subject=config["voxels_per_subject"],
        b_max=config["b_max"],
        preload=config["preload"],
        ram_limit_gb=config["ram_limit_gb"],
        cache_size=config["cache_size"],
        n_load_workers=config["n_load_workers"],
        augment=True,
    )
    val_dataset = MaskedQSpaceDataset(
        val_dirs,
        mask_ratio=config["mask_ratio"],
        masking_strategy="random",
        voxels_per_subject=config["voxels_per_subject"] // 2,
        b_max=config["b_max"],
        preload=config["preload"],
        ram_limit_gb=max(10.0, config["ram_limit_gb"] * 0.15),  # 15% do limite para val
        cache_size=max(5, config["cache_size"] // 4),
        n_load_workers=config["n_load_workers"],
        augment=False,
        seed=123,
    )

    # ---- Modelo ----
    model = QSpaceModel(
        in_features  = config["in_features"],
        query_dim    = config["query_dim"],
        hidden_dim   = config["hidden_dim"],
        latent_dim   = config["latent_dim"],
        n_enc_layers = config["n_enc_layers"],
        n_dec_layers = config["n_dec_layers"],
        omega_0      = config["omega_0"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModelo: {n_params:,} parâmetros treináveis")

    # ---- Eval only ----
    if args.eval_only:
        if args.resume is None:
            raise ValueError("--eval_only requer --resume")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        from torch.utils.data import DataLoader
        from dataset import collate_variable_dwi
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                                collate_fn=collate_variable_dwi)
        from metrics import evaluate_model
        metrics = evaluate_model(model, val_loader, device)
        print("\n=== Métricas de Avaliação ===")
        for k, v in sorted(metrics.items()):
            print(f"  {k:20s}: {v:.4f}")
        return

    # ---- Treino ----
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
    )
    trainer.fit(resume_from=args.resume)


if __name__ == "__main__":
    main()