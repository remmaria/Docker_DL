"""
sanity_check.py
---------------
Teste rápido de todos os componentes — roda em ~30 segundos na CPU.
Não requer nenhum dado real.

Execute antes de treinar para garantir que tudo está funcionando:
  python sanity_check.py
"""

import torch
import numpy as np
import sys
import time

def header(text):
    print(f"\n{'='*55}")
    print(f"  {text}")
    print(f"{'='*55}")

def ok(text):
    print(f"  ✓ {text}")

def fail(text, e):
    print(f"  ✗ {text}: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
header("1. Imports")
# ---------------------------------------------------------------------------
try:
    from siren import SIRENEncoder, SIRENDecoder, SirenLayer
    ok("siren.py importado")
except Exception as e:
    fail("siren.py", e)

try:
    from losses import QSpaceLoss, MonotonicityLoss, AntipodalSymmetryLoss
    ok("losses.py importado")
except Exception as e:
    fail("losses.py", e)

try:
    from trainer import QSpaceModel
    ok("trainer.py importado")
except Exception as e:
    fail("trainer.py", e)

try:
    from train import generate_synthetic_dwi, _patch_dataset_for_npy
    ok("train.py importado")
except Exception as e:
    fail("train.py", e)


# ---------------------------------------------------------------------------
header("2. SirenLayer — shapes e inicialização")
# ---------------------------------------------------------------------------
try:
    layer_first  = SirenLayer(4, 256, omega_0=30.0, is_first=True)
    layer_hidden = SirenLayer(256, 256, omega_0=30.0)
    layer_last   = SirenLayer(256, 1, is_last=True)

    x = torch.randn(8, 4)
    h = layer_first(x)
    assert h.shape == (8, 256), f"shape errado: {h.shape}"
    h2 = layer_hidden(h)
    assert h2.shape == (8, 256)
    out = layer_last(h2)
    assert out.shape == (8, 1)

    # Checa que ativações estão em [-1, 1] (seno)
    assert h.abs().max() <= 1.01, f"ativações fora do range: {h.abs().max():.3f}"
    ok(f"SirenLayer: shapes OK, ativações em [-1,1] ✓")
except Exception as e:
    fail("SirenLayer", e)


# ---------------------------------------------------------------------------
header("3. SIRENEncoder — forward com batch variável")
# ---------------------------------------------------------------------------
try:
    encoder = SIRENEncoder(in_features=5, hidden_dim=128, latent_dim=64, n_layers=3)

    # Batch: 4 voxels, 20 direções de contexto, 5 features
    x = torch.randn(4, 20, 5)
    z = encoder(x)
    assert z.shape == (4, 64), f"z shape errado: {z.shape}"

    # Testa com número diferente de direções (protocolo multi-shell)
    x2 = torch.randn(4, 45, 5)
    z2 = encoder(x2)
    assert z2.shape == (4, 64)

    ok(f"Encoder: (B=4, N=20) → z {z.shape} ✓")
    ok(f"Encoder: (B=4, N=45) → z {z2.shape} ✓ (N variável funciona)")
except Exception as e:
    fail("SIRENEncoder", e)


# ---------------------------------------------------------------------------
header("4. SIRENDecoder — FiLM conditioning")
# ---------------------------------------------------------------------------
try:
    decoder = SIRENDecoder(query_dim=4, latent_dim=64, hidden_dim=128, n_layers=3)

    z      = torch.randn(4, 64)
    q      = torch.randn(4, 15, 4)   # 15 queries
    S_pred = decoder(z, q)
    assert S_pred.shape == (4, 15, 1), f"shape errado: {S_pred.shape}"

    # Checa range [0,1] (Sigmoid)
    assert S_pred.min() >= 0 and S_pred.max() <= 1, \
        f"sinal fora de [0,1]: [{S_pred.min():.3f}, {S_pred.max():.3f}]"

    ok(f"Decoder: z{z.shape} + q{q.shape} → S{S_pred.shape} ✓")
    ok(f"Decoder: range [{S_pred.min():.3f}, {S_pred.max():.3f}] ⊂ [0,1] ✓")
except Exception as e:
    fail("SIRENDecoder", e)


# ---------------------------------------------------------------------------
header("5. QSpaceModel — forward completo")
# ---------------------------------------------------------------------------
try:
    model = QSpaceModel(
        in_features=5, query_dim=4, hidden_dim=128,
        latent_dim=64, n_enc_layers=3, n_dec_layers=3
    )

    B, N_ctx, N_q = 4, 25, 10
    x_ctx    = torch.randn(B, N_ctx, 5)
    q_query  = torch.randn(B, N_q, 4)
    ctx_mask = torch.zeros(B, N_ctx, dtype=torch.bool)
    ctx_mask[0, 20:] = True   # Simula padding no primeiro item

    S_pred, z = model(x_ctx, q_query, ctx_mask)
    assert S_pred.shape == (B, N_q, 1)
    assert z.shape == (B, 64)

    n_params = sum(p.numel() for p in model.parameters())
    ok(f"QSpaceModel forward: S_pred{S_pred.shape}, z{z.shape} ✓")
    ok(f"Parâmetros: {n_params:,}")
except Exception as e:
    fail("QSpaceModel", e)


# ---------------------------------------------------------------------------
header("6. QSpaceLoss — todas as componentes")
# ---------------------------------------------------------------------------
try:
    criterion = QSpaceLoss(lambda_recon=1.0, lambda_mono=0.1, lambda_smooth=0.05)

    S_pred   = torch.rand(4, 10, 1, requires_grad=True)
    S_target = torch.rand(4, 10)
    q_query  = torch.randn(4, 10, 4)
    q_mask   = torch.zeros(4, 10, dtype=torch.bool)
    q_mask[0, 8:] = True   # Padding nos últimos 2
    b_vals   = torch.tensor([[0,1000,1000,2000,2000,1000,0,2000,1000,1000]] * 4).float()

    losses = criterion(S_pred, S_target, q_query, q_mask, b_vals)

    assert "total" in losses and "recon" in losses and "mono" in losses
    assert losses["total"].requires_grad, "total loss não tem grad!"
    assert losses["total"].item() > 0

    ok(f"Loss total:  {losses['total'].item():.4f}")
    ok(f"Loss recon:  {losses['recon'].item():.4f}")
    ok(f"Loss mono:   {losses['mono'].item():.4f}")
    ok(f"Loss smooth: {losses['smooth'].item():.4f}")
except Exception as e:
    fail("QSpaceLoss", e)


# ---------------------------------------------------------------------------
header("7. Backward pass e gradientes")
# ---------------------------------------------------------------------------
try:
    model = QSpaceModel(in_features=5, query_dim=4, hidden_dim=64, latent_dim=32,
                        n_enc_layers=3, n_dec_layers=3)
    criterion = QSpaceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x_ctx    = torch.randn(4, 20, 5)
    q_query  = torch.randn(4, 8, 4)
    S_target = torch.rand(4, 8)
    q_mask   = torch.zeros(4, 8, dtype=torch.bool)
    b_vals   = torch.ones(4, 8) * 1000

    t0 = time.time()
    S_pred, z = model(x_ctx, q_query)
    losses = criterion(S_pred, S_target, q_query, q_mask, b_vals)
    losses["total"].backward()

    # Checa que todos os parâmetros têm gradiente
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"Parâmetros sem grad: {no_grad}"

    optimizer.step()
    optimizer.zero_grad()
    t1 = time.time()

    ok(f"Backward OK — nenhum parâmetro sem gradiente")
    ok(f"Tempo de 1 step: {(t1-t0)*1000:.1f}ms")
except Exception as e:
    fail("Backward pass", e)


# ---------------------------------------------------------------------------
header("8. Geração de dados sintéticos")
# ---------------------------------------------------------------------------
try:
    import shutil
    _patch_dataset_for_npy()

    for protocol in ["single_shell", "multi_shell"]:
        d = generate_synthetic_dwi(n_voxels=200, protocol=protocol,
                                   save_dir=f"_test_synth/{protocol}")
        ok(f"Protocolo '{protocol}' gerado: {d}")

    # Testa carregamento
    from dataset import MaskedQSpaceDataset
    ds = MaskedQSpaceDataset(
        ["_test_synth/single_shell", "_test_synth/multi_shell"],
        mask_ratio=0.3,
        voxels_per_subject=50,
        augment=False,
    )
    assert len(ds) > 0
    item = ds[0]
    assert "x_context" in item and "q_query" in item and "S_target" in item
    ok(f"Dataset: {len(ds)} amostras carregadas ✓")
    ok(f"Item[0]: context{item['x_context'].shape}, "
       f"query{item['q_query'].shape}, target{item['S_target'].shape}")

    # Cleanup
    shutil.rmtree("_test_synth", ignore_errors=True)
except Exception as e:
    fail("Dados sintéticos", e)


# ---------------------------------------------------------------------------
header("9. DataLoader com collate variável")
# ---------------------------------------------------------------------------
try:
    from torch.utils.data import DataLoader
    from dataset import collate_variable_dwi
    import shutil

    _patch_dataset_for_npy()

    generate_synthetic_dwi(n_voxels=100, protocol="single_shell", save_dir="_test_dl/sub_ss")
    generate_synthetic_dwi(n_voxels=100, protocol="multi_shell",  save_dir="_test_dl/sub_ms")

    ds = MaskedQSpaceDataset(
        ["_test_dl/sub_ss", "_test_dl/sub_ms"],
        mask_ratio=0.3, voxels_per_subject=40, augment=False
    )
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_variable_dwi)
    batch = next(iter(loader))

    assert batch["x_context"].shape[0] == 8
    assert batch["S_target"].shape[0] == 8
    ok(f"Batch: x_context{batch['x_context'].shape}, "
       f"q_query{batch['q_query'].shape} ✓")
    ok(f"ctx_mask sum (padding): {batch['ctx_mask'].sum().item()} posições")

    shutil.rmtree("_test_dl", ignore_errors=True)
except Exception as e:
    fail("DataLoader", e)


# ---------------------------------------------------------------------------
header("TODOS OS TESTES PASSARAM ✓")
# ---------------------------------------------------------------------------
print("\n  Pronto para treinar. Execute:")
print("  python train.py --synthetic --output_dir runs/test_synth\n")