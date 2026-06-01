# q-SHINE — Stage 1: SIREN + Masked Q-Space Modeling

## Estrutura dos arquivos

```
qshine/
├── siren.py           # Arquitetura: SIRENEncoder, SIRENDecoder, SirenLayer
├── losses.py          # Losses: reconstrução + monotonicidade + suavidade
├── dataset.py         # Dataset: masking, normalização, DataLoader
├── trainer.py         # Loop de treino, scheduler, checkpointing
├── metrics.py         # Métricas e visualizações
├── train.py           # Script principal (entry point)
├── sanity_check.py    # Teste rápido de todos os componentes
└── requirements.txt
```

---

## Instalação

```bash
# Ambiente virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Dependências
pip install -r requirements.txt

# Com suporte a GPU (CUDA 11.8):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Teste rápido (sem dados reais)

```bash
# 1. Verifica que todos os componentes funcionam (~30 segundos)
python sanity_check.py

# 2. Treino com dados sintéticos (~5 min na CPU, ~1 min na GPU)
python train.py --synthetic --output_dir runs/test_synth

# 3. Visualiza curvas no TensorBoard
tensorboard --logdir runs/test_synth/tensorboard
```

---

## Estrutura esperada dos dados reais

```
/data/subjects/
├── sub-001/
│   ├── dwi.nii.gz          # (X, Y, Z, N_dwi) — float32
│   ├── bvals               # uma linha, N_dwi valores em s/mm²
│   ├── bvecs               # (3 x N_dwi) ou (N_dwi x 3) — FSL convention
│   └── brain_mask.nii.gz   # (X, Y, Z) — binária
├── sub-002/
│   └── ...
```

---

## Treino com dados reais

```bash
# Treino básico
python train.py \
    --data_dir /data/subjects \
    --output_dir runs/experiment_01

# Com config customizada
python train.py \
    --data_dir /data/subjects \
    --output_dir runs/experiment_02 \
    --config my_config.json

# Retomar treino interrompido
python train.py \
    --data_dir /data/subjects \
    --output_dir runs/experiment_01 \
    --resume runs/experiment_01/last_checkpoint.pt

# Apenas avaliação
python train.py \
    --data_dir /data/subjects \
    --eval_only \
    --resume runs/experiment_01/best_model.pt
```

---

## Config customizada (my_config.json)

```json
{
    "epochs": 150,
    "batch_size": 64,
    "lr": 5e-5,
    "hidden_dim": 512,
    "latent_dim": 256,
    "n_enc_layers": 6,
    "mask_ratio": 0.35,
    "masking_strategy": "shell",
    "lambda_mono": 0.2,
    "voxels_per_subject": 5000
}
```

---

## Estratégias de masking (progressão recomendada)

| Semana | Estratégia | Dificuldade | O que treina |
|--------|-----------|-------------|--------------|
| 1      | `random`  | Fácil       | Interpolação geral |
| 1-2    | `angular` | Médio       | Interpolação direcional |
| 2      | `shell`   | Difícil     | Extrapolação em b |

Troque no config: `"masking_strategy": "angular"`

---

## Diagnóstico — o que observar no TensorBoard

### Sinal saudável de treino:
- `train/loss_epoch` decresce suavemente nas primeiras 20 épocas
- `train/mono` deve cair para < 0.01 rapidamente (a rede aprende física)
- `val/mae_b1000` < 0.05 indica boa reconstrução single-shell
- `val/mae_b2000` e `val/mae_b3000` > `val/mae_b1000` é esperado (mais difícil)

### Sinais de problema:
| Sintoma | Causa provável | Solução |
|---------|---------------|---------|
| Loss oscila muito | LR muito alta | Reduza `lr` para 5e-5 |
| Loss não desce | omega_0 inadequado | Tente omega_0=10 para dados suaves |
| mono loss alta | Sinal crescendo com b | Aumente `lambda_mono` para 0.3 |
| Overfitting (val >> train) | Poucas amostras | Reduza `voxels_per_subject` |
| NaN no loss | Exploding gradients | clip_grad_norm já está em 1.0; reduza LR |

---

## Outputs gerados

```
runs/experiment_01/
├── config.json             # Config salva
├── best_model.pt           # Melhor checkpoint (menor val_loss)
├── last_checkpoint.pt      # Último checkpoint
└── tensorboard/            # Logs do TensorBoard
```

### Conteúdo do checkpoint:
```python
ckpt = torch.load("runs/experiment_01/best_model.pt")
# ckpt["model_state_dict"]      → pesos do modelo
# ckpt["epoch"]                 → época do melhor resultado
# ckpt["val_loss"]              → val loss no melhor checkpoint
# ckpt["config"]                → config usada no treino
```

---

## Inferência após treino

```python
import torch
from trainer import QSpaceModel
import numpy as np

# Carrega modelo
ckpt = torch.load("runs/experiment_01/best_model.pt", map_location="cpu")
model = QSpaceModel(**{k: ckpt["config"][k] for k in
    ["in_features","query_dim","hidden_dim","latent_dim","n_enc_layers","n_dec_layers"]})
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Para um voxel com N=30 medições:
# x_context: (1, N_ctx, 5) = [b_norm, gx, gy, gz, S_obs]
# q_query:   (1, N_q, 4)   = [b_norm, gx, gy, gz] — novas direções

with torch.no_grad():
    S_pred, z = model(x_context, q_query)
# S_pred: (1, N_q, 1) — sinal normalizado predito
# z:      (1, 128)    — representação latente do tecido
```

---

## Próximos passos (Semana 3-4)

Após validar a reconstrução, adicionar:
1. **Protocol conditioning no decoder** — `p = MLP(b_values, n_dirs, n_shells)`
2. **Adversarial head** — gradient reversal para disentanglement
3. **Contrastive loss** anatômico — pares de regiões via atlas

Esses componentes são construídos sobre o encoder/decoder desta semana.