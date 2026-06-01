"""
losses.py
---------
Funções de loss com constraints físicos do sinal DWI.

Por que não só MSE?
  O sinal DWI S(b,g) = S0 · exp(-b · ADC(g)) tem propriedades físicas
  conhecidas. Usar só MSE deixa a rede livre para produzir resultados
  fisicamente impossíveis (sinal crescendo com b, assimetria antipodal).
  As regularizações abaixo "ensinam física" à rede.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Loss principal: reconstrução no espaço-q
# ---------------------------------------------------------------------------

class MaskedReconstructionLoss(nn.Module):
    """
    MSE entre o sinal predito e o alvo, apenas nas posições não-mascaradas
    (q_mask indica padding, não as direções mascaradas).

    Usa Huber loss (smooth L1) em vez de MSE puro:
    - Menos sensível a outliers (voxels com artefatos)
    - Gradientes mais estáveis no início do treino
    """

    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(
        self,
        S_pred: torch.Tensor,   # (B, N_query, 1)
        S_target: torch.Tensor, # (B, N_query)
        q_mask: torch.Tensor,   # (B, N_query) bool — True = padding
    ) -> torch.Tensor:
        S_pred = S_pred.squeeze(-1)                        # (B, N_query)
        loss = self.huber(S_pred, S_target)                # (B, N_query)
        # Zera posições de padding
        valid = (~q_mask).float()
        loss = (loss * valid).sum() / (valid.sum() + 1e-8)
        return loss


# ---------------------------------------------------------------------------
# Regularização 1: Monotonicidade em b
# ---------------------------------------------------------------------------

class MonotonicityLoss(nn.Module):
    """
    Penaliza violações da monotonicidade: S(b2, g) > S(b1, g) para b2 > b1.

    O sinal DWI deve DECRESCER com b para a mesma direção g.
    Amostramos pares (b1, b2) aleatórios do batch e penalizamos
    quando S_pred(b2) > S_pred(b1).

    loss = mean(ReLU(S_pred(b2, g) - S_pred(b1, g) + margin))
    """

    def __init__(self, margin: float = 0.01, n_pairs: int = 100):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs

    def forward(
        self,
        S_pred: torch.Tensor,   # (B, N_query, 1)
        b_vals: torch.Tensor,   # (B, N_query) — b-values das queries
        q_mask: torch.Tensor,   # (B, N_query) bool — True = padding
    ) -> torch.Tensor:
        S_pred = S_pred.squeeze(-1)    # (B, N_query)
        B, N = S_pred.shape

        # Só considera posições válidas (não-padding)
        valid = ~q_mask   # (B, N)

        total_loss = torch.tensor(0.0, device=S_pred.device)
        count = 0

        for b_idx in range(B):
            valid_idx = torch.where(valid[b_idx])[0]
            if len(valid_idx) < 2:
                continue

            # Amostra pares aleatórios
            n_valid = len(valid_idx)
            n_pairs = min(self.n_pairs, n_valid * (n_valid - 1) // 2)
            if n_pairs == 0:
                continue

            # Gera pares (i, j) com b_i < b_j
            i_idx = torch.randint(0, n_valid, (n_pairs * 2,), device=S_pred.device)
            j_idx = torch.randint(0, n_valid, (n_pairs * 2,), device=S_pred.device)

            # Filtra para i ≠ j e seleciona n_pairs pares
            diff_mask = i_idx != j_idx
            i_idx = i_idx[diff_mask][:n_pairs]
            j_idx = j_idx[diff_mask][:n_pairs]

            if len(i_idx) == 0:
                continue

            vi = valid_idx[i_idx]
            vj = valid_idx[j_idx]

            b_i = b_vals[b_idx, vi]
            b_j = b_vals[b_idx, vj]
            S_i = S_pred[b_idx, vi]
            S_j = S_pred[b_idx, vj]

            # Onde b_j > b_i, S_j deve ser menor que S_i
            higher_b = b_j > b_i
            if higher_b.sum() == 0:
                continue

            violation = F.relu(S_j[higher_b] - S_i[higher_b] + self.margin)
            total_loss = total_loss + violation.mean()
            count += 1

        if count == 0:
            return (S_pred * 0).sum()   # tensor com grad, valor 0
        return total_loss / count


# ---------------------------------------------------------------------------
# Regularização 2: Simetria antipodal
# ---------------------------------------------------------------------------

class AntipodalSymmetryLoss(nn.Module):
    """
    Penaliza assimetria: S(b, g) ≈ S(b, -g).

    O sinal DWI em tecidos biológicos é antipodalmente simétrico
    (a difusão não tem "direção preferencial" no sentido vetorial).

    Estratégia: para cada query (b, g), adiciona query (b, -g) e
    penaliza a diferença de predição.

    NOTA: Este loss é aplicado durante o forward com queries espelhadas.
    O modelo recebe as queries originais E as negadas.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        model_decoder,          # referência ao decoder para forward
        z: torch.Tensor,        # (B, latent_dim)
        q_query: torch.Tensor,  # (B, N_query, 4) = [b, gx, gy, gz]
        q_mask: torch.Tensor,   # (B, N_query) bool
    ) -> torch.Tensor:
        # Espelha as direções: [b, gx, gy, gz] → [b, -gx, -gy, -gz]
        q_antipodal = q_query.clone()
        q_antipodal[:, :, 1:] = -q_antipodal[:, :, 1:]   # inverte g

        with torch.no_grad():
            S_original  = model_decoder(z.detach(), q_query)      # (B, N_q, 1)
        S_antipodal = model_decoder(z, q_antipodal)                # (B, N_q, 1)

        diff = (S_original.squeeze(-1) - S_antipodal.squeeze(-1)).pow(2)
        valid = (~q_mask).float()
        loss = (diff * valid).sum() / (valid.sum() + 1e-8)
        return loss


# ---------------------------------------------------------------------------
# Regularização 3: Suavidade angular
# ---------------------------------------------------------------------------

class AngularSmoothnessLoss(nn.Module):
    """
    Penaliza variações bruscas entre direções angularmente próximas.

    S(b, g1) e S(b, g2) para g1 ≈ g2 devem ser similares.
    Isso é fisicamente correto: o perfil de difusão é suave na esfera.

    Analogia: é como um L2 no gradiente da função no espaço esférico.
    """

    def __init__(self, angle_threshold_deg: float = 20.0, weight: float = 0.5):
        super().__init__()
        self.cos_threshold = float(np.cos(np.radians(angle_threshold_deg)))
        self.weight = weight

    def forward(
        self,
        S_pred: torch.Tensor,   # (B, N_query, 1)
        q_query: torch.Tensor,  # (B, N_query, 4)
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        S_pred = S_pred.squeeze(-1)    # (B, N_query)
        bvecs = q_query[:, :, 1:]      # (B, N_query, 3)

        # Similaridade coseno entre todos os pares de direções no batch
        # Apenas computamos para o primeiro exemplo (amostral, economiza memória)
        B, N, _ = bvecs.shape

        # Normaliza
        norms = bvecs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        bvecs_norm = bvecs / norms

        # Produto interno: (B, N, N)
        cos_sim = torch.bmm(bvecs_norm, bvecs_norm.transpose(1, 2))

        # Pares próximos (excluindo diagonal)
        close_pairs = (cos_sim.abs() > self.cos_threshold)
        eye = torch.eye(N, device=S_pred.device).bool().unsqueeze(0)
        close_pairs = close_pairs & ~eye

        # Diferença de sinal entre pares próximos
        # S_i expandido: (B, N, 1) - (B, 1, N) = (B, N, N)
        S_diff = (S_pred.unsqueeze(2) - S_pred.unsqueeze(1)).pow(2)

        # Mask de validade
        valid = ~q_mask   # (B, N)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)

        active = close_pairs & pair_valid
        if active.sum() == 0:
            return (S_pred * 0).sum()   # grad-safe zero

        loss = (S_diff * active.float()).sum() / (active.float().sum() + 1e-8)
        return self.weight * loss


import numpy as np

# ---------------------------------------------------------------------------
# Loss combinada
# ---------------------------------------------------------------------------

class QSpaceLoss(nn.Module):
    """
    Combina todas as losses.

    λ1 · L_recon + λ2 · L_mono + λ3 · L_smooth
    (L_antipodal é calculada separadamente no training loop pois precisa
     de dois forwards)
    """

    def __init__(
        self,
        lambda_recon: float  = 1.0,
        lambda_mono: float   = 0.1,
        lambda_smooth: float = 0.05,
        huber_delta: float   = 0.1,
    ):
        super().__init__()
        self.lambda_recon  = lambda_recon
        self.lambda_mono   = lambda_mono
        self.lambda_smooth = lambda_smooth

        self.recon_loss  = MaskedReconstructionLoss(delta=huber_delta)
        self.mono_loss   = MonotonicityLoss()
        self.smooth_loss = AngularSmoothnessLoss()

    def forward(
        self,
        S_pred:   torch.Tensor,
        S_target: torch.Tensor,
        q_query:  torch.Tensor,
        q_mask:   torch.Tensor,
        b_vals:   torch.Tensor,
    ) -> dict:
        L_recon  = self.recon_loss(S_pred, S_target, q_mask)
        L_mono   = self.mono_loss(S_pred, b_vals, q_mask)
        L_smooth = self.smooth_loss(S_pred, q_query, q_mask)

        total = (self.lambda_recon  * L_recon
               + self.lambda_mono   * L_mono
               + self.lambda_smooth * L_smooth)

        return {
            "total":  total,
            "recon":  L_recon.detach(),
            "mono":   L_mono.detach(),
            "smooth": L_smooth.detach() if isinstance(L_smooth, torch.Tensor) and L_smooth.requires_grad else L_smooth,
        }