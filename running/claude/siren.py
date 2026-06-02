"""
siren.py
--------
SIREN (Sinusoidal Representation Network) para aprender a função contínua
S(b, g) → sinal DWI de um voxel/patch no espaço-q.

Referência: Sitzmann et al., 2020 — "Implicit Neural Representations with
Periodic Activation Functions".

Por que SIREN para espaço-q?
  - O sinal DWI é suave mas tem gradientes finos (anisotropia).
  - ReLU networks introduzem artefatos de segunda derivada.
  - Senos capturam exatamente a estrutura oscilatória de S(b,g).
"""

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Bloco elementar SIREN
# ---------------------------------------------------------------------------

class SirenLayer(nn.Module):
    """
    Uma camada linear seguida de ativação seno.

    y = sin(ω₀ · (Wx + b))

    ω₀ (omega_0): fator de frequência.
      - Primeira camada: omega_0 = 30  (padrão Sitzmann et al.)
      - Camadas ocultas: omega_0 = 30  (pode ser reduzido para sinais mais
        suaves, ex: 10-15 para espaço-q)
      - Última camada:  sem ativação seno (linear)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features, is_first)

    def _init_weights(self, in_features: int, is_first: bool):
        """
        Inicialização específica do SIREN — crítica para convergência.

        Primeira camada: U(-1/in, 1/in)
        Demais camadas:  U(-sqrt(6/in)/ω₀, sqrt(6/in)/ω₀)

        Isso preserva a distribuição de ativações ao longo da rede.
        """
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = np.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            # bias uniforme pequeno
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.is_last:
            return out          # última camada: linear (sem seno)
        return torch.sin(self.omega_0 * out)


# ---------------------------------------------------------------------------
# Encoder SIREN completo: (b, g, features_espaciais) → z_tecido
# ---------------------------------------------------------------------------

class SIRENEncoder(nn.Module):
    """
    Aprende uma representação latente z do tecido dado um conjunto de
    medições (b_i, g_i, S_i).

    Input por medição:
        [b_norm, g_x, g_y, g_z, SH_feats...]   →  shape (N_dwi, in_dim)

    Processo:
        1. Cada medição passa pelo SIREN → embedding por medição
        2. Pooling permutation-invariant (mean + max) sobre as N_dwi medições
        3. MLP final → z_tecido (128-d)

    Isso é um "set encoder": a ordem das direções não importa.
    """

    def __init__(
        self,
        in_features: int = 7,       # [b_norm, gx, gy, gz, SH0, SH2, SH4] ou custom
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_layers: int = 5,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- Bloco SIREN por medição ----
        layers = []
        layers.append(SirenLayer(in_features, hidden_dim, omega_0=omega_0, is_first=True))
        for _ in range(n_layers - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        # penúltima camada ainda com seno
        layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.siren_body = nn.Sequential(*layers)

        # ---- MLP após pooling ----
        # entrada = mean_pool(hidden) + max_pool(hidden) = 2 * hidden_dim
        self.pool_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N_dwi, in_features)
            B        = batch size (voxels ou patches)
            N_dwi    = número de direções fornecidas (pode variar entre protocolos)
            in_features = [b_norm, gx, gy, gz, + features opcionais]

        Retorna:
            z: (B, latent_dim)
        """
        # Processa cada medição independentemente
        B, N, _ = x.shape
        x_flat = x.view(B * N, -1)                     # (B*N, in_features)
        h = self.siren_body(x_flat)                     # (B*N, hidden_dim)
        h = h.view(B, N, -1)                            # (B, N, hidden_dim)

        # Pooling permutation-invariant
        h_mean = h.mean(dim=1)                          # (B, hidden_dim)
        h_max  = h.max(dim=1).values                    # (B, hidden_dim)
        h_pool = torch.cat([h_mean, h_max], dim=-1)     # (B, 2*hidden_dim)

        z = self.pool_mlp(h_pool)                       # (B, latent_dim)
        return z


class ProtocolEncoder(nn.Module):

    def __init__(
        self,
        in_dim=4,
        hidden_dim=64,
        protocol_dim=32,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, protocol_dim),
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Decoder: (z_tecido, b_query, g_query) → S_predito
# ---------------------------------------------------------------------------

class SIRENDecoder(nn.Module):
    """
    Dado o latente z do tecido e um ponto de query (b, g) no espaço-q,
    prediz o sinal S(b, g).

    Conditioning: z é injetado via FiLM (Feature-wise Linear Modulation).
    FiLM é melhor que concatenação porque permite modulação multiplicativa,
    equivalente a adaptar os "pesos" do decoder ao tecido específico.

    γ, β = MLP(z)
    h_out = γ * h + β   (após cada camada SIREN)
    """

    def __init__(
        self,
        query_dim: int = 4,
        latent_dim: int = 128,
        protocol_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 4,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Camadas SIREN do decoder (sem FiLM embutido — aplicamos manualmente)
        self.siren_layers = nn.ModuleList()
        self.siren_layers.append(
            SirenLayer(query_dim, hidden_dim, omega_0=omega_0, is_first=True)
        )
        for _ in range(n_layers - 1):
            self.siren_layers.append(
                SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0)
            )

        # FiLM generators: um por camada oculta (não na primeira)
        condition_dim = latent_dim + protocol_dim
        self.film_generators = nn.ModuleList([
            nn.Linear(condition_dim,
                    2 * hidden_dim)
            for _ in range(n_layers - 1)
        ])
        
        # Saída final: sinal escalar (ou multi-canal se quiser prever vários b0s)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),           # sinal DWI normalizado ∈ [0,1]
        )

    def forward(
        self,
        z: torch.Tensor,            # (B, latent_dim)
        q: torch.Tensor,            # (B, N_query, query_dim)
    ) -> torch.Tensor:
        """
        z: representação do tecido
        q: pontos de query no espaço-q  [b_norm, gx, gy, gz]

        Retorna:
            S_pred: (B, N_query, 1) — sinal predito em cada query
        """
        B, N_q, _ = q.shape
        h = q.view(B * N_q, -1)                 # (B*N_q, query_dim)

        # Primeira camada (sem FiLM)
        h = self.siren_layers[0](h)             # (B*N_q, hidden_dim)

        # Demais camadas com FiLM conditioning
        z_exp = z.unsqueeze(1).expand(-1, N_q, -1)     # (B, N_q, latent_dim)
        z_flat = z_exp.reshape(B * N_q, -1)            # (B*N_q, latent_dim)

        for i, (layer, film_gen) in enumerate(
            zip(self.siren_layers[1:], self.film_generators)
        ):
            h = layer(h)
            film = film_gen(z_flat)                     # (B*N_q, 2*hidden_dim)
            gamma, beta = film.chunk(2, dim=-1)
            h = (1 + gamma) * h + beta                 # FiLM modulation

        S_pred = self.output_layer(h)                   # (B*N_q, 1)
        return S_pred.view(B, N_q, 1)