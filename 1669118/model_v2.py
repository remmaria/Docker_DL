import torch
import torch.nn as nn
import torch.nn.functional as F


class QSpaceAttentionNetwork_v2(nn.Module):
    """
    Rede de atenção no espaço-q para interpolação e harmonização de DWI.

    Melhorias desta versão
    ----------------------
    1. Residual scaling learnable
       A rede começa próxima da física pura:
           output ≈ média ponderada
       e aprende gradualmente quanto residual adicionar.

    2. Softmax numericamente estável
       Evita overflow/NaN.

    3. Receptive field ampliado
       Uso de convoluções dilatadas (dilation=2)
       sem perder resolução espacial.

    4. Decoder mais estável
       GroupNorm adicional.

    5. SiLU no MLP
       Melhor suavidade geométrica no espaço-q.

    6. Clamp final
       Evita explosões no SSIM.

    7. Delta_b explícito
       Mantido — extremamente importante para cross-shell.
    """

    def __init__(
        self,
        k_neighbors,
        feature_channels=64,
        output_clamp=(0.0, 5.0),
    ):

        super().__init__()

        self.K = k_neighbors

        self.output_clamp = output_clamp

        # =================================================
        # hiperparâmetros learnable
        # =================================================

        self.res_scale = nn.Parameter(
            torch.tensor(0.1)
        )

        self.angular_temp = nn.Parameter(
            torch.tensor(0.1)
        )

        self.b_penalty = nn.Parameter(
            torch.tensor(2.0)
        )

        # =================================================
        # ENCODER
        # receptive field ampliado
        # =================================================

        self.encoder = nn.Sequential(

            # RF = 3
            nn.Conv3d(
                1,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                feature_channels,
            ),

            nn.SiLU(inplace=True),

            # RF cresce muito aqui
            # dilation=2 -> RF efetivo ~7
            nn.Conv3d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),

            nn.GroupNorm(
                8,
                feature_channels,
            ),

            nn.SiLU(inplace=True),
        )

        # =================================================
        # QUERY MLP
        # =================================================

        self.query_mlp = nn.Sequential(

            nn.Linear(
                k_neighbors * 4,
                feature_channels * 2,
            ),

            nn.LayerNorm(
                feature_channels * 2
            ),

            nn.SiLU(inplace=True),

            nn.Linear(
                feature_channels * 2,
                feature_channels,
            ),

            nn.Sigmoid(),
        )

        # =================================================
        # DECODER
        # =================================================

        self.decoder = nn.Sequential(

            # RF aumenta novamente
            nn.Conv3d(
                feature_channels + 1,
                feature_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),

            nn.GroupNorm(
                8,
                feature_channels,
            ),

            nn.SiLU(inplace=True),

            nn.Conv3d(
                feature_channels,
                feature_channels // 2,
                kernel_size=3,
                padding=1,
            ),

            nn.GroupNorm(
                8,
                feature_channels // 2,
            ),

            nn.SiLU(inplace=True),

            nn.Conv3d(
                feature_channels // 2,
                1,
                kernel_size=3,
                padding=1,
            ),
        )

        self._init_weights()

    # =====================================================
    # weight init
    # =====================================================

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv3d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu',
                )

                if m.bias is not None:
                    nn.init.constant_(
                        m.bias,
                        0,
                    )

            elif isinstance(m, nn.Linear):

                nn.init.xavier_normal_(
                    m.weight
                )

                nn.init.constant_(
                    m.bias,
                    0,
                )

        # residual começa zerado
        nn.init.zeros_(
            self.decoder[-1].weight
        )

        nn.init.zeros_(
            self.decoder[-1].bias
        )

    # =====================================================
    # forward
    # =====================================================

    def forward(
        self,
        x_neighbors,
        q_query,
        neighbors_coords,
    ):

        """
        Parameters
        ----------
        x_neighbors:
            [B, K, 1, H, W, D]

        q_query:
            [B, 4]

        neighbors_coords:
            [B, K, 4]
        """

        B, K, C, H, W, D = x_neighbors.shape

        # =================================================
        # q-space geometry
        # =================================================

        target_v = q_query[:, 1:]              # [B, 3]

        target_b = q_query[:, 0:1]             # [B, 1]

        neigh_vs = neighbors_coords[:, :, 1:]  # [B, K, 3]

        neigh_bs = neighbors_coords[:, :, 0]   # [B, K]

        # =================================================
        # angular similarity
        # =================================================

        dot_product = torch.sum(
            neigh_vs * target_v.unsqueeze(1),
            dim=-1,
        )

        # =================================================
        # shell penalty
        # =================================================

        b_diff = torch.abs(
            neigh_bs - target_b
        )

        combined_scores = (
            dot_product / (
                self.angular_temp.abs()
                + 1e-6
            )
        ) - (
            b_diff * self.b_penalty.abs()
        )

        # softmax estável
        combined_scores = (
            combined_scores
            - combined_scores.max(
                dim=1,
                keepdim=True,
            )[0]
        )

        weights = F.softmax(
            combined_scores,
            dim=1,
        )

        # =================================================
        # weighted physical interpolation
        # =================================================

        weights_vol = weights.view(
            B,
            K,
            1,
            1,
            1,
            1,
        )

        media_ponderada = torch.sum(
            x_neighbors * weights_vol,
            dim=1,
        )

        # =================================================
        # query embedding
        # =================================================

        delta_coords = (
            neighbors_coords
            - q_query.unsqueeze(1)
        )

        q_input = delta_coords.view(
            B,
            -1,
        )

        q_emb = self.query_mlp(
            q_input
        )

        q_emb = q_emb.view(
            B,
            -1,
            1,
            1,
            1,
        )

        # =================================================
        # encoder
        # =================================================

        x_flat = x_neighbors.view(
            B * K,
            C,
            H,
            W,
            D,
        )

        features = self.encoder(
            x_flat
        )

        features = features.view(
            B,
            K,
            -1,
            H,
            W,
            D,
        )

        # =================================================
        # weighted feature fusion
        # =================================================

        fused = torch.sum(
            features * weights.view(
                B,
                K,
                1,
                1,
                1,
                1,
            ),
            dim=1,
        )

        fused = fused * q_emb

        # =================================================
        # delta_b injection
        # =================================================

        b_target = q_query[:, 0]

        b_neighbors = neighbors_coords[
            :, :, 0
        ].mean(dim=1)

        delta_b = (
            b_target - b_neighbors
        )

        delta_b_vol = delta_b.view(
            B,
            1,
            1,
            1,
            1,
        ).expand(
            B,
            1,
            H,
            W,
            D,
        )

        fused_with_db = torch.cat(
            [
                fused,
                delta_b_vol,
            ],
            dim=1,
        )

        # =================================================
        # decoder
        # =================================================

        residuo = self.decoder(
            fused_with_db
        )

        # =================================================
        # residual scaling
        # =================================================

        output_final = (
            media_ponderada
            + self.res_scale * residuo
        )

        # =================================================
        # safety clamp
        # =================================================

        output_final = torch.clamp(
            output_final,
            self.output_clamp[0],
            self.output_clamp[1],
        )

        # =================================================
        # nan safety
        # =================================================

        output_final = torch.nan_to_num(
            output_final,
            nan=0.0,
            posinf=self.output_clamp[1],
            neginf=self.output_clamp[0],
        )

        residuo = torch.nan_to_num(
            residuo,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        media_ponderada = torch.nan_to_num(
            media_ponderada,
            nan=0.0,
            posinf=self.output_clamp[1],
            neginf=self.output_clamp[0],
        )

        return (
            output_final,
            residuo,
            media_ponderada,
        )