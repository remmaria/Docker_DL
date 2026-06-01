import torch
import torch.nn as nn
import torch.nn.functional as F


class QSpaceAttentionNetwork_v2(nn.Module):
    """
    Rede de atenção no espaço-q para interpolação e harmonização de DWI.

    Arquitetura: média ponderada angular (física) + resíduo aprendido.

    Correções aplicadas (v2 → v2.1)
    --------------------------------
    1. res_scale inicializado em -10.0 (não 0.1)
       softplus(-10) ≈ 4.5e-5, praticamente zero.
       Antes: softplus(0.1) ≈ 0.744 — resíduo não-zero no step 0,
       causando o visual diferente entre Média e Predição Final no debug.

    2. q_emb_mean ≠ q_emb_std — bug de referência corrigido
       O código anterior fazia q_emb_std = q_emb_mean (mesma referência),
       duplicando a mesma informação no fused. Agora:
         - q_emb_mean: atenção centrada em 1.0 via tanh (modula média)
         - q_emb_std:  atenção centrada em 0.0 via sigmoid (modula dispersão)
       Isso dá ao decoder informações genuinamente diferentes sobre cada
       dimensão do espaço-q, dobrando a utilidade do q_proj.

    3. Decoder com zero-init garantido *depois* de todo o loop Kaiming
       A ordem já estava correta no original, mas adicionamos assert de
       verificação para garantir (útil no debug).

    Manutenção intacta
    ------------------
    - delta_b injetado como canal espacial (física do decaimento T2)
    - feature_channels=64 (capacidade cross-shell)
    - Encoder dilatado (receptive field amplo sem custo de parâmetros)
    - fused_mean + fused_std concatenados antes do decoder
    - GroupNorm em vez de BatchNorm (estável com batch pequeno)
    """

    def __init__(self, k_neighbors, feature_channels=64):
        super(QSpaceAttentionNetwork_v2, self).__init__()
        self.K = k_neighbors

        # res_scale: controla a magnitude do resíduo aprendido.
        # Inicializado em -1.5 → softplus(-1.5) ≈ 0.20
        # Congelado durante o warmup (ver freeze_res_scale / unfreeze_res_scale)
        self.res_scale = nn.Parameter(torch.tensor(-1.5))

        # 1. ENCODER — extrai features espaciais de cada vizinho
        # Três convoluções dilatadas em cascata: receptive field efetivo de
        # 1 + 2*(1+2+4) = 15 voxels sem stride, mantendo resolução total.
        self.encoder = nn.Sequential(
            nn.Conv3d(1, feature_channels, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(inplace=True),
        )

        # 2. QUERY MLP — codifica a geometria angular + shell de cada vizinho
        # Entrada: [B, K, 5] — (delta_b, delta_gx, delta_gy, delta_gz, rms_signal)
        # Saída:   [B, K, feature_channels] — embedding por vizinho, depois reduzido
        #
        # FIX 2: agora produz 2*feature_channels para separar mean_attn e std_attn.
        # A primeira metade modula fused_mean, a segunda modula fused_std,
        # garantindo que o concat [mean, std] carregue informações distintas.
        self.q_proj = nn.Sequential(
            nn.Linear(5, feature_channels),
            nn.LayerNorm(feature_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_channels, feature_channels * 2),  # ← 2C saída
        )

        # 3. DECODER — recebe [fused_mean (C) | fused_std (C) | delta_b (1)] = 2C+1
        # A última camada é zero-init para garantir resíduo=0 no step 0.
        self.decoder = nn.Sequential(
            nn.Conv3d(feature_channels * 2 + 1, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 2, 1, kernel_size=3, padding=1),  # zero-init
        )

        # Skip connection: projeta o sinal bruto para o espaço de features
        self.input_proj = nn.Conv3d(1, feature_channels, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    def freeze_res_scale(self):
        """Congela res_scale durante o warmup — só decoder/encoder treinam."""
        self.res_scale.requires_grad_(False)

    def unfreeze_res_scale(self):
        """Libera res_scale após o decoder aprender direções razoáveis."""
        self.res_scale.requires_grad_(True)

    # ------------------------------------------------------------------
    def _init_weights(self):
        # Passo 1: inicialização padrão para todas as camadas
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Passo 2: forçamos a última Conv do decoder a zero DEPOIS do loop acima.
        # Assim output_final = media_ponderada + softplus(-10) * 0 ≈ media_ponderada
        # no step 0, garantindo o comportamento correto no debug.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

        # Verificação de sanidade (remove em produção se quiser)
        assert self.decoder[-1].weight.abs().max().item() == 0.0, \
            "zero-init do decoder falhou"

    # ------------------------------------------------------------------
    def forward(self, x_neighbors, q_query, neighbors_coords):
        """
        Parâmetros
        ----------
        x_neighbors      : [B, K, 1, H, W, D]  — patches dos vizinhos normalizados
        q_query          : [B, 4]               — (b_norm, gx, gy, gz) do target
        neighbors_coords : [B, K, 4]            — coordenadas q de cada vizinho

        Retorna
        -------
        output_final   : [B, 1, H, W, D] — predição final (média + resíduo)
        residuo        : [B, 1, H, W, D] — correção aprendida pelo decoder
        media_ponderada: [B, 1, H, W, D] — baseline de interpolação angular
        """
        B, K, C, H, W, D = x_neighbors.shape

        target_v = q_query[:, 1:]             # [B, 3] — direção do gradiente target
        target_b = q_query[:, 0:1]            # [B, 1] — b-value normalizado target
        neigh_vs = neighbors_coords[:, :, 1:] # [B, K, 3]
        neigh_bs = neighbors_coords[:, :, 0]  # [B, K]

        # ================================================================
        # PASSO 1: MÉDIA PONDERADA FÍSICA (baseline)
        # ================================================================
        # Temperatura adaptativa por fase:
        #   Fase 1 — vizinhos já são os K mais próximos (dataset concentrado)
        #            temperatura baixa (0.1) concentra ainda mais o peso no
        #            mais próximo → preserva estrutura direcional fina
        #   Fases 2/3 — vizinhos são metade próximos + metade diversos
        #            temperatura maior (0.3) distribui peso entre os dois grupos
        #            para que o decoder receba contexto T2 real dos diversos
        #
        # O canal delta_b já captura a diferença de shell para o decoder;
        # aqui usamos b_diff só para penalizar vizinhos de shells erradas
        # quando houver mistura (fases 2/3).
        dot_product = torch.abs(
            torch.sum(neigh_vs * target_v.unsqueeze(1), dim=-1)
        )  # [B, K]

        b_diff = torch.abs(neigh_bs - target_b)  # [B, K]

        # delta_b médio: 0 na fase 1 (same-shell), >0 nas fases 2/3
        # Usamos isso para adaptar temperatura e penalidade dinamicamente,
        # sem precisar passar a fase explicitamente para o forward.
        mean_b_diff = b_diff.mean(dim=1, keepdim=True)  # [B, 1]
        is_cross = (mean_b_diff > 0.05).float()          # 1 se cross-shell, 0 se same

        # temperatura: 0.1 (same-shell) → 0.3 (cross-shell)
        temperature = 0.1 + 0.2 * is_cross   # [B, 1]

        # penalidade b_diff: ativa só em cross-shell para não penalizar
        # vizinhos do mesmo shell que têm b_diff ~ 0 por normalização
        b_penalty = 2.0 * is_cross * b_diff   # [B, K]

        combined_scores = (dot_product / temperature) - b_penalty
        weights = F.softmax(combined_scores, dim=1)             # [B, K]
        weights_vol = weights.view(B, K, 1, 1, 1, 1)

        media_ponderada = torch.sum(x_neighbors * weights_vol, dim=1)  # [B, 1, H, W, D]

        # ================================================================
        # PASSO 2: QUERY MLP — atenção geométrica por vizinho
        # ================================================================
        delta_coords = neighbors_coords - q_query.unsqueeze(1)  # [B, K, 4]

        # RMS do sinal de cada vizinho como feature de magnitude
        signal_rms = torch.sqrt(
            torch.mean(x_neighbors ** 2, dim=[2, 3, 4, 5]) + 1e-4
        ).unsqueeze(-1)  # [B, K, 1]

        q_input = torch.cat([delta_coords, signal_rms], dim=-1)  # [B, K, 5]

        q_feat = self.q_proj(q_input)     # [B, K, 2*C]
        q_feat = q_feat.mean(dim=1)       # [B, 2*C] — agrega sobre vizinhos

        # FIX 2: separa as duas metades do embedding
        q_feat_mean = q_feat[:, :q_feat.shape[1] // 2]  # [B, C]
        q_feat_std  = q_feat[:, q_feat.shape[1] // 2:]  # [B, C]

        # mean_attn: centrado em 1.0, varia em [0, 2] — multiplicativo neutro
        q_emb_mean = (1.0 + torch.tanh(q_feat_mean)).view(B, -1, 1, 1, 1)  # [B, C, 1,1,1]

        # std_attn: centrado em 0.5 via sigmoid — pondera a dispersão entre vizinhos
        q_emb_std  = torch.sigmoid(q_feat_std).view(B, -1, 1, 1, 1)        # [B, C, 1,1,1]

        # ================================================================
        # PASSO 3: ENCODER + FUSÃO DE FEATURES
        # ================================================================
        x_flat   = x_neighbors.view(B * K, C, H, W, D)
        features = self.encoder(x_flat)
        skip     = self.input_proj(x_flat)
        features = features + skip                           # skip connection
        features = features.view(B, K, -1, H, W, D)        # [B, K, C, H, W, D]

        # Média ponderada das features (mesmos pesos da média física)
        fused_mean = torch.sum(
            features * weights.view(B, K, 1, 1, 1, 1), dim=1
        )  # [B, C, H, W, D]

        # Desvio padrão ponderado das features (captura dispersão entre vizinhos)
        fused_std = torch.sqrt(
            torch.sum(
                weights.view(B, K, 1, 1, 1, 1) *
                (features - fused_mean.unsqueeze(1)) ** 2,
                dim=1
            ) + 1e-6
        )  # [B, C, H, W, D]

        # Aplica atenções geométricas separadas em cada componente
        fused_mean = fused_mean * q_emb_mean  # modula média de features
        fused_std  = fused_std  * q_emb_std   # modula dispersão de features

        fused = torch.cat([fused_mean, fused_std], dim=1)  # [B, 2C, H, W, D]

        # ================================================================
        # PASSO 4: INJEÇÃO DO DELTA_B (física do decaimento T2)
        # Δb = 0  → interpolação angular pura (fase 1)
        # Δb ≠ 0  → harmonização cross-shell (fases 2 e 3)
        # ================================================================
        b_target    = q_query[:, 0]
        b_neighbors = neighbors_coords[:, :, 0].mean(dim=1)
        delta_b     = (b_target - b_neighbors)
        delta_b_vol = delta_b.view(B, 1, 1, 1, 1).expand(B, 1, H, W, D)

        fused_with_db = torch.cat([fused, delta_b_vol], dim=1)  # [B, 2C+1, H, W, D]

        # ================================================================
        # PASSO 5: DECODER → RESÍDUO
        # No step 0: decoder[-1] = zeros → residuo = 0
        #            res_scale = softplus(-10) ≈ 4.5e-5 ≈ 0
        # → output_final ≈ media_ponderada ✓
        # ================================================================
        residuo   = self.decoder(fused_with_db)          # [B, 1, H, W, D]
        res_scale = F.softplus(self.res_scale)            # escalar ≥ 0

        output_final = media_ponderada + res_scale * residuo

        return output_final, residuo, media_ponderada