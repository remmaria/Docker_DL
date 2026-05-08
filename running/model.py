import torch
import torch.nn as nn
import torch.nn.functional as F


class QSpaceAttentionNetwork(nn.Module):
    """
    Rede de atenção no espaço-q para interpolação e harmonização de DWI.

    Arquitetura: média ponderada angular (física) + resíduo aprendido.

    Mudanças desta versão
    ---------------------
    1. feature_channels: 32 → 64
       Capacidade dobrada no encoder/decoder para lidar com a tarefa
       cross-shell, que exige modelar diferenças de decaimento T2.

    2. delta_b injetado como canal espacial no decoder
       A diferença de b-value entre input e target (Δb = b_target - b_input)
       é broadcast para [B, 1, H, W, D] e concatenada às features fundidas
       antes do decoder. Isso dá ao modelo a física diretamente:
         - Δb = 0   → interpolação angular pura (fase 1)
         - Δb ≠ 0   → harmonização cross-shell (fases 2 e 3)
       Sem esse canal, o decoder precisa inferir Δb implicitamente a partir
       dos delta_coords no MLP — possível, mas muito mais difícil.

    3. Decoder agora recebe feature_channels + 1 canais (features + delta_b)
       e tem uma camada intermediária a mais para absorver a nova informação.
    """

    def __init__(self, k_neighbors, feature_channels=64):
        super(QSpaceAttentionNetwork, self).__init__()
        self.K = k_neighbors

        # 1. ENCODER — extrai features espaciais de cada vizinho
        self.encoder = nn.Sequential(
            nn.Conv3d(1, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(inplace=True),
        )

        # 2. QUERY MLP — codifica a geometria dos deltas no espaço-q
        # Entrada: [B, K*4] — delta entre cada vizinho e o target
        # Saída:   [B, feature_channels] — máscara multiplicativa de atenção
        self.query_mlp = nn.Sequential(
            nn.Linear(k_neighbors * 4, feature_channels * 2),
            nn.LayerNorm(feature_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels * 2, feature_channels),
            nn.Sigmoid(),
        )

        # 3. DECODER — recebe features (C) + canal delta_b (1) = C+1 canais
        # O canal delta_b carrega a física do decaimento T2 cross-shell
        # diretamente no espaço de features volumétrico.
        self.decoder = nn.Sequential(
            nn.Conv3d(feature_channels + 1, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 2, 1, kernel_size=3, padding=1),
        )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Inicialização Kaiming para as camadas internas (bom para ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # A "SACADA" PARA O RES_RATIO:
        # Forçamos a última camada do decoder a começar em ZERO.
        # Assim, no início do treino: Output = Média Física + 0
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
    def forward(self, x_neighbors, q_query, neighbors_coords):
        """
        Parâmetros
        ----------
        x_neighbors      : [B, K, 1, H, W, D]  — patches dos vizinhos normalizados
        q_query          : [B, 4]               — (b_norm, gx, gy, gz) do target
        neighbors_coords : [B, K, 4]            — coordenadas q de cada vizinho

        Retorna
        -------
        output_final  : [B, 1, H, W, D] — predição final (média + resíduo)
        residuo       : [B, 1, H, W, D] — correção aprendida pelo decoder
        media_ponderada: [B, 1, H, W, D] — baseline de interpolação angular
        """
        B, K, C, H, W, D = x_neighbors.shape
        
        target_v = q_query[:, 1:]        # [B, 3]
        target_b = q_query[:, 0:1]       # [B, 1]
        neigh_vs = neighbors_coords[:, :, 1:] # [B, K, 3]
        neigh_bs = neighbors_coords[:, :, 0]  # [B, K]

        # 1. Similaridade Angular (Cosseno)
        dot_product = torch.sum(neigh_vs * target_v.unsqueeze(1), dim=-1) # [B, K]
        
        # 2. Penalidade por distância de Shell (Diferença absoluta de b-value)
        # Quanto mais longe o b do vizinho estiver do b do target, menor o peso.
        # O fator 2.0 é um hiperparâmetro de 'sharpness' para o b-value
        b_diff = torch.abs(neigh_bs - target_b)
        
        # Combinamos ambos antes do Softmax
        # Note: subtraímos b_diff porque queremos que valores menores aumentem a probabilidade
        combined_scores = (dot_product / 0.1) - (b_diff * 2.0) 
        weights = F.softmax(combined_scores, dim=1) # [B, K]
        # weights: [B, K] -> [B, K, 1, 1, 1, 1]
        weights_vol = weights.view(B, K, 1, 1, 1, 1) 
        media_ponderada = torch.sum(x_neighbors * weights_vol, dim=1)

        # --- PASSO 2: DELTA DE COORDENADAS → ATENÇÃO GEOMÉTRICA ---
        delta_coords = neighbors_coords - q_query.unsqueeze(1)  # [B, K, 4]
        q_input      = delta_coords.view(B, -1)                  # [B, K*4]
        q_emb        = self.query_mlp(q_input)                   # [B, feature_channels]
        q_emb        = q_emb.view(B, -1, 1, 1, 1)               # [B, C, 1, 1, 1]

        # --- PASSO 3: ENCODER + FUSÃO DE FEATURES ---
        x_flat   = x_neighbors.view(B * K, C, H, W, D)
        features = self.encoder(x_flat)               # [B*K, feature_channels, H, W, D]
        features = features.view(B, K, -1, H, W, D)

        fused = torch.sum(
            features * weights.view(B, K, 1, 1, 1, 1), dim=1  # [B, C, H, W, D]
        )

        # Atenção geométrica: escala canal a canal pelas importâncias do MLP
        fused = fused * q_emb  # [B, C, H, W, D]

        # --- PASSO 4: INJEÇÃO DO DELTA_B (física do decaimento T2) ---
        # delta_b = b_target - b_input_médio, normalizado pelo mesmo bval_max
        # usado no dataset. Valor positivo = subindo de shell; negativo = descendo.
        # Broadcast para volume inteiro: [B] → [B, 1, H, W, D]
        b_target    = q_query[:, 0]                          # [B]
        b_neighbors = neighbors_coords[:, :, 0].mean(dim=1) # [B] — média dos b_input
        delta_b     = (b_target - b_neighbors)               # [B]
        delta_b_vol = delta_b.view(B, 1, 1, 1, 1).expand(B, 1, H, W, D)  # [B, 1, H, W, D]

        # Concatena delta_b às features: [B, C+1, H, W, D]
        fused_with_db = torch.cat([fused, delta_b_vol], dim=1)

        # --- PASSO 5: DECODER → RESÍDUO ---
        residuo = self.decoder(fused_with_db)  # [B, 1, H, W, D]

        output_final = media_ponderada + residuo

        return output_final, residuo, media_ponderada