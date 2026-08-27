"""
Rede inspirada em RRIN/RIFE (interpolacao de quadros de video por fluxo
otico + refinamento residual), adaptada para 3D e para o par de direcoes
de entrada + posicao relativa `t` escolhidos por
scripts/02b_build_rrin_triplets.py -- ver protocolo, secao 10.1 (a linha
"original" da tese, retomada como diagnostico quantitativo de o quanto a
suposicao de fluxo otico entre direcoes de gradiente se sustenta).

IMPORTANTE (leia antes de interpretar resultados) -- isto NAO e um
"RRIN para dMRI" no sentido de reproduzir a arquitetura 2D original
componente a componente, e uma adaptacao 3D enxuta que reaproveita a MESMA
IDEIA central (estimar fluxo entre duas amostras "vizinhas" + interpolar
por warping + refinar com um residuo), por dois motivos:
  (a) a RRIN original e uma rede 2D (video); aqui trabalhamos com patches
      3D (como o RCAE, ver model/rcae.py) -- encoder/decoder de fluxo,
      warping e refinamento foram reimplementados diretamente em 3D, nao
      portados linha a linha do codigo 2D original.
  (b) o objetivo desta rede e TESTAR a hipotese de fluxo entre direcoes de
      gradiente (e quantificar o quanto/onde ela falha), nao maximizar
      fidelidade a um paper -- a arquitetura foi mantida deliberadamente
      simples (poucos canais, 2 niveis de downsample) pra caber em patches
      pequenos (10-24^3) e treinar rapido, nao pra competir com o RCAE em
      capacidade. Uma comparacao "justa" de capacidade (nº de parametros)
      entre RRIN3D e RCAE fica como checagem a fazer depois, se o
      resultado preliminar pedir isso.

Fluxo (analogo ao VFI classico -- RRIN/RIFE):
  1. FlowNet3D: recebe vol_a, vol_b (as duas direcoes de entrada do par
     escolhido por 02b_build_rrin_triplets.py) + a geometria da trinca
     (bvec_a, bvec_b, bvec_alvo, t) e prediz dois campos de fluxo 3D
     (flow_a->t, flow_b->t) e um mapa de visibilidade V.
  2. warp3d: aplica cada campo de fluxo (grid_sample trilinear) a vol_a e
     vol_b, produzindo warped_a, warped_b.
  3. Blend: combina warped_a/warped_b ponderado por (1-t)/t (posicao
     relativa do alvo no arco) E pelo mapa de visibilidade V -- mesmo
     espirito da combinacao usada em RRIN/RIFE.
  4. RefineNet3D: rede pequena de refinamento que ve o blend + os dois
     volumes de entrada crus e prediz um residuo somado ao blend --
     corrige erros de warping/oclusao que o fluxo sozinho nao capturaria.

Condicionamento opcional na QUALIDADE da trinca (`use_quality_cond`, ver
docstring de RRIN3D mais abaixo): alem de bvec_a/bvec_b/bvec_alvo/t, a rede
pode receber tambem `residual_deg`/`gap_deg` da trinca (normalizados) como
2 canais extras -- em vez de so aceitar/rejeitar trincas ruins fora do
treino, isso deixa a rede aprender a confiar menos no fluxo quando a
geometria nao sustenta a suposicao de interpolacao. Desativado por padrao
(mantem o teste "cego" mais proximo de VFI de video de verdade); ativar e
comparar as duas versoes separa o quanto da limitacao e falta de contexto
(corrigivel) do que e estrutural.

Requer PyTorch (nao disponivel neste ambiente de desenvolvimento -- revisado
manualmente, testado apenas por compilacao de sintaxe; validar no cluster
com `python -m model.rrin3d`, smoke test no fim do arquivo, mesmo padrao de
model/rcae.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm3d(norm_type: str, out_ch: int) -> nn.Module:
    """Camada de normalizacao usada em `_conv3d`, escolhida por `norm_type`.
    Ver protocolo ("artefato de costura de patch") para o motivo de existir
    a opcao "batch" alem do "instance" original.

    "instance" (default, comportamento ORIGINAL/compativel com todos os
    checkpoints ja treinados): `InstanceNorm3d` calcula media/variancia por
    amostra, sobre a extensao espacial (D,H,W) do que estiver dentro do
    patch atual. Como a reconstrucao usa sliding-window com overlap parcial
    (ver scripts/05b_reconstruct_rrin.py, --patch-size/--stride), o MESMO
    voxel cai em patches vizinhos com conteudo ao redor ligeiramente
    diferente -> estatisticas diferentes -> normalizacoes diferentes ->
    "costura" visivel entre patches na reconstrucao (confirmado
    empiricamente: o artefato listrado atenua bastante com stride menor,
    mais overlap). Aumentar o overlap (stride menor) e um paliativo (dilui
    a diferenca via media ponderada do overlap-add), nao remove a causa.

    "batch": `BatchNorm3d` resolve a causa raiz, nao so o sintoma -- em
    modo de avaliacao (model.eval()), BatchNorm usa `running_mean`/
    `running_var` FIXOS (acumulados ao longo de TODO o treino), nao
    estatisticas calculadas a partir do patch/batch atual. Ou seja, na
    reconstrucao a normalizacao vira uma transformacao afim fixa por canal,
    igual nao importa em qual patch/janela o voxel caiu -- elimina a
    costura por construcao. Custo: precisa ser treinado do ZERO (as
    estatisticas/parametros aprendidos nao sao intercambiaveis com um
    checkpoint "instance" existente) e depende de lotes de treino
    razoavelmente estaveis (--batch-size, ver scripts/04b_train_rrin.py --
    o default 8, combinado com a extensao espacial de cada patch, da uma
    amostra estatistica efetiva grande o suficiente na pratica).

    NAO confundir com GroupNorm: GroupNorm tambem calcula estatisticas
    sobre a extensao espacial da amostra ATUAL (so muda o agrupamento de
    canais), entao sofre do MESMO problema de costura que InstanceNorm --
    por isso nao foi oferecido aqui como alternativa."""
    if norm_type == "instance":
        return nn.InstanceNorm3d(out_ch, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm3d(out_ch, affine=True)
    raise ValueError(f"norm_type desconhecido: {norm_type!r} (use 'instance' ou 'batch')")


def _conv3d(in_ch: int, out_ch: int, stride: int = 1, norm_type: str = "instance") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        _norm3d(norm_type, out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


def _repeat_vec_3d(vec: torch.Tensor, spatial_shape) -> torch.Tensor:
    """vec: (B, C) -> (B, C, D, H, W), broadcast espacial -- mesma ideia de
    `_repeat_bvec` em model/rcae.py, mas sem eixo de "direcoes" (aqui so ha
    1 par por item, nao uma sequencia de N direcoes)."""
    b, c = vec.shape
    x = vec.view(b, c, 1, 1, 1)
    return x.expand(b, c, *spatial_shape)


def warp3d(vol: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp trilinear de `vol` (B,C,D,H,W) por um campo de deslocamento
    `flow` (B,3,D,H,W), em unidades NORMALIZADAS (-1..1 cobre o volume
    inteiro em cada eixo -- mesma convencao de F.grid_sample). Canais de
    `flow` na ordem (dx,dy,dz) correspondendo a (W,H,D) -- a ordem que
    F.grid_sample espera no ultimo eixo do grid (x,y,z) = (W,H,D)."""
    b, _c, d, h, w = vol.shape
    device = vol.device
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, d, device=device),
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy, zz], dim=-1)                    # (D,H,W,3), ordem (x,y,z)
    base_grid = base_grid.unsqueeze(0).expand(b, -1, -1, -1, -1)     # (B,D,H,W,3)
    flow_grid = flow.permute(0, 2, 3, 4, 1)                          # (B,D,H,W,3), ja em (dx,dy,dz)
    sample_grid = torch.clamp(base_grid + flow_grid, -1.0, 1.0)
    return F.grid_sample(vol, sample_grid, mode="bilinear", padding_mode="border",
                          align_corners=True)


class FlowNet3D(nn.Module):
    """U-Net 3D pequena (2 niveis de downsample) que prediz, a partir de
    (vol_a, vol_b, bvec_a, bvec_b, bvec_alvo, t): dois campos de fluxo
    (flow_a, flow_b, 3 canais cada) e um mapa de visibilidade V (1 canal,
    logit antes da sigmoid)."""

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, use_quality_cond: bool = False,
                 norm_type: str = "instance"):
        super().__init__()
        self.max_disp = max_disp
        self.use_quality_cond = use_quality_cond
        self.norm_type = norm_type
        in_ch = 1 + 1 + 3 + 3 + 3 + 1  # vol_a, vol_b, bvec_a, bvec_b, bvec_t, t
        if use_quality_cond:
            in_ch += 2  # residual_norm, gap_norm (ver docstring da classe e RRIN3D)
        self.enc1 = _conv3d(in_ch, base_ch, norm_type=norm_type)
        self.enc2 = _conv3d(base_ch, base_ch * 2, stride=2, norm_type=norm_type)
        self.enc3 = _conv3d(base_ch * 2, base_ch * 4, stride=2, norm_type=norm_type)
        self.dec2 = _conv3d(base_ch * 4, base_ch * 2, norm_type=norm_type)
        self.dec1 = _conv3d(base_ch * 2 + base_ch * 2, base_ch, norm_type=norm_type)
        self.head = _conv3d(base_ch + base_ch, base_ch, norm_type=norm_type)
        self.out = nn.Conv3d(base_ch, 7, kernel_size=3, padding=1)  # 3(flow_a)+3(flow_b)+1(vis)
        # inicializacao "morna" (zero-init da ultima camada, pratica padrao em
        # redes de fluxo optico/STN -- ver protocolo, sugestao de melhoria do
        # RRIN): com peso/bias zerados, a saida bruta comeca em 0 -> flow_a=
        # flow_b=tanh(0)*max_disp=0 (nenhum deslocamento) e vis=sigmoid(0)=0.5
        # (neutro entre a e b), em vez de valores aleatorios pequenos vindos
        # da inicializacao padrao do Conv3d. Isso NAO trava o gradiente (so a
        # ULTIMA camada e zerada, as anteriores continuam com init normal,
        # entao ha sinal suficiente pra rede se afastar de zero conforme
        # precisar) -- so evita que o treino comece propondo deslocamentos
        # grandes e arbitrarios antes de aprender que "nao fazer nada" ja e
        # uma predicao razoavel de partida. So afeta treinos NOVOS (do zero,
        # sem --resume-checkpoint/last.pt existente) -- carregar um
        # checkpoint via load_state_dict sobrescreve isso normalmente.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None):
        spatial = vol_a.shape[-3:]
        bvec_a_map = _repeat_vec_3d(bvec_a, spatial)
        bvec_b_map = _repeat_vec_3d(bvec_b, spatial)
        bvec_t_map = _repeat_vec_3d(bvec_t, spatial)
        t_map = _repeat_vec_3d(t.view(-1, 1), spatial)
        parts = [vol_a, vol_b, bvec_a_map, bvec_b_map, bvec_t_map, t_map]
        if self.use_quality_cond:
            if quality is None:
                raise ValueError("use_quality_cond=True mas `quality` nao foi passado ao forward")
            parts.append(_repeat_vec_3d(quality, spatial))  # quality: (B,2) -> (B,2,D,H,W)
        x = torch.cat(parts, dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(e3)
        d2 = F.interpolate(d2, size=e2.shape[-3:], mode="trilinear", align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-3:], mode="trilinear", align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)

        feat = self.head(d1)
        raw = self.out(feat)
        flow_a = torch.tanh(raw[:, 0:3]) * self.max_disp
        flow_b = torch.tanh(raw[:, 3:6]) * self.max_disp
        vis_logit = raw[:, 6:7]
        return flow_a, flow_b, vis_logit


class RefineNet3D(nn.Module):
    """Rede de refinamento residual (pequena): ve o blend inicial + os dois
    volumes de entrada crus (sem warp) e prediz um residuo somado ao
    blend -- mesma ideia do "residue refinement" da RRIN, adaptada em 3D."""

    def __init__(self, base_ch: int = 16, norm_type: str = "instance"):
        super().__init__()
        in_ch = 1 + 1 + 1  # blend, vol_a, vol_b
        self.net = nn.Sequential(
            _conv3d(in_ch, base_ch, norm_type=norm_type),
            _conv3d(base_ch, base_ch, norm_type=norm_type),
            nn.Conv3d(base_ch, 1, kernel_size=3, padding=1),
        )

    def forward(self, blend, vol_a, vol_b):
        x = torch.cat([blend, vol_a, vol_b], dim=1)
        return self.net(x)


class RRIN3D(nn.Module):
    """Modelo completo: FlowNet3D -> warp -> blend (por t e visibilidade)
    -> RefineNet3D (residuo). Ver docstring do modulo para a analogia com
    RRIN/RIFE e as diferencas deliberadas.

    use_quality_cond (default False -- ver protocolo secao 10.1): quando
    True, a rede recebe tambem `residual_deg`/`gap_deg` da trinca (a
    distancia perpendicular ao plano da circunferencia e o quanto os dois
    pares de entrada estao afastados -- ver utils/gradients.py e
    scripts/02b_build_rrin_triplets.py), normalizados por 90 (o maximo
    possivel com simetria antipodal), como 2 canais extras de
    condicionamento no FlowNet3D. A ideia: em vez de so filtrar trincas
    ruins fora do treino (binario valido/invalido), dar a rede a
    informacao de QUAO boa e a trinca deixa ela aprender a confiar menos no
    fluxo (ex.: visibilidade quase uniforme, fluxo perto de zero) quando a
    geometria nao sustenta a suposicao de interpolacao -- em vez de tentar
    "adivinhar" um warping as cegas com um par ruim.

    Duas leituras validas ao comparar use_quality_cond=False vs True:
    False e o teste mais "puro" da hipotese de fluxo (a rede nao sabe se o
    par que recebeu e bom ou ruim, igual um modelo de VFI de video nunca
    precisa saber isso -- a proximidade dos quadros e sempre implicitamente
    pequena); True e a versao mais "pratica" (deve performar melhor, mas
    passa a testar uma pergunta ligeiramente diferente: "uma rede consciente
    da qualidade geometrica da trinca consegue compensar" em vez de "o
    fluxo entre direcoes de gradiente funciona como em video"). Rodar as
    duas e comparar diz o quanto da limitacao e falta de contexto
    (corrigivel) vs. estrutural (nao tem fluxo pra aprender ali, seja qual
    for o contexto dado).

    norm_type (default "instance" -- ver docstring de `_norm3d`): "instance"
    e o comportamento ORIGINAL (compativel com todos os checkpoints ja
    treinados). "batch" troca por BatchNorm3d, que resolve de vez o
    artefato de "costura" entre patches na reconstrucao (ver protocolo,
    "artefato de patch-tiling/InstanceNorm3d") -- mas exige treinar do
    ZERO (nao carrega em cima de um checkpoint "instance"), ver
    scripts/04b_train_rrin.py --norm-type.

    Uso:
        model = RRIN3D(use_quality_cond=True)
        pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=quality)
    vol_a, vol_b: (B, 1, D, H, W) -- as duas direcoes de entrada do par
        (pair_a/pair_b de scripts/02b_build_rrin_triplets.py).
    bvec_a, bvec_b, bvec_t: (B, 3) -- vetores unitarios de gradiente.
    t: (B,) -- posicao relativa do alvo no arco (0 = coincide com a, 1 =
        coincide com b), campo `t_frac` do mesmo esquema de trincas.
    quality: (B, 2) ou None -- [residual_deg/90, gap_deg/90] da trinca;
        obrigatorio se use_quality_cond=True, ignorado se False.
    retorna: (B, 1, D, H, W) -- direcao-alvo predita.
    """

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, use_quality_cond: bool = False,
                 norm_type: str = "instance"):
        super().__init__()
        self.use_quality_cond = use_quality_cond
        self.norm_type = norm_type
        self.flow_net = FlowNet3D(base_ch=base_ch, max_disp=max_disp,
                                   use_quality_cond=use_quality_cond, norm_type=norm_type)
        self.refine_net = RefineNet3D(base_ch=base_ch, norm_type=norm_type)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None):
        flow_a, flow_b, vis_logit = self.flow_net(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t,
                                                    quality=quality)
        warped_a = warp3d(vol_a, flow_a)
        warped_b = warp3d(vol_b, flow_b)

        vis = torch.sigmoid(vis_logit)  # (B,1,D,H,W) -- peso relativo de warped_a
        t_map = t.view(-1, 1, 1, 1, 1)
        # combina a ponderacao "temporal" (1-t)/t (quanto mais perto de a,
        # mais peso pra warped_a) com o mapa de visibilidade aprendido --
        # mesmo espirito de RRIN/RIFE (que combinam (1-t) com uma mascara
        # de oclusao aprendida).
        w_a = (1.0 - t_map) * vis
        w_b = t_map * (1.0 - vis)
        denom = (w_a + w_b).clamp(min=1e-6)
        blend = (w_a * warped_a + w_b * warped_b) / denom

        residual = self.refine_net(blend, vol_a, vol_b)
        return blend + residual


class FlowNet3DLayered(nn.Module):
    """Generalizacao em K camadas da FlowNet3D (ver protocolo secao 13,
    "Toward a layered-flow extension for crossing fibers"): em vez de um
    unico par de campos de fluxo + 1 mapa de visibilidade, prediz K camadas
    INDEPENDENTES, cada uma com seu proprio par de fluxo (flow_a, flow_b) e
    seu proprio mapa de visibilidade, mais um logit de selecao de camada por
    voxel (combinado por softmax entre as K camadas em RRIN3DLayered).

    Motivacao: um unico mapa de visibilidade so decide "confia mais em a ou
    em b" -- bom pra oclusao simples (o caso "VFI de video" padrao), mas nao
    consegue representar um voxel que e uma MISTURA de duas populacoes de
    fibra diferentes (crossing), onde nenhum warp unico explica o voxel
    inteiro. Cada camada pode em tese se especializar numa populacao
    diferente; qual camada "vence" em cada voxel e decidido pelo softmax de
    selecao, aprendido end-to-end so com a loss de reconstrucao (ver
    RRIN3DLayered.forward e a discussao no protocolo sobre nao usar nenhuma
    supervisao externa tipo CSD/peak-count de inicio -- so acrescentar isso
    se for observado colapso de modo).

    NAO usar para K=1 -- para K=1 use FlowNet3D (arquitetura original com
    exatamente os mesmos parametros/comportamento), que mantem
    compatibilidade com os checkpoints ja treinados (rrin, rrin_qc, etc.)."""

    def __init__(self, num_layers: int, base_ch: int = 16, max_disp: float = 0.5,
                 use_quality_cond: bool = False, norm_type: str = "instance"):
        super().__init__()
        if num_layers < 2:
            raise ValueError("FlowNet3DLayered e para num_layers>=2 -- use FlowNet3D para K=1")
        self.num_layers = num_layers
        self.max_disp = max_disp
        self.use_quality_cond = use_quality_cond
        self.norm_type = norm_type
        in_ch = 1 + 1 + 3 + 3 + 3 + 1  # vol_a, vol_b, bvec_a, bvec_b, bvec_t, t
        if use_quality_cond:
            in_ch += 2
        self.enc1 = _conv3d(in_ch, base_ch, norm_type=norm_type)
        self.enc2 = _conv3d(base_ch, base_ch * 2, stride=2, norm_type=norm_type)
        self.enc3 = _conv3d(base_ch * 2, base_ch * 4, stride=2, norm_type=norm_type)
        self.dec2 = _conv3d(base_ch * 4, base_ch * 2, norm_type=norm_type)
        self.dec1 = _conv3d(base_ch * 2 + base_ch * 2, base_ch, norm_type=norm_type)
        self.head = _conv3d(base_ch + base_ch, base_ch, norm_type=norm_type)
        # por camada: 3 (flow_a) + 3 (flow_b) + 1 (vis_logit) + 1 (layer_logit)
        self.out = nn.Conv3d(base_ch, 8 * num_layers, kernel_size=3, padding=1)
        # mesma inicializacao "morna" de FlowNet3D (ver comentario la): saida
        # bruta comeca em 0 -> flow=0, vis=0.5 em toda camada, e o
        # layer_logit tambem comeca em 0 -> softmax uniforme entre as K
        # camadas (1/K cada) -- ponto de partida neutro, sem nenhuma camada
        # favorecida a priori. So afeta treinos NOVOS (sem resume).
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None):
        spatial = vol_a.shape[-3:]
        bvec_a_map = _repeat_vec_3d(bvec_a, spatial)
        bvec_b_map = _repeat_vec_3d(bvec_b, spatial)
        bvec_t_map = _repeat_vec_3d(bvec_t, spatial)
        t_map = _repeat_vec_3d(t.view(-1, 1), spatial)
        parts = [vol_a, vol_b, bvec_a_map, bvec_b_map, bvec_t_map, t_map]
        if self.use_quality_cond:
            if quality is None:
                raise ValueError("use_quality_cond=True mas `quality` nao foi passado ao forward")
            parts.append(_repeat_vec_3d(quality, spatial))
        x = torch.cat(parts, dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(e3)
        d2 = F.interpolate(d2, size=e2.shape[-3:], mode="trilinear", align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-3:], mode="trilinear", align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)

        feat = self.head(d1)
        raw = self.out(feat)  # (B, 8*K, D, H, W)
        b = raw.shape[0]
        spatial_shape = raw.shape[2:]
        K = self.num_layers
        raw = raw.view(b, K, 8, *spatial_shape)
        flow_a = torch.tanh(raw[:, :, 0:3]) * self.max_disp   # (B, K, 3, D, H, W)
        flow_b = torch.tanh(raw[:, :, 3:6]) * self.max_disp   # (B, K, 3, D, H, W)
        vis_logit = raw[:, :, 6]                              # (B, K, D, H, W)
        layer_logit = raw[:, :, 7]                            # (B, K, D, H, W)
        return flow_a, flow_b, vis_logit, layer_logit


class RRIN3DLayered(nn.Module):
    """Generalizacao em K camadas de RRIN3D -- ver FlowNet3DLayered acima e
    protocolo secao 13 ("Toward a layered-flow extension for crossing
    fibers"). Com K camadas, cada camada faz warp+blend exatamente como a
    RRIN3D original (fluxo bidirecional + visibilidade); as K camadas sao
    entao combinadas por um softmax POR VOXEL sobre um logit de selecao de
    camada.

    IMPORTANTE (pra nao confundir "K" com algo que varia por voxel): K e um
    hiperparametro FIXO e GLOBAL da arquitetura, escolhido antes do treino e
    igual pra todo voxel/sujeito/n_level -- a rede sempre calcula as K
    camadas em todo lugar. O que varia por voxel e o USO dessas camadas: os
    pesos do softmax (`pi`, ver forward) sao computados independentemente
    em cada posicao espacial, entao um voxel de fibra unica pode aprender a
    colapsar quase todo peso numa camada so (comportamento efetivo K=1
    localmente), e um voxel de crossing pode aprender a dividir o peso entre
    duas -- tudo aprendido end-to-end so com a loss de reconstrucao,
    SEM nenhuma supervisao externa dizendo quantas fibras tem ali. A
    recomendacao no protocolo e comecar assim (K=2/K=3 "crus") e so
    considerar uma supervisao auxiliar tipo CSD/peak-count (calculada da
    aquisicao COMPLETA, nunca do n_level subamostrado -- ver protocolo) se
    for observado colapso de modo (as K camadas convergindo pra prever a
    mesma coisa, sem se especializar espacialmente).

    Para K=1, use RRIN3D (nao esta classe) -- mantem compatibilidade exata
    (mesmos parametros, mesmo comportamento) com os checkpoints ja
    treinados (rrin, rrin_qc, rrin_qc_inclinv etc.). Use `build_rrin_model`
    abaixo para nao ter que decidir isso manualmente em cada script.
    """

    def __init__(self, num_layers: int, base_ch: int = 16, max_disp: float = 0.5,
                 use_quality_cond: bool = False, norm_type: str = "instance"):
        super().__init__()
        if num_layers < 2:
            raise ValueError("RRIN3DLayered e para num_layers>=2 -- use RRIN3D para K=1")
        self.num_layers = num_layers
        self.use_quality_cond = use_quality_cond
        self.norm_type = norm_type
        self.flow_net = FlowNet3DLayered(num_layers=num_layers, base_ch=base_ch,
                                          max_disp=max_disp, use_quality_cond=use_quality_cond,
                                          norm_type=norm_type)
        self.refine_net = RefineNet3D(base_ch=base_ch, norm_type=norm_type)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None,
                return_layers=False):
        flow_a, flow_b, vis_logit, layer_logit = self.flow_net(
            vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=quality)
        K = self.num_layers
        t_map = t.view(-1, 1, 1, 1, 1)  # (B,1,1,1,1) -- broadcast com (B,1,D,H,W)

        pi = torch.softmax(layer_logit, dim=1)  # (B, K, D, H, W), soma 1 por voxel

        blend = torch.zeros_like(vol_a)
        for k in range(K):
            warped_a_k = warp3d(vol_a, flow_a[:, k])      # (B,1,D,H,W)
            warped_b_k = warp3d(vol_b, flow_b[:, k])
            vis_k = torch.sigmoid(vis_logit[:, k:k + 1])  # (B,1,D,H,W)
            # mesmo espirito da RRIN3D original: combina (1-t)/t com a
            # visibilidade DESSA camada -- cada camada tem sua propria
            # nocao de "quanto confiar em a vs b".
            w_a = (1.0 - t_map) * vis_k
            w_b = t_map * (1.0 - vis_k)
            denom = (w_a + w_b).clamp(min=1e-6)
            layer_blend_k = (w_a * warped_a_k + w_b * warped_b_k) / denom
            pi_k = pi[:, k:k + 1]  # (B,1,D,H,W) -- peso desta camada, por voxel
            blend = blend + pi_k * layer_blend_k

        residual = self.refine_net(blend, vol_a, vol_b)
        out = blend + residual
        if return_layers:
            # util pra inspecionar os mapas de pi/flow por camada (ver
            # protocolo -- checar se aparece estrutura espacial parecida
            # com regioes de crossing conhecidas, tipo centrum semiovale).
            return out, {"pi": pi, "flow_a": flow_a, "flow_b": flow_b, "vis_logit": vis_logit}
        return out


def build_rrin_model(num_layers: int = 1, base_ch: int = 16, max_disp: float = 0.5,
                      use_quality_cond: bool = False, norm_type: str = "instance"):
    """Escolhe RRIN3D (K=1, arquitetura original) ou RRIN3DLayered (K>=2,
    ver docstring de RRIN3DLayered) de acordo com `num_layers`. Usar esta
    funcao em scripts/04b_train_rrin.py e scripts/05b_reconstruct_rrin.py
    em vez de instanciar as classes diretamente, para as duas ficarem
    sempre sincronizadas (o checkpoint grava `num_layers`/`norm_type` em
    `args`, e a reconstrucao le de la -- ver scripts/05b_reconstruct_rrin.py).

    norm_type: "instance" (default, compativel com todos os checkpoints ja
    treinados) ou "batch" (resolve de vez o artefato de costura entre
    patches na reconstrucao, ver docstring de `_norm3d` -- exige treinar do
    zero)."""
    if num_layers <= 1:
        return RRIN3D(base_ch=base_ch, max_disp=max_disp, use_quality_cond=use_quality_cond,
                       norm_type=norm_type)
    return RRIN3DLayered(num_layers=num_layers, base_ch=base_ch, max_disp=max_disp,
                          use_quality_cond=use_quality_cond, norm_type=norm_type)


def _smoke_test():
    """Forward pass com tensores pequenos aleatorios, so pra checar shapes
    -- mesmo padrao de model/rcae.py. Testa as duas variantes
    (use_quality_cond False/True). Rodar no cluster: python -m model.rrin3d"""
    torch.manual_seed(0)
    b, d, h, w = 2, 10, 10, 10
    vol_a = torch.rand(b, 1, d, h, w)
    vol_b = torch.rand(b, 1, d, h, w)
    bvec_a = torch.randn(b, 3); bvec_a = bvec_a / bvec_a.norm(dim=-1, keepdim=True)
    bvec_b = torch.randn(b, 3); bvec_b = bvec_b / bvec_b.norm(dim=-1, keepdim=True)
    bvec_t = torch.randn(b, 3); bvec_t = bvec_t / bvec_t.norm(dim=-1, keepdim=True)
    t = torch.rand(b)
    expected = (b, 1, d, h, w)

    model = RRIN3D(base_ch=8, use_quality_cond=False)
    out = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
    assert out.shape == expected, f"shape mismatch (sem quality): {out.shape} != {expected}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"smoke test OK (use_quality_cond=False), output shape: {tuple(out.shape)}, "
          f"{n_params} parametros")

    model_q = RRIN3D(base_ch=8, use_quality_cond=True)
    quality = torch.rand(b, 2)  # [residual_norm, gap_norm], ambos em [0,1]
    out_q = model_q(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=quality)
    assert out_q.shape == expected, f"shape mismatch (com quality): {out_q.shape} != {expected}"
    n_params_q = sum(p.numel() for p in model_q.parameters())
    print(f"smoke test OK (use_quality_cond=True), output shape: {tuple(out_q.shape)}, "
          f"{n_params_q} parametros")

    # confere que esquecer `quality` com use_quality_cond=True falha alto e
    # claro, nao silenciosamente com shape errado
    try:
        model_q(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
        raise AssertionError("deveria ter levantado ValueError sem `quality`")
    except ValueError:
        print("OK: chamar sem `quality` com use_quality_cond=True levanta ValueError, como esperado")

    # --- variantes em camadas (K>=2, ver protocolo secao 13) ---
    for K in (2, 3):
        for use_qc in (False, True):
            model_k = build_rrin_model(num_layers=K, base_ch=8, use_quality_cond=use_qc)
            assert isinstance(model_k, RRIN3DLayered)
            quality_k = torch.rand(b, 2) if use_qc else None
            out_k, layers = model_k(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t,
                                     quality=quality_k, return_layers=True)
            assert out_k.shape == expected, \
                f"shape mismatch (K={K}, use_qc={use_qc}): {out_k.shape} != {expected}"
            assert layers["pi"].shape == (b, K, d, h, w)
            # softmax por voxel deve somar 1 ao longo das K camadas
            pi_sum = layers["pi"].sum(dim=1)
            assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5), \
                "pesos de camada (pi) nao somam 1 por voxel"
            n_params_k = sum(p.numel() for p in model_k.parameters())
            print(f"smoke test OK (RRIN3DLayered, K={K}, use_quality_cond={use_qc}), "
                  f"output shape: {tuple(out_k.shape)}, {n_params_k} parametros")

    # build_rrin_model(num_layers=1) deve devolver a arquitetura ORIGINAL
    # (RRIN3D), nao RRIN3DLayered -- garante compatibilidade de checkpoint.
    model_default = build_rrin_model(num_layers=1, base_ch=8)
    assert isinstance(model_default, RRIN3D) and not isinstance(model_default, RRIN3DLayered)
    assert isinstance(model_default.flow_net.enc1[1], nn.InstanceNorm3d)
    print("OK: build_rrin_model(num_layers=1) devolve RRIN3D (nao RRIN3DLayered), "
          "compatibilidade de checkpoint preservada")

    # --- norm_type="batch" (ver protocolo, artefato de patch-tiling) ---
    for K in (1, 2):
        model_bn = build_rrin_model(num_layers=K, base_ch=8, norm_type="batch")
        assert isinstance(model_bn.flow_net.enc1[1], nn.BatchNorm3d)
        out_bn = model_bn(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
        if K > 1:
            out_bn = out_bn[0]
        assert out_bn.shape == expected, f"shape mismatch (norm_type=batch, K={K}): {out_bn.shape}"
        # BatchNorm3d em eval() usa running stats fixas, independentes do
        # patch/batch atual -- exatamente a propriedade que remove a
        # costura entre patches na reconstrucao.
        model_bn.eval()
        with torch.no_grad():
            out_bn_eval = model_bn(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
            if K > 1:
                out_bn_eval = out_bn_eval[0]
        assert out_bn_eval.shape == expected
        print(f"smoke test OK (norm_type=batch, K={K}), output shape: {tuple(out_bn_eval.shape)}")
    try:
        _norm3d("groupnorm_nao_existe", 8)
        raise AssertionError("norm_type invalido deveria levantar ValueError")
    except ValueError:
        print("OK: norm_type invalido levanta ValueError, como esperado")


if __name__ == "__main__":
    _smoke_test()