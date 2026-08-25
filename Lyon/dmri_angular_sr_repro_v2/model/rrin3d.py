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


def _conv3d(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
        nn.InstanceNorm3d(out_ch, affine=True),
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

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, use_quality_cond: bool = False):
        super().__init__()
        self.max_disp = max_disp
        self.use_quality_cond = use_quality_cond
        in_ch = 1 + 1 + 3 + 3 + 3 + 1  # vol_a, vol_b, bvec_a, bvec_b, bvec_t, t
        if use_quality_cond:
            in_ch += 2  # residual_norm, gap_norm (ver docstring da classe e RRIN3D)
        self.enc1 = _conv3d(in_ch, base_ch)
        self.enc2 = _conv3d(base_ch, base_ch * 2, stride=2)
        self.enc3 = _conv3d(base_ch * 2, base_ch * 4, stride=2)
        self.dec2 = _conv3d(base_ch * 4, base_ch * 2)
        self.dec1 = _conv3d(base_ch * 2 + base_ch * 2, base_ch)
        self.head = _conv3d(base_ch + base_ch, base_ch)
        self.out = nn.Conv3d(base_ch, 7, kernel_size=3, padding=1)  # 3(flow_a)+3(flow_b)+1(vis)

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

    def __init__(self, base_ch: int = 16):
        super().__init__()
        in_ch = 1 + 1 + 1  # blend, vol_a, vol_b
        self.net = nn.Sequential(
            _conv3d(in_ch, base_ch),
            _conv3d(base_ch, base_ch),
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

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, use_quality_cond: bool = False):
        super().__init__()
        self.use_quality_cond = use_quality_cond
        self.flow_net = FlowNet3D(base_ch=base_ch, max_disp=max_disp,
                                   use_quality_cond=use_quality_cond)
        self.refine_net = RefineNet3D(base_ch=base_ch)

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


if __name__ == "__main__":
    _smoke_test()