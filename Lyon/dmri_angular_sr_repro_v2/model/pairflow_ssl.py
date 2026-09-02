"""
Fluxo bidirecional 3D AUTO-SUPERVISIONADO entre pares de direcoes reais
(etapas 4g/4h/5j -- addendum secao 20.15, ideia originada da pergunta da
usuaria sobre fluxo optico auto-supervisionado / EMA-VFI).

MOTIVACAO: RRIN3D (model/rrin3d.py) so aprende fluxo em TRINCAS curadas
(par + UM alvo real dentro do teto de residuo, ver
utils/gradients.py:spherical_triplet_residual e
scripts/02b_build_rrin_triplets.py) -- o numero de trincas validas e' o
gargalo que fez M "nao importar" sob threshold apertado (secoes 19/20.12).
A ideia aqui: treinar o fluxo entre A e B usando SO os proprios dois
extremos como supervisao mutua (bidirecional: flow_ab reconstroi B a
partir de A via warp, flow_ba reconstroi A a partir de B) -- SEM precisar
de nenhum terceiro ponto real "entre" eles. Isso libera o universo de
pares de treino de "trincas que passam no teto de residuo" pra "qualquer
par de direcoes medidas" (O(N^2) por sujeito, ver
utils/pairflow_ssl_dataset.py), ao custo de nunca ver, durante este
pre-treino, nenhuma correcao direta no MEIO do arco.

Por isso a proposta e' em DUAS ETAPAS:

  Etapa 1 (auto-supervisionada, `PairFlowNet3D` + `pairflow_ssl_losses`
  abaixo, treinada por scripts/04g_train_pairflow_ssl.py sobre
  utils/pairflow_ssl_dataset.py:PairFlowSSLDataset): pre-treina o fluxo
  bidirecional no pool grande e sem curadoria de trinca.

  Etapa 2 (supervisionada, `PairFlowInterp3D` abaixo, treinada por
  scripts/04h_train_pairflow_finetune.py sobre as MESMAS trincas de
  sempre via utils/rrin_dataset.py:RRINTripletDataset): parte do fluxo
  pre-treinado (opcionalmente congelado via --freeze-flow), extrapola
  LINEARMENTE para o t do alvo real -- suposicao de "velocidade angular
  constante" ao longo do arco, mesma ideia usada por Deep Voxel Flow (Liu
  et al., ICCV 2017) e pela combinacao linear de fluxo do Super SloMo
  (Jiang et al., CVPR 2018) pra virar "fluxo entre 2 pontos" em "fluxo pra
  t arbitrario" -- faz blend + RefineNet3D, e ai SIM e' corrigida contra o
  alvo real, ancorando a extrapolacao no unico lugar onde existe verdade
  fundamental para checar (o que o pre-treino puramente auto-supervisionado
  da Etapa 1, sozinho, nunca tem).

Reaproveita de proposito os blocos GENERICOS ja usados por mais de um
modelo desta linha (`_conv3d`/`_repeat_vec_3d`/`warp3d`/`RefineNet3D`,
todos de model/rrin3d.py) -- mesmo espirito de reaproveitamento ja
registrado para model/implicit_angular.py (secao 20.11): zero duplicacao
de blocos genericos, mas este modulo continua ARQUITETURALMENTE
INDEPENDENTE (nao importa `RRIN3D`/`RRIN3DStar` em si, so as pecas
soltas), entao pode ser comparado como uma linha a parte no comparativo
final.

Requer PyTorch (nao disponivel neste ambiente de desenvolvimento --
revisado manualmente, testado apenas por compilacao de sintaxe; validar no
cluster com `python -m model.pairflow_ssl` -- smoke test no fim do
arquivo, mesmo padrao de model/rrin3d.py/model/implicit_angular.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rrin3d import _conv3d, _repeat_vec_3d, warp3d, RefineNet3D


class PairFlowNet3D(nn.Module):
    """U-Net 3D pequena (MESMA topologia de 2 niveis de downsample de
    `model.rrin3d.FlowNet3D`), mas SEM condicionamento de alvo (sem
    `bvec_t`, sem `t`): recebe so `(vol_a, vol_b, bvec_a, bvec_b)` e preve
    UM UNICO campo de fluxo (3 canais) -- "quanto deslocar cada voxel do
    GRID DE SAIDA pra amostrar `vol_a` e reconstruir `vol_b`" (convencao de
    warp BACKWARD, mesma de `warp3d`/`FlowNet3D`: `warp3d(vol_a, flow_ab)`
    deve aproximar `vol_b`).

    Pra obter o fluxo NO SENTIDO CONTRARIO (B->A), os MESMOS pesos sao
    reaplicados aos argumentos trocados -- `vol_b, vol_a, bvec_b, bvec_a`
    -- em vez de ter duas cabecas de saida assimetricas (mais simples de
    implementar/verificar). `bidirectional_flow` abaixo empacota as duas
    direcoes num UNICO forward em batch `2B` (em vez de duas chamadas
    sequenciais de batch `B`) -- mesmo truque de empilhar no eixo de
    batch ja usado pelo feixe `sh_q_out` em
    scripts/04b_train_rrin.py:_sh_bundle_forward (que tambem reaproveita
    UM modelo so' pra varias "instancias" do mesmo item, batched em vez de
    um loop Python)."""

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, norm_type: str = "instance"):
        super().__init__()
        self.max_disp = max_disp
        self.norm_type = norm_type
        in_ch = 1 + 1 + 3 + 3  # vol_a, vol_b, bvec_a, bvec_b (SEM bvec_t/t -- ver docstring)
        self.enc1 = _conv3d(in_ch, base_ch, norm_type=norm_type)
        self.enc2 = _conv3d(base_ch, base_ch * 2, stride=2, norm_type=norm_type)
        self.enc3 = _conv3d(base_ch * 2, base_ch * 4, stride=2, norm_type=norm_type)
        self.dec2 = _conv3d(base_ch * 4, base_ch * 2, norm_type=norm_type)
        self.dec1 = _conv3d(base_ch * 2 + base_ch * 2, base_ch, norm_type=norm_type)
        self.head = _conv3d(base_ch + base_ch, base_ch, norm_type=norm_type)
        self.out = nn.Conv3d(base_ch, 3, kernel_size=3, padding=1)  # so o fluxo (sem vis -- nao ha t aqui)
        # mesma inicializacao "morna" de FlowNet3D (ver comentario la, e
        # protocolo/sugestao original de melhoria do RRIN): comeca prevendo
        # fluxo ZERO (warp3d(vol,0) == identidade) em vez de deslocamento
        # aleatorio grande -- so a ultima camada e zerada, nao trava
        # gradiente.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b):
        spatial = vol_a.shape[-3:]
        bvec_a_map = _repeat_vec_3d(bvec_a, spatial)
        bvec_b_map = _repeat_vec_3d(bvec_b, spatial)
        x = torch.cat([vol_a, vol_b, bvec_a_map, bvec_b_map], dim=1)

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
        return torch.tanh(raw) * self.max_disp


def bidirectional_flow(flow_net: PairFlowNet3D, vol_a, vol_b, bvec_a, bvec_b):
    """Devolve `(flow_ab, flow_ba)` -- fluxo A->B e B->A, mesmo par de
    entrada.

    OTIMIZACAO (2026-09-01, pedido explicito da usuaria ao perguntar sobre
    acelerar o treino -- ver addendum secao 20.15): em vez de DUAS
    chamadas sequenciais de `flow_net` com batch B cada (a->b, depois
    b->a, cada uma um lancamento de kernel CUDA separado -- a versao
    original desta funcao), empilha os dois sentidos num UNICO forward de
    batch `2B` (`torch.cat` no eixo 0): as primeiras B linhas do resultado
    sao exatamente `flow_net(vol_a, vol_b, bvec_a, bvec_b)` (== flow_ab), as
    ultimas B sao `flow_net(vol_b, vol_a, bvec_b, bvec_a)` (== flow_ba) --
    MESMO resultado numerico de antes com `norm_type="instance"` (default:
    `InstanceNorm3d` normaliza cada amostra do batch de forma
    INDEPENDENTE, sobre a extensao espacial -- empilhar no eixo de batch
    nao faz nenhuma amostra "ver" estatisticas de outra), so' com metade
    dos lancamentos de kernel pro mesmo trabalho (mesmo espirito de
    `_sh_bundle_forward` em scripts/04b_train_rrin.py: empilhar no eixo de
    batch em vez de fazer um loop Python de forwards separados).

    RESSALVA (`norm_type="batch"`, NAO-default): `BatchNorm3d` calcula
    estatisticas SOBRE O BATCH INTEIRO -- aqui, o batch efetivo passa a
    ser as `2B` amostras (A e B misturadas) em vez de duas passadas
    independentes de tamanho B. Isso muda (levemente, mais amostras por
    estatistica) o comportamento de `norm_type="batch"` em relacao a uma
    hipotetica versao anterior com duas chamadas separadas -- mas como
    nenhum checkpoint de `PairFlowNet3D` foi treinado antes desta mudanca
    (a rede e' nova nesta sessao), nao ha problema de compatibilidade
    retroativa a resolver aqui, so' documentando a diferenca pra quem for
    comparar `norm_type=batch` mais tarde."""
    b = vol_a.shape[0]
    vol_first = torch.cat([vol_a, vol_b], dim=0)     # (2B, ...): [a; b]
    vol_second = torch.cat([vol_b, vol_a], dim=0)    # (2B, ...): [b; a]
    bvec_first = torch.cat([bvec_a, bvec_b], dim=0)
    bvec_second = torch.cat([bvec_b, bvec_a], dim=0)
    flow = flow_net(vol_first, vol_second, bvec_first, bvec_second)
    flow_ab, flow_ba = flow[:b], flow[b:]
    return flow_ab, flow_ba


def pairflow_ssl_losses(vol_a, vol_b, flow_ab, flow_ba, mask=None,
                         consistency_weight: float = 0.1, smooth_weight: float = 0.0) -> dict:
    """Losses auto-supervisionadas do pre-treino (Etapa 1) -- NENHUMA delas
    usa um terceiro ponto/alvo, so os proprios `vol_a`/`vol_b` (ver
    docstring do modulo).

    - `recon_ab`/`recon_ba`: MAE (mascarada, se `mask` for passada) entre o
      volume reconstruido por warp e o volume REAL do outro lado --
      `warp3d(vol_a, flow_ab)` deve reconstruir `vol_b`, e vice-versa.
      Mesma MAE usada em todo o resto do pipeline (RCAE/RRIN/implicito).
    - `consistency` (peso default 0.1 -- ideia de UnFlow/ARFlow, discutida
      com a usuaria): desfazer `flow_ab` com `flow_ba` deveria aproximar o
      campo nulo em regioes sem oclusao real. Formalmente, pra um ponto p
      no grid de b, `flow_ab(p)` aponta pra dentro do grid de a em
      `q = p + flow_ab(p)`; se os dois fluxos forem consistentes,
      `flow_ba` avaliado NESSA MESMA posicao q deveria apontar de volta
      pra p, ou seja `flow_ab(p) + flow_ba(q) ~= 0`. `flow_ba(q)` e'
      obtido reamostrando o CAMPO `flow_ba` (tratado como um "volume" de 3
      canais) com `warp3d(flow_ba, flow_ab)`. Onde essa consistencia falha
      muito e' um sinal barato (sem CSD nenhum) de que o fluxo ali nao e'
      confiavel -- candidato natural a proxy de oclusao/estrutura
      complexa levantado na conversa com a usuaria; usado aqui so como
      termo de loss, ainda NAO explorado como feature/mapa de incerteza
      separado (ideia registrada, nao implementada).
    - `smooth` (peso default 0.0 = DESLIGADO DE PROPOSITO): variacao total
      (TV) dos dois campos de fluxo. Fica desligado por padrao porque
      suavidade e' exatamente o vies que pode apagar estrutura real de
      cruzamento -- o sinal de difusao NAO e' suave perto de cruzamento, e
      durante este pre-treino nao ha NENHUM terceiro ponto real no meio do
      arco pra corrigir isso (ver docstring do modulo e discussao da
      usuaria sobre o risco de reproduzir o mesmo vies de super-suavizacao
      ja diagnosticado no RCAE, secoes 20.13/20.14). So ligar com peso
      pequeno e de forma consciente -- idealmente gateado por alguma nocao
      de "provavel cruzamento" (nao implementado aqui; a propria
      `consistency` acima e' a candidata mais natural a esse gate, mas
      isso ainda nao foi testado).

    `mask` (opcional, `(B,1,D,H,W)` ou `None`): mesma mascara de cerebro
    usada no resto do pipeline, aplicada nas duas losses de reconstrucao
    (nao faz sentido cobrar reconstrucao fora do cerebro).

    Retorna dict com as componentes e `"total"` (soma ponderada) -- mesma
    convencao de retorno em dict usada por utils/sh_angular_loss.py."""

    def _masked_mae(pred, target):
        diff = (pred - target).abs()
        if mask is not None:
            denom = mask.sum().clamp(min=1.0)
            return (diff * mask).sum() / denom
        return diff.mean()

    recon_ab = warp3d(vol_a, flow_ab)
    recon_ba = warp3d(vol_b, flow_ba)
    loss_recon_ab = _masked_mae(recon_ab, vol_b)
    loss_recon_ba = _masked_mae(recon_ba, vol_a)

    flow_ba_at_q = warp3d(flow_ba, flow_ab)  # flow_ba reamostrado na posicao pra onde flow_ab aponta
    loss_consistency = (flow_ab + flow_ba_at_q).abs().mean()

    total = loss_recon_ab + loss_recon_ba + consistency_weight * loss_consistency

    loss_smooth = torch.zeros((), device=vol_a.device, dtype=vol_a.dtype)
    if smooth_weight > 0:
        def _tv(flow):
            dx = (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).abs().mean()
            dy = (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).abs().mean()
            dz = (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).abs().mean()
            return dx + dy + dz
        loss_smooth = _tv(flow_ab) + _tv(flow_ba)
        total = total + smooth_weight * loss_smooth

    return {
        "total": total,
        "recon_ab": loss_recon_ab,
        "recon_ba": loss_recon_ba,
        "consistency": loss_consistency,
        "smooth": loss_smooth,
    }


def extrapolate_flow_to_t(flow_ab: torch.Tensor, flow_ba: torch.Tensor, t: torch.Tensor):
    """Suposicao de velocidade angular CONSTANTE ao longo do arco a->b (ver
    docstring do modulo, Deep Voxel Flow/Super SloMo): dado o fluxo
    bidirecional COMPLETO entre os dois extremos, aproxima o fluxo parcial
    até um `t` em [0,1] (0=coincide com a, 1=coincide com b) por escala
    linear -- `flow_a_to_t = t * flow_ab`, `flow_b_to_t = (1-t) * flow_ba`.

    Propriedade de contorno (verificavel sem GPU, so por inspecao): em
    t=0, `flow_a_to_t=0` -> `warp3d(vol_a, 0)` e' a identidade -> a
    predicao final de `PairFlowInterp3D` fica EXATAMENTE `vol_a` (a menos
    do residuo do RefineNet3D); simetricamente em t=1 fica `vol_b`. Isso
    da' uma checagem de sanidade barata pro `_smoke_test()` abaixo: nao
    prova que o fluxo aprendido faz sentido fisicamente, mas prova que a
    extrapolacao nao tem erro de sinal/indexacao nos extremos, onde a
    resposta certa e' conhecida por construcao."""
    t_map = t.view(-1, 1, 1, 1, 1)
    flow_a_to_t = t_map * flow_ab
    flow_b_to_t = (1.0 - t_map) * flow_ba
    return flow_a_to_t, flow_b_to_t


class PairFlowInterp3D(nn.Module):
    """Etapa 2 (supervisionada, ver docstring do modulo): usa o
    `PairFlowNet3D` (pre-treinavel na Etapa 1, auto-supervisionado) pra
    obter fluxo bidirecional entre o par de entrada, extrapola linearmente
    pra um t arbitrario via `extrapolate_flow_to_t`, faz blend +
    `RefineNet3D` (mesmo bloco de refino residual do RRIN3D, reaproveitado
    sem nenhuma modificacao).

    MESMA assinatura de `forward` de `model.rrin3d.RRIN3D`
    `(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None)` de proposito
    -- permite reaproveitar os scripts de treino/reconstrucao existentes
    (utils/rrin_dataset.py:RRINTripletDataset ja entrega exatamente esses
    campos) com o minimo de adaptacao. `bvec_t`/`quality` NAO sao usados
    aqui (o fluxo em si nao depende do alvo, so a extrapolacao por `t`
    depende) -- mantidos no forward so por compatibilidade de assinatura,
    descartados explicitamente logo no inicio.

    `freeze_flow` (default False): quando True, congela `self.flow_net`
    (nao acumula gradiente, so `RefineNet3D` e' treinado) -- util pra medir
    quanto do ganho da Etapa 2 vem SO do blend/refino aprendendo a
    compensar um fluxo fixo (pre-treinado na Etapa 1) versus deixar o
    proprio fluxo se ajustar tambem aos alvos reais (freeze_flow=False,
    'fine-tuning' de verdade)."""

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, norm_type: str = "instance",
                 freeze_flow: bool = False):
        super().__init__()
        self.flow_net = PairFlowNet3D(base_ch=base_ch, max_disp=max_disp, norm_type=norm_type)
        self.refine_net = RefineNet3D(base_ch=base_ch, norm_type=norm_type)
        self.freeze_flow = freeze_flow
        if freeze_flow:
            for p in self.flow_net.parameters():
                p.requires_grad_(False)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None):
        del bvec_t, quality  # ver docstring -- mantidos so por compatibilidade de assinatura

        if self.freeze_flow:
            with torch.no_grad():
                flow_ab, flow_ba = bidirectional_flow(self.flow_net, vol_a, vol_b, bvec_a, bvec_b)
        else:
            flow_ab, flow_ba = bidirectional_flow(self.flow_net, vol_a, vol_b, bvec_a, bvec_b)

        flow_a_to_t, flow_b_to_t = extrapolate_flow_to_t(flow_ab, flow_ba, t)
        warped_a = warp3d(vol_a, flow_a_to_t)
        warped_b = warp3d(vol_b, flow_b_to_t)

        t_map = t.view(-1, 1, 1, 1, 1)
        # blend linear simples por t (SEM mapa de visibilidade aprendido --
        # ao contrario de RRIN3D.forward, que tem um `vis_logit` produzido
        # junto com o fluxo condicionado em t; aqui o fluxo nao conhece t,
        # entao nao ha como prever visibilidade condicionada ao alvo do
        # mesmo jeito. O RefineNet3D abaixo ve blend+vol_a+vol_b crus e
        # pode aprender a compensar isso).
        blend = (1.0 - t_map) * warped_a + t_map * warped_b

        residual = self.refine_net(blend, vol_a, vol_b)
        return blend + residual


def build_pairflow_ssl_model(base_ch: int = 16, max_disp: float = 0.5,
                              norm_type: str = "instance") -> PairFlowNet3D:
    """Factory da Etapa 1 (so o fluxo bidirecional, sem blend/refino) --
    mesmo espirito de `build_rrin_model`/`build_star_model`/
    `build_implicit_model`."""
    return PairFlowNet3D(base_ch=base_ch, max_disp=max_disp, norm_type=norm_type)


def build_pairflow_interp_model(base_ch: int = 16, max_disp: float = 0.5,
                                 norm_type: str = "instance",
                                 freeze_flow: bool = False) -> PairFlowInterp3D:
    """Factory da Etapa 2 (fluxo + extrapolacao + blend/refino)."""
    return PairFlowInterp3D(base_ch=base_ch, max_disp=max_disp, norm_type=norm_type,
                             freeze_flow=freeze_flow)


def _smoke_test():
    """Roda no cluster (`python -m model.pairflow_ssl`), nao neste ambiente
    de desenvolvimento (sem PyTorch/GPU aqui). Cobre:
    - shapes de `PairFlowNet3D`/`bidirectional_flow` (flow_ab/flow_ba com o
      shape espacial certo, 3 canais);
    - `pairflow_ssl_losses` roda sem erro e produz um dict com as 5 chaves
      esperadas, `total` finito;
    - a propriedade de CONTORNO de `extrapolate_flow_to_t` descrita no seu
      docstring: com uma `PairFlowInterp3D` recem-inicializada (pesos
      "mornos", `RefineNet3D` tambem com saida perto de zero por
      inicializacao padrao pequena) rodando em t=0 e t=1, a predicao deve
      ficar proxima de vol_a/vol_b respectivamente -- checagem de
      sinal/indexacao da extrapolacao, nao de qualidade do fluxo aprendido;
    - `freeze_flow=True` de fato zera o grad de `flow_net` apos um
      `backward()` (confirma que o `torch.no_grad()` no forward esta
      funcionando, nao so documentado)."""
    torch.manual_seed(0)
    b, ps = 2, 10
    vol_a = torch.rand(b, 1, ps, ps, ps)
    vol_b = torch.rand(b, 1, ps, ps, ps)
    bvec_a = torch.randn(b, 3); bvec_a = bvec_a / bvec_a.norm(dim=-1, keepdim=True)
    bvec_b = torch.randn(b, 3); bvec_b = bvec_b / bvec_b.norm(dim=-1, keepdim=True)

    flow_net = build_pairflow_ssl_model()
    flow_ab, flow_ba = bidirectional_flow(flow_net, vol_a, vol_b, bvec_a, bvec_b)
    assert flow_ab.shape == (b, 3, ps, ps, ps)
    assert flow_ba.shape == (b, 3, ps, ps, ps)
    print("bidirectional_flow OK", flow_ab.shape)

    losses = pairflow_ssl_losses(vol_a, vol_b, flow_ab, flow_ba)
    for key in ("total", "recon_ab", "recon_ba", "consistency", "smooth"):
        assert key in losses
    assert torch.isfinite(losses["total"])
    print("pairflow_ssl_losses OK", {k: float(v) for k, v in losses.items()})

    model = build_pairflow_interp_model()
    bvec_t = torch.randn(b, 3); bvec_t = bvec_t / bvec_t.norm(dim=-1, keepdim=True)
    t0 = torch.zeros(b)
    t1 = torch.ones(b)
    pred_t0 = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t0)
    pred_t1 = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t1)
    # inicializacao morna (flow=0, refine_net com saida pequena por init
    # padrao) -- nao exatamente igual (RefineNet3D nao e' zero-init), so
    # "proximo" -- tolerancia generosa so pra pegar erro grosseiro de
    # sinal/indexacao, nao pra validar qualidade.
    assert torch.allclose(pred_t0, vol_a, atol=0.5), "t=0 deveria ficar perto de vol_a"
    assert torch.allclose(pred_t1, vol_b, atol=0.5), "t=1 deveria ficar perto de vol_b"
    print("extrapolate_flow_to_t (contorno t=0/t=1) OK")

    model_frozen = build_pairflow_interp_model(freeze_flow=True)
    t_mid = torch.full((b,), 0.5)
    pred_mid = model_frozen(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_mid)
    loss = pred_mid.abs().mean()
    loss.backward()
    for p in model_frozen.flow_net.parameters():
        assert p.grad is None or torch.all(p.grad == 0), "freeze_flow deveria zerar o grad do flow_net"
    print("freeze_flow OK (flow_net sem gradiente)")

    print("Todos os smoke tests de model/pairflow_ssl.py passaram.")


if __name__ == "__main__":
    _smoke_test()