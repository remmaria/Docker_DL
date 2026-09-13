"""
"Ensemble em estrela" para a linha PairFlow (ver model/pairflow_ssl.py e
addendum 2026-08-27/2026-09-01, secao 20.15) -- pedido explicito da usuaria
em 2026-09-03 ("quero o pairflow ensemble"), analogo direto de
model/rrin3d_star.py:RRIN3DStar mas para a linha PairFlow em vez da linha
RRIN3D.

IDEIA CENTRAL (identica a RRIN3DStar, so' trocando o "motor" de fluxo):
`PairFlowInterp3D` (model/pairflow_ssl.py) preve uma direcao-alvo a partir
de UM UNICO par (a,b), extrapolando o fluxo bidirecional auto-supervisionado
linearmente ate o `t` do alvo. Mas o MESMO alvo normalmente tem VARIOS pares
candidatos "aceitaveis" (ver utils/gradients.py:find_star_ensemble_batch),
cada um dando uma extrapolacao LIGEIRAMENTE diferente (o proprio
find_star_ensemble_batch escolhe o de menor gap_deg pra montar o par-unico
de sempre). `PairFlowStar` recebe ate M pares DIVERSOS para o MESMO alvo,
roda o MESMO pipeline (fluxo bidirecional + extrapolacao + blend + refino,
pesos COMPARTILHADOS entre os M pares -- "siames", igual RRIN3DStar) para
cada um, e funde as M predicoes por um softmax POR VOXEL sobre um logit de
confianca aprendido por par (`PairFlowWeightHead3D`) -- MESMO mecanismo de
fusao de RRIN3DStar, so' que aqui as M "camadas" fundidas sao M extrapolacoes
de fluxo diferentes (nao M pares warpados por um FlowNet3D condicionado a
t/vis, que a linha PairFlow nem tem).

Por que compartilhar pesos entre os M pares (em vez de M subredes
independentes)? MESMA justificativa de RRIN3DStar (ver docstring la, nao
duplicada aqui): o numero de pares candidatos varia por alvo/sujeito, e cada
par (a,b) e' uma amostra intercambiavel do mesmo problema -- nao ha ordem
privilegiada entre as M posicoes do feixe.

ARQUITETURALMENTE INDEPENDENTE da linha RRIN/RRIN-star (mesma convencao ja
adotada por model/pairflow_ssl.py em relacao a model/rrin3d.py, ver docstring
do modulo la): reaproveita SO os blocos genericos (`PairFlowNet3D`,
`bidirectional_flow`, `extrapolate_flow_to_t` de model/pairflow_ssl.py;
`RefineNet3D`/`warp3d`/`_conv3d`/`_repeat_vec_3d` de model/rrin3d.py), mas
NAO importa `PairWeightHead3D`/`RRIN3DStar` -- `PairFlowWeightHead3D` abaixo
e' uma copia deliberada (mesmo espirito de RefineNet3D ser reaproveitado SEM
mudanca, mas PairWeightHead3D ser pequena o bastante, e especifica o bastante
da linha RRIN-star, que faz mais sentido duplicar do que criar uma
dependencia cruzada entre as duas linhas de ensemble).

Fonte dos M pares: MESMO mecanismo de RRIN3DStar --
utils/gradients.py:find_star_ensemble_batch (offline, etapa 2b,
scripts/02b_build_rrin_triplets.py --ensemble-m) via
utils/rrin_dataset.py:RRINTripletDataset.ensemble_m (`vol_a_ens`/`vol_b_ens`/
`bvec_a_ens`/`bvec_b_ens`/`bvec_t_ens`/`t_frac_ens`/`ensemble_mask`/
`quality_ens`) -- MESMO dataset da linha RRIN-star, nenhum dataset novo
precisou ser escrito.

`freeze_flow` (default False, ver PairFlowInterp3D): congela `self.flow_net`
durante o treino do ensemble -- util pra medir quanto do ganho vem so' do
blend/refino/fusao aprendendo a compensar um fluxo pre-treinado FIXO
(tipicamente carregado de um checkpoint da Etapa 1,
scripts/04g_train_pairflow_ssl.py, via --init-checkpoint no script de
treino) versus deixar o proprio fluxo se re-ajustar tambem.

`weight_quality_cond` (default False, ADITIVO -- mesmo espirito do
`weight_quality_cond` de RRIN3DStar): alimenta `residual_deg`/`gap_deg` de
cada par DIRETO na `PairFlowWeightHead3D`, em vez de deixar a cabeca de fusao
inferir confiabilidade so' pelo conteudo de imagem. NAO ha um
`use_quality_cond` equivalente aqui a nivel de fluxo (`PairFlowNet3D` nao tem
esse parametro -- ver model/pairflow_ssl.py, o fluxo bidirecional nunca viu
nenhum sinal de qualidade, so' os dois volumes crus), entao (ao contrario de
RRIN3DStar) so' existe UMA flag de qualidade aqui, nao duas desacopladas.

Requer PyTorch (nao disponivel neste ambiente de desenvolvimento -- revisado
manualmente, testado apenas por compilacao de sintaxe; validar no cluster com
`python -m model.pairflow_star`, smoke test no fim do arquivo, mesmo padrao
de model/rrin3d_star.py/model/pairflow_ssl.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .rrin3d import RefineNet3D, _conv3d, _repeat_vec_3d, warp3d
from .pairflow_ssl import PairFlowNet3D, bidirectional_flow, extrapolate_flow_to_t


class PairFlowWeightHead3D(nn.Module):
    """Copia deliberada de model/rrin3d_star.py:PairWeightHead3D (ver
    docstring do modulo acima para a justificativa de NAO importar de la) --
    mesma arquitetura, mesmo proposito (logit de confianca POR VOXEL para UMA
    predicao candidata do feixe, saida SEM sigmoid/softmax -- a normalizacao
    entre os M pares e' feita em PairFlowStar.forward, que precisa ver todos
    os M logits e a mascara de padding ao mesmo tempo).

    `use_quality_cond` (default False): concatena `quality`
    (residual_deg/90, gap_deg/90 -- mesma convencao de RRIN3DStar) como 2
    canais constantes extras via `_repeat_vec_3d`."""

    def __init__(self, base_ch: int = 16, norm_type: str = "instance",
                 use_quality_cond: bool = False):
        super().__init__()
        self.use_quality_cond = use_quality_cond
        in_ch = 1 + 1 + 1  # blend, vol_a, vol_b
        if use_quality_cond:
            in_ch += 2
        self.net = nn.Sequential(
            _conv3d(in_ch, base_ch, norm_type=norm_type),
            nn.Conv3d(base_ch, 1, kernel_size=3, padding=1),
        )
        # zero-init -- ponto de partida neutro (pesos iguais entre os M
        # pares ate a rede aprender a diferenciar), mesmo espirito de
        # PairWeightHead3D/FlowNet3D. So afeta treinos NOVOS (sem resume).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, blend, vol_a, vol_b, quality=None):
        parts = [blend, vol_a, vol_b]
        if self.use_quality_cond:
            if quality is None:
                raise ValueError(
                    "PairFlowWeightHead3D.use_quality_cond=True mas `quality` nao foi passado "
                    "ao forward -- ver PairFlowStar.weight_quality_cond/docstring do modulo.")
            spatial = blend.shape[-3:]
            parts.append(_repeat_vec_3d(quality, spatial))
        x = torch.cat(parts, dim=1)
        return self.net(x)  # (B,1,D,H,W), logit cru


class PairFlowStar(nn.Module):
    """Ensemble em estrela da linha PairFlow: funde ate M predicoes
    `PairFlowInterp3D`-equivalentes (fluxo bidirecional COMPARTILHADO,
    extrapolado por t, blend + refino) de M pares candidatos DIFERENTES para
    o MESMO alvo, via softmax POR VOXEL sobre um logit de confianca
    (`PairFlowWeightHead3D`). Ver docstring do modulo para a motivacao
    completa e a analogia/diferenca com RRIN3DStar.

    Sem hiperparametro `M` fixo na arquitetura -- lido do shape de
    `ensemble_mask` em tempo de execucao (mesmo espirito de RRIN3DStar):
    o mesmo checkpoint funciona pra qualquer M>=1 na reconstrucao.

    Uso:
        model = PairFlowStar()
        pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                      quality=quality)
    vol_a, vol_b: (B, M, 1, D, H, W) -- M pares candidatos de entrada.
    bvec_a, bvec_b: (B, M, 3).
    bvec_t: (B, M, 3) -- MANTIDO na assinatura por compatibilidade com
        RRIN3DStar/o batch dict de RRINTripletDataset, mas NAO USADO (o
        fluxo da linha PairFlow nao conhece o alvo -- ver
        model/pairflow_ssl.py:PairFlowInterp3D.forward, mesmo `del` logo no
        inicio).
    t: (B, M) -- t_frac de cada par (varia entre posicoes, MESMO alvo).
    ensemble_mask: (B, M) bool -- True = par real nesta posicao do feixe.
    quality: (B, M, 2) ou None -- usado so' se weight_quality_cond=True.
    retorna: (B, 1, D, H, W) -- direcao-alvo predita (fusao dos <=M pares).
    """

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, norm_type: str = "instance",
                 freeze_flow: bool = False, weight_quality_cond: bool = False):
        super().__init__()
        self.norm_type = norm_type
        self.freeze_flow = freeze_flow
        self.weight_quality_cond = weight_quality_cond
        self.flow_net = PairFlowNet3D(base_ch=base_ch, max_disp=max_disp, norm_type=norm_type)
        self.refine_net = RefineNet3D(base_ch=base_ch, norm_type=norm_type)
        self.weight_head = PairFlowWeightHead3D(base_ch=base_ch, norm_type=norm_type,
                                                 use_quality_cond=weight_quality_cond)
        if freeze_flow:
            for p in self.flow_net.parameters():
                p.requires_grad_(False)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=None,
                return_pairs=False):
        del bvec_t  # ver docstring -- linha PairFlow nao condiciona o fluxo pelo alvo

        b, m = vol_a.shape[0], vol_a.shape[1]

        def _flat(x):
            return x.reshape(b * m, *x.shape[2:])

        vol_a_f = _flat(vol_a)
        vol_b_f = _flat(vol_b)
        bvec_a_f = _flat(bvec_a)
        bvec_b_f = _flat(bvec_b)
        t_f = t.reshape(b * m)
        quality_f = _flat(quality) if quality is not None else None

        # pipeline PairFlowInterp3D-equivalente compartilhado, rodado UMA VEZ
        # para as B*M "amostras" -- mesmo truque de achatamento em batch de
        # RRIN3DStar.forward.
        if self.freeze_flow:
            with torch.no_grad():
                flow_ab, flow_ba = bidirectional_flow(self.flow_net, vol_a_f, vol_b_f,
                                                        bvec_a_f, bvec_b_f)
        else:
            flow_ab, flow_ba = bidirectional_flow(self.flow_net, vol_a_f, vol_b_f,
                                                    bvec_a_f, bvec_b_f)

        flow_a_to_t, flow_b_to_t = extrapolate_flow_to_t(flow_ab, flow_ba, t_f)
        warped_a = warp3d(vol_a_f, flow_a_to_t)
        warped_b = warp3d(vol_b_f, flow_b_to_t)

        t_map = t_f.view(-1, 1, 1, 1, 1)
        blend_f = (1.0 - t_map) * warped_a + t_map * warped_b  # sem mapa de visibilidade (ver
                                                                 # PairFlowInterp3D.forward)
        residual_f = self.refine_net(blend_f, vol_a_f, vol_b_f)
        pred_f = blend_f + residual_f                          # (B*M,1,D,H,W)
        weight_logit_f = self.weight_head(
            blend_f, vol_a_f, vol_b_f,
            quality=quality_f if self.weight_quality_cond else None)  # (B*M,1,D,H,W)

        def _unflat(x):
            return x.reshape(b, m, *x.shape[1:])

        pred = _unflat(pred_f)                    # (B,M,1,D,H,W)
        weight_logit = _unflat(weight_logit_f)    # (B,M,1,D,H,W)

        mask = ensemble_mask.view(b, m, 1, 1, 1, 1)
        neg_inf = torch.finfo(weight_logit.dtype).min
        weight_logit = torch.where(mask, weight_logit, torch.full_like(weight_logit, neg_inf))
        pi = torch.softmax(weight_logit, dim=1)  # (B,M,1,D,H,W), soma 1 sobre as posicoes reais

        out = (pi * pred).sum(dim=1)  # (B,1,D,H,W)
        if return_pairs:
            return out, {"pred": pred, "pi": pi}
        return out


def build_pairflow_star_model(base_ch: int = 16, max_disp: float = 0.5,
                               norm_type: str = "instance", freeze_flow: bool = False,
                               weight_quality_cond: bool = False) -> PairFlowStar:
    """Wrapper trivial (mesmo espirito de build_star_model/
    build_pairflow_interp_model) -- existe so' para
    scripts/04i_train_pairflow_star.py e scripts/05k_reconstruct_pairflow_star.py
    nao precisarem instanciar a classe diretamente."""
    return PairFlowStar(base_ch=base_ch, max_disp=max_disp, norm_type=norm_type,
                         freeze_flow=freeze_flow, weight_quality_cond=weight_quality_cond)


def _smoke_test():
    """Forward pass com tensores pequenos aleatorios -- mesmo padrao de
    model/rrin3d_star.py/model/pairflow_ssl.py. Roda no cluster:
    python -m model.pairflow_star"""
    torch.manual_seed(0)
    b, m, d, h, w = 2, 3, 10, 10, 10
    vol_a = torch.rand(b, m, 1, d, h, w)
    vol_b = torch.rand(b, m, 1, d, h, w)

    def rand_bvecs(shape):
        v = torch.randn(*shape)
        return v / v.norm(dim=-1, keepdim=True)

    bvec_a = rand_bvecs((b, m, 3))
    bvec_b = rand_bvecs((b, m, 3))
    bvec_t = rand_bvecs((b, 1, 3)).expand(b, m, 3).contiguous()  # MESMO alvo nas M posicoes
    t = torch.rand(b, m)
    ensemble_mask = torch.ones(b, m, dtype=torch.bool)
    ensemble_mask[0, -1] = False  # simula 1o item do batch com so' 2/3 pares reais
    expected = (b, 1, d, h, w)

    model = build_pairflow_star_model(base_ch=8)
    out = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask)
    assert out.shape == expected, f"shape mismatch: {out.shape} != {expected}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"smoke test OK (weight_quality_cond=False), output shape: {tuple(out.shape)}, "
          f"{n_params} parametros")

    out2, extra = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, return_pairs=True)
    assert torch.allclose(out, out2), "forward deveria ser deterministico (sem dropout/RNG)"
    pi = extra["pi"]
    assert pi.shape == (b, m, 1, d, h, w)
    pi_sum = pi.sum(dim=1)
    assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5), \
        "pesos de fusao (pi) nao somam 1 por voxel"
    masked_pi = pi[0, -1]
    assert torch.allclose(masked_pi, torch.zeros_like(masked_pi)), \
        "posicao mascarada do feixe deveria ter peso de fusao exatamente 0"
    print("OK: pi soma 1 por voxel e posicoes mascaradas tem peso exatamente 0")

    # weight_quality_cond=True
    model_wqc = build_pairflow_star_model(base_ch=8, weight_quality_cond=True)
    quality = torch.rand(b, m, 2)
    out_wqc = model_wqc(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=quality)
    assert out_wqc.shape == expected
    print(f"smoke test OK (weight_quality_cond=True), output shape: {tuple(out_wqc.shape)}")

    # weight_quality_cond=True mas esquecendo `quality` no forward deve levantar erro claro
    try:
        model_wqc(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=None)
        raise AssertionError("deveria ter levantado ValueError sem `quality`")
    except ValueError:
        print("OK: weight_quality_cond=True sem `quality` levanta ValueError, como esperado")

    # zero-init da PairFlowWeightHead3D: pi nao deveria mudar com quality num modelo NOVO
    model_wqc2 = build_pairflow_star_model(base_ch=8, weight_quality_cond=True)
    _, extra_a = model_wqc2(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                             quality=torch.zeros(b, m, 2), return_pairs=True)
    _, extra_b = model_wqc2(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                             quality=torch.ones(b, m, 2), return_pairs=True)
    assert torch.allclose(extra_a["pi"], extra_b["pi"]), \
        "com a ultima camada da PairFlowWeightHead3D zero-init, pi nao deveria mudar com quality"
    print("OK: zero-init da PairFlowWeightHead3D preserva pi uniforme mesmo com quality diferente")

    # M=1 (ensemble degenerado a um unico par)
    vol_a1, vol_b1 = vol_a[:, :1], vol_b[:, :1]
    bvec_a1, bvec_b1, bvec_t1 = bvec_a[:, :1], bvec_b[:, :1], bvec_t[:, :1]
    t1 = t[:, :1]
    mask1 = torch.ones(b, 1, dtype=torch.bool)
    out1, extra1 = model(vol_a1, vol_b1, bvec_a1, bvec_b1, bvec_t1, t1, mask1, return_pairs=True)
    assert out1.shape == expected
    assert torch.allclose(extra1["pi"], torch.ones_like(extra1["pi"]))
    print("OK: M=1 funciona (peso de fusao = 1.0, degenerado a PairFlowInterp3D de par unico)")

    # linha do batch com so' 1 posicao real no feixe
    mask_single_real = torch.zeros(b, m, dtype=torch.bool)
    mask_single_real[:, 0] = True
    out_single, extra_single = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, mask_single_real,
                                      return_pairs=True)
    assert not torch.isnan(out_single).any(), "saida nao deveria ter NaN"
    assert torch.allclose(extra_single["pi"][:, 0], torch.ones_like(extra_single["pi"][:, 0]))
    print("OK: linha com so' 1 posicao real no feixe nao gera NaN (peso todo nela)")

    # freeze_flow: confirma que o grad de flow_net fica zerado apos um backward
    model_frozen = build_pairflow_star_model(base_ch=8, freeze_flow=True)
    out_frozen = model_frozen(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask)
    loss = out_frozen.abs().mean()
    loss.backward()
    for p in model_frozen.flow_net.parameters():
        assert p.grad is None or torch.all(p.grad == 0), \
            "freeze_flow deveria zerar o grad do flow_net"
    print("OK: freeze_flow=True zera o grad do flow_net")

    # propriedade de contorno por par: com t=0 numa posicao do feixe, a
    # extrapolacao daquele par fica em flow_a_to_t=0 -> pred daquele par ~=
    # vol_a[esse par] (a menos do residuo pequeno do RefineNet3D recem-
    # inicializado) -- mesma checagem de sanidade de
    # model/pairflow_ssl.py:_smoke_test, agora POR POSICAO do feixe.
    model_boundary = build_pairflow_star_model(base_ch=8)
    t_zero = torch.zeros(b, m)
    mask_all = torch.ones(b, m, dtype=torch.bool)
    _, extra_boundary = model_boundary(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_zero, mask_all,
                                        return_pairs=True)
    pred_per_pair = extra_boundary["pred"]  # (B,M,1,D,H,W)
    assert torch.allclose(pred_per_pair, vol_a, atol=0.5), \
        "com t=0 em toda posicao do feixe, cada predicao candidata deveria ficar perto de vol_a"
    print("OK: extrapolacao por par em t=0 fica proxima de vol_a (checagem de contorno)")

    print("Todos os smoke tests de model/pairflow_star.py passaram.")


if __name__ == "__main__":
    _smoke_test()