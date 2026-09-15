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

ATUALIZACAO (2026-09-14 -- item 1, mesma mudanca de model/rrin3d_star.py,
COPIA DELIBERADA aqui, ver docstring la para a motivacao completa): a
`PairFlowWeightHead3D` sempre calculou o logit de confianca de cada
candidato de forma totalmente INDEPENDENTE dos outros M-1 candidatos do
mesmo feixe. `PairFlowWeightHead3D`/`PairFlowStar` ganharam
`cross_candidate_attention`/`cross_candidate_attn_heads` PROPRIOS, ADITIVOS
(default desligado) -- insere um bloco de self-attention COMPLETA
(`CrossCandidateAttention3D`, copia deliberada da classe homonima em
model/rrin3d_star.py) entre os M candidatos, POR VOXEL, entre a projecao
inicial e o `out_conv` que produz o logit. Mesma mascara de padding via
`key_padding_mask`, mesma preservacao de pi uniforme por zero-init do
out_conv.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .rrin3d import RefineNet3D, _conv3d, _repeat_vec_3d, warp3d
from .pairflow_ssl import PairFlowNet3D, bidirectional_flow, extrapolate_flow_to_t


# MESMA convencao de model/rrin3d_star.py:REFINE_COND_CH (copia deliberada,
# ver docstring do modulo sobre independencia arquitetural entre as linhas):
# bvec_a (3) + bvec_b (3) + bvec_t (3) + t_frac (1) + quality (2) = 12.
#
# NOTA sobre `bvec_t` nesta linha: o FLUXO do PairFlow e' deliberadamente
# cego ao alvo (ver docstring de PairFlowStar.forward e model/pairflow_ssl.py
# -- e o que permite pre-treina-lo de forma auto-supervisionada, sem alvo).
# Isso vale para o FLUXO, nao para o refino: a RefineNet3D ja e' treinada de
# forma supervisionada contra o alvo de qualquer jeito, entao deixa-la saber
# QUAL direcao esta sendo predita nao cria nenhum vazamento novo nem
# invalida o pre-treino auto-supervisionado de `flow_net` (que continua
# recebendo exatamente os mesmos tensores de antes).
REFINE_COND_CH = 12


def _refine_cond_vector(bvec_a, bvec_b, bvec_t, t, quality):
    """Copia deliberada de model/rrin3d_star.py:_refine_cond_vector (mesma
    ordem de canais, mesma semantica). Ver docstring la."""
    parts = [bvec_a, bvec_b, bvec_t, t.view(-1, 1)]
    if quality is not None:
        parts.append(quality)
    else:
        parts.append(torch.zeros(bvec_a.shape[0], 2, dtype=bvec_a.dtype, device=bvec_a.device))
    return torch.cat(parts, dim=1)


class CrossCandidateAttention3D(nn.Module):
    """Copia deliberada de model/rrin3d_star.py:CrossCandidateAttention3D
    (ver docstring do modulo acima para a justificativa de NAO importar de
    la) -- mesma arquitetura, mesmo proposito: self-attention COMPLETA entre
    os M candidatos do ensemble em estrela, aplicada POR VOXEL, ANTES do
    logit de confianca (ver PairFlowWeightHead3D.forward). Mascara posicoes
    de PADDING (`ensemble_mask=False`) via `key_padding_mask` de
    `nn.MultiheadAttention`, PERMUTATION-EQUIVARIANTE por construcao (sem
    codificacao de posicao de sequencia)."""

    def __init__(self, channels: int, num_heads: int = 4, ff_mult: int = 2):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) precisa ser divisivel por num_heads ({num_heads}) -- "
                f"ver --base-ch/--cross-candidate-attn-heads em scripts/04i_train_pairflow_star.py.")
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads,
                                           batch_first=True)
        self.ln1 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.ReLU(inplace=True),
            nn.Linear(channels * ff_mult, channels),
        )
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, feat: torch.Tensor, ensemble_mask=None) -> torch.Tensor:
        """feat: (B, M, C, D, H, W). ensemble_mask: (B, M) bool ou None.
        Retorna o MESMO shape."""
        b, m, c, d, h, w = feat.shape
        x = feat.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, m, c)
        key_padding_mask = None
        if ensemble_mask is not None:
            pad_mask = (~ensemble_mask).unsqueeze(1).expand(b, d * h * w, m)
            key_padding_mask = pad_mask.reshape(b * d * h * w, m)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x.reshape(b, d, h, w, m, c).permute(0, 4, 5, 1, 2, 3)


class PairFlowWeightHead3D(nn.Module):
    """Copia deliberada de model/rrin3d_star.py:PairWeightHead3D (ver
    docstring do modulo acima para a justificativa de NAO importar de la) --
    mesma arquitetura, mesmo proposito (logit de confianca POR VOXEL para UMA
    predicao candidata do feixe, saida SEM sigmoid/softmax -- a normalizacao
    entre os M pares e' feita em PairFlowStar.forward, que precisa ver todos
    os M logits e a mascara de padding ao mesmo tempo).

    `use_quality_cond` (default False): concatena `quality`
    (residual_deg/90, gap_deg/90 -- mesma convencao de RRIN3DStar) como 2
    canais constantes extras via `_repeat_vec_3d`.

    `cross_candidate_attention`/`cross_candidate_attn_heads` (default
    False/4, ADITIVO -- item 1, copia deliberada do mecanismo homonimo de
    model/rrin3d_star.py:PairWeightHead3D, ver docstring la): insere
    self-attention entre os M candidatos entre a projecao inicial e o
    `out_conv`. MUDANCA DE CONVENCAO (2026-09-14): `forward` agora recebe o
    feixe INTEIRO (B,M,...) de uma vez, em vez de um candidato achatado em
    B*M por chamada -- PairFlowStar.forward foi atualizado de acordo."""

    def __init__(self, base_ch: int = 16, norm_type: str = "instance",
                 use_quality_cond: bool = False,
                 cross_candidate_attention: bool = False,
                 cross_candidate_attn_heads: int = 4):
        super().__init__()
        self.use_quality_cond = use_quality_cond
        self.cross_candidate_attention = cross_candidate_attention
        in_ch = 1 + 1 + 1  # blend, vol_a, vol_b
        if use_quality_cond:
            in_ch += 2
        # ModuleList preserva as chaves `net.0.*`/`net.1.*` de um Sequential
        # equivalente -- ver "REGRESSAO DE 2026-09-14" na docstring de
        # model/rrin3d_star.py:PairWeightHead3D (a mesma quebra aconteceu
        # nas duas linhas, pelo mesmo motivo, e foi corrigida junto).
        self.net = nn.ModuleList([
            _conv3d(in_ch, base_ch, norm_type=norm_type),
            nn.Conv3d(base_ch, 1, kernel_size=3, padding=1),
        ])
        self.cross_attn = None
        if cross_candidate_attention:
            self.cross_attn = CrossCandidateAttention3D(base_ch, num_heads=cross_candidate_attn_heads)
        # zero-init -- ponto de partida neutro (pesos iguais entre os M
        # pares ate a rede aprender a diferenciar), mesmo espirito de
        # PairWeightHead3D/FlowNet3D. So afeta treinos NOVOS (sem resume).
        # Continua valendo com cross_candidate_attention=True.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Mesma compatibilidade de model/rrin3d_star.py:PairWeightHead3D --
        remapeia `proj.*`/`out_conv.*` (gravados pela versao intermediaria
        de 2026-09-14) para `net.0.*`/`net.1.*`."""
        for old, new in (("proj.", "net.0."), ("out_conv.", "net.1.")):
            stale = [k for k in state_dict if k.startswith(prefix + old)]
            for k in stale:
                state_dict[prefix + new + k[len(prefix + old):]] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, blend, vol_a, vol_b, quality=None, ensemble_mask=None):
        """blend, vol_a, vol_b: (B,M,1,D,H,W) -- feixe INTEIRO (ver
        "MUDANCA DE CONVENCAO" na docstring da classe). quality: (B,M,2) ou
        None. ensemble_mask: (B,M) bool ou None (usado so se
        cross_candidate_attention=True). Retorna (B,M,1,D,H,W), logit cru."""
        b, m = blend.shape[0], blend.shape[1]
        parts = [blend, vol_a, vol_b]
        if self.use_quality_cond:
            if quality is None:
                raise ValueError(
                    "PairFlowWeightHead3D.use_quality_cond=True mas `quality` nao foi passado "
                    "ao forward -- ver PairFlowStar.weight_quality_cond/docstring do modulo.")
            spatial = blend.shape[-3:]
            quality_flat = quality.reshape(b * m, quality.shape[-1])
            quality_map = _repeat_vec_3d(quality_flat, spatial).reshape(
                b, m, quality.shape[-1], *spatial)
            parts.append(quality_map)
        x = torch.cat(parts, dim=2)                          # (B,M,in_ch,D,H,W)
        x_flat = x.reshape(b * m, x.shape[2], *x.shape[3:])
        feat_flat = self.net[0](x_flat)
        if self.cross_attn is not None:
            feat = feat_flat.reshape(b, m, feat_flat.shape[1], *feat_flat.shape[2:])
            feat = self.cross_attn(feat, ensemble_mask=ensemble_mask)
            feat_flat = feat.reshape(b * m, feat.shape[2], *feat.shape[3:])
        logit_flat = self.net[1](feat_flat)
        return logit_flat.reshape(b, m, 1, *logit_flat.shape[2:])


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
                 freeze_flow: bool = False, weight_quality_cond: bool = False,
                 cross_candidate_attention: bool = False, cross_candidate_attn_heads: int = 4,
                 refine_base_ch: int = None, refine_depth: int = 2, refine_cond: bool = False):
        super().__init__()
        self.norm_type = norm_type
        self.freeze_flow = freeze_flow
        self.weight_quality_cond = weight_quality_cond
        self.cross_candidate_attention = cross_candidate_attention
        self.cross_candidate_attn_heads = cross_candidate_attn_heads
        self.refine_base_ch = refine_base_ch if refine_base_ch is not None else base_ch
        self.refine_depth = refine_depth
        self.refine_cond = refine_cond
        self.flow_net = PairFlowNet3D(base_ch=base_ch, max_disp=max_disp, norm_type=norm_type)
        self.refine_net = RefineNet3D(base_ch=self.refine_base_ch, norm_type=norm_type,
                                       depth=refine_depth,
                                       cond_ch=REFINE_COND_CH if refine_cond else 0)
        self.weight_head = PairFlowWeightHead3D(
            base_ch=base_ch, norm_type=norm_type, use_quality_cond=weight_quality_cond,
            cross_candidate_attention=cross_candidate_attention,
            cross_candidate_attn_heads=cross_candidate_attn_heads)
        if freeze_flow:
            for p in self.flow_net.parameters():
                p.requires_grad_(False)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=None,
                return_pairs=False):
        # `bvec_t` continua NAO sendo usado pelo fluxo (ver docstring) -- com
        # `refine_cond=True` ele passa a ser usado SO' pela RefineNet3D (ver
        # nota em REFINE_COND_CH acima sobre por que isso nao viola a
        # cegueira-ao-alvo do fluxo auto-supervisionado).
        if not self.refine_cond:
            del bvec_t

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
        refine_cond_f = None
        if self.refine_cond:
            refine_cond_f = _refine_cond_vector(bvec_a_f, bvec_b_f, _flat(bvec_t), t_f, quality_f)
        residual_f = self.refine_net(blend_f, vol_a_f, vol_b_f, cond=refine_cond_f)
        pred_f = blend_f + residual_f                          # (B*M,1,D,H,W)

        def _unflat(x):
            return x.reshape(b, m, *x.shape[1:])

        pred = _unflat(pred_f)                    # (B,M,1,D,H,W)

        # weight_head agora recebe o feixe INTEIRO (B,M,...) de uma vez (ver
        # "MUDANCA DE CONVENCAO" em PairFlowWeightHead3D.__doc__, item 1,
        # 2026-09-14) -- mesma mudanca de RRIN3DStar.forward.
        weight_logit = self.weight_head(
            _unflat(blend_f), _unflat(vol_a_f), _unflat(vol_b_f),
            quality=quality if self.weight_quality_cond else None,
            ensemble_mask=ensemble_mask)  # (B,M,1,D,H,W)

        mask = ensemble_mask.view(b, m, 1, 1, 1, 1)
        neg_inf = torch.finfo(weight_logit.dtype).min
        weight_logit = torch.where(mask, weight_logit, torch.full_like(weight_logit, neg_inf))
        pi = torch.softmax(weight_logit, dim=1)  # (B,M,1,D,H,W), soma 1 sobre as posicoes reais

        out = (pi * pred).sum(dim=1)  # (B,1,D,H,W)
        if return_pairs:
            residual = _unflat(residual_f)  # (B,M,1,D,H,W) -- ver
            # --residual-l2-weight em scripts/04i_train_pairflow_star.py (addendum
            # 2026-09-13, mesmo mecanismo de model/rrin3d_star.py): exposto aqui
            # so quando return_pairs=True, entao nenhum chamador existente (que
            # usa o default False) e afetado.
            return out, {"pred": pred, "pi": pi, "residual": residual}
        return out


def build_pairflow_star_model(base_ch: int = 16, max_disp: float = 0.5,
                               norm_type: str = "instance", freeze_flow: bool = False,
                               weight_quality_cond: bool = False,
                               cross_candidate_attention: bool = False,
                               cross_candidate_attn_heads: int = 4,
                               refine_base_ch: int = None, refine_depth: int = 2,
                               refine_cond: bool = False) -> PairFlowStar:
    """Wrapper trivial (mesmo espirito de build_star_model/
    build_pairflow_interp_model) -- existe so' para
    scripts/04i_train_pairflow_star.py e scripts/05k_reconstruct_pairflow_star.py
    nao precisarem instanciar a classe diretamente."""
    return PairFlowStar(base_ch=base_ch, max_disp=max_disp, norm_type=norm_type,
                         freeze_flow=freeze_flow, weight_quality_cond=weight_quality_cond,
                         cross_candidate_attention=cross_candidate_attention,
                         cross_candidate_attn_heads=cross_candidate_attn_heads,
                         refine_base_ch=refine_base_ch, refine_depth=refine_depth,
                         refine_cond=refine_cond)


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

    # ITEM 1 (2026-09-14): cross_candidate_attention -- mesmos testes de
    # model/rrin3d_star.py:_smoke_test (copia deliberada), adaptados pra
    # PairFlowStar/PairFlowWeightHead3D.
    model_cattn = build_pairflow_star_model(base_ch=8, cross_candidate_attention=True,
                                             cross_candidate_attn_heads=4)
    out_cattn = model_cattn(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask)
    assert out_cattn.shape == expected, \
        f"shape mismatch (cross_candidate_attention=True): {out_cattn.shape} != {expected}"
    print(f"smoke test OK (cross_candidate_attention=True), output shape: {tuple(out_cattn.shape)}")

    _, extra_cattn = model_cattn(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                                  return_pairs=True)
    pi_cattn = extra_cattn["pi"]
    real_pi = pi_cattn[0, :2]
    assert torch.allclose(real_pi, real_pi[:1].expand_as(real_pi), atol=1e-5), \
        "com net[-1] zero-init, pi deveria ficar uniforme entre candidatos reais mesmo com " \
        "cross_candidate_attention=True"
    masked_pi_cattn = pi_cattn[0, -1]
    assert torch.allclose(masked_pi_cattn, torch.zeros_like(masked_pi_cattn)), \
        "posicao mascarada do feixe deveria ter peso de fusao exatamente 0 mesmo com " \
        "cross_candidate_attention=True"
    print("OK: cross_candidate_attention=True com net[-1] zero-init preserva pi uniforme "
          "entre candidatos reais e zero na posicao mascarada")

    torch.manual_seed(1)
    model_perm = build_pairflow_star_model(base_ch=8, cross_candidate_attention=True,
                                            cross_candidate_attn_heads=4)
    nn.init.normal_(model_perm.weight_head.net[-1].weight)
    nn.init.normal_(model_perm.weight_head.net[-1].bias)
    model_perm.eval()
    perm = torch.tensor([2, 0, 1])
    out_orig, extra_orig = model_perm(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                                       return_pairs=True)
    out_perm, extra_perm = model_perm(
        vol_a[:, perm], vol_b[:, perm], bvec_a[:, perm], bvec_b[:, perm], bvec_t[:, perm],
        t[:, perm], ensemble_mask[:, perm], return_pairs=True)
    assert torch.allclose(out_orig, out_perm, atol=1e-4), \
        "a fusao final (out) nao deveria mudar so por permutar a ordem dos M candidatos de entrada"
    assert torch.allclose(extra_orig["pi"][:, perm], extra_perm["pi"], atol=1e-4), \
        "pi deveria ser equivariante a permutacao dos M candidatos (mesma ordem da entrada)"
    print("OK: cross_candidate_attention=True e equivariante a permutacao dos M candidatos "
          "(out final invariante, pi permutado na mesma ordem)")

    vol_a_pert = vol_a.clone()
    vol_a_pert[0, -1] += 100.0
    _, extra_pert = model_perm(vol_a_pert, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                                return_pairs=True)
    assert torch.allclose(extra_orig["pi"][0, :2], extra_pert["pi"][0, :2], atol=1e-4), \
        "perturbar o conteudo de uma posicao MASCARADA do feixe nao deveria mudar pi dos " \
        "candidatos reais (key_padding_mask deveria impedir essa fuga de informacao)"
    print("OK: perturbar uma posicao mascarada (padding) nao vaza pro pi dos candidatos reais")

    try:
        build_pairflow_star_model(base_ch=8, cross_candidate_attention=True,
                                   cross_candidate_attn_heads=3)
        raise AssertionError("deveria ter levantado ValueError (8 nao e divisivel por 3)")
    except ValueError:
        print("OK: cross_candidate_attn_heads que nao divide base_ch levanta ValueError cedo")


    # GUARDA DE NOMES DO state_dict (2026-09-14, apos a regressao descrita na
    # docstring de PairFlowWeightHead3D): a lista abaixo e o conjunto EXATO de
    # chaves que essa cabeca sempre teve. Se alguem reestruturar a classe e
    # renomear os pesos, TODOS os checkpoints star ja treinados param de
    # carregar -- e o erro so aparece la na frente, num script de
    # reconstrucao/diagnostico, nao aqui. Este teste transforma isso numa
    # falha imediata e obvia.
    head_default = build_pairflow_star_model(base_ch=16).weight_head
    got = sorted(k for k, _ in head_default.named_parameters())
    esperado = sorted(["net.0.0.weight", "net.0.0.bias", "net.0.1.weight", "net.0.1.bias",
                        "net.1.weight", "net.1.bias"])
    assert got == esperado, (
        f"as chaves do state_dict de PairFlowWeightHead3D mudaram!\n"
        f"  esperado: {esperado}\n  obtido:   {got}\n"
        f"Isso INVALIDA todos os checkpoints star ja treinados -- ver a nota de regressao "
        f"na docstring da classe antes de mudar qualquer coisa aqui.")
    print(f"OK: PairFlowWeightHead3D preserva as {len(got)} chaves historicas do state_dict")

    # e o remapeamento de compatibilidade da janela quebrada funciona?
    quebrado = {}
    for k, v in head_default.state_dict().items():
        novo = k.replace("net.0.", "proj.").replace("net.1.", "out_conv.")
        quebrado[novo] = v
    head_novo = build_pairflow_star_model(base_ch=16).weight_head
    head_novo.load_state_dict(quebrado)   # nao pode levantar
    for k, v in head_default.state_dict().items():
        assert torch.allclose(head_novo.state_dict()[k], v), f"remapeamento errou em {k}"
    print("OK: checkpoints da janela quebrada (proj./out_conv.) sao remapeados no load")

    # CAPACIDADE DO REFINO (2026-09-14) -- mesmos testes de
    # model/rrin3d_star.py:_smoke_test (copia deliberada), incluindo a guarda
    # de retrocompatibilidade dos 8.737 parametros historicos.
    model_default = build_pairflow_star_model(base_ch=16)
    n_refine_default = sum(p.numel() for p in model_default.refine_net.parameters())
    assert n_refine_default == 8737, \
        (f"RefineNet3D com defaults deveria manter os 8.737 parametros historicos em "
         f"base_ch=16, obtido {n_refine_default} -- isso INVALIDA checkpoints existentes")
    print(f"OK: RefineNet3D com defaults preserva os {n_refine_default} parametros historicos")

    model_refine = build_pairflow_star_model(base_ch=8, refine_base_ch=32, refine_depth=4,
                                              refine_cond=True)
    out_refine = model_refine(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                               quality=quality)
    assert out_refine.shape == expected, f"shape mismatch (refino maior): {out_refine.shape}"
    n_refine_big = sum(p.numel() for p in model_refine.refine_net.parameters())
    print(f"smoke test OK (refine_base_ch=32, refine_depth=4, refine_cond=True): "
          f"refine_net com {n_refine_big} parametros ({n_refine_big / n_refine_default:.1f}x o default)")

    out_nq = model_refine(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=None)
    assert out_nq.shape == expected
    print("OK: refine_cond=True sem `quality` roda (canais de qualidade entram zerados)")

    # o condicionamento precisa realmente chegar na saida. ATENCAO: nesta
    # linha o FLUXO e' cego ao alvo por design (`del bvec_t` quando
    # refine_cond=False), entao este teste tambem confirma que o unico
    # caminho por onde bvec_t influencia a saida e' a RefineNet3D.
    bvec_t2 = rand_bvecs((b, 1, 3)).expand(b, m, 3).contiguous()
    out_t2 = model_refine(vol_a, vol_b, bvec_a, bvec_b, bvec_t2, t, ensemble_mask,
                           quality=quality)
    assert not torch.allclose(out_refine, out_t2, atol=1e-6), \
        "com refine_cond=True, mudar bvec_t deveria mudar a predicao (condicionamento inerte?)"
    # e o contrario: com refine_cond=False, bvec_t nao pode ter NENHUM efeito
    # (o `del bvec_t` do forward garante isso -- este teste e' a guarda dessa
    # propriedade de design da linha PairFlow).
    model_blind = build_pairflow_star_model(base_ch=8)
    out_blind_1 = model_blind(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask)
    out_blind_2 = model_blind(vol_a, vol_b, bvec_a, bvec_b, bvec_t2, t, ensemble_mask)
    assert torch.allclose(out_blind_1, out_blind_2), \
        "com refine_cond=False, a linha PairFlow deveria ser TOTALMENTE cega ao alvo"
    print("OK: refine_cond=True ativa o alvo so' no refino; com a flag desligada a linha "
          "continua totalmente cega ao alvo (propriedade de design preservada)")

    try:
        build_pairflow_star_model(base_ch=8, refine_depth=0)
        raise AssertionError("deveria ter levantado ValueError (refine_depth=0)")
    except ValueError:
        print("OK: refine_depth < 1 levanta ValueError cedo")

    print("Todos os smoke tests de model/pairflow_star.py passaram.")


if __name__ == "__main__":
    _smoke_test()