"""
"Ensemble em estrela" para a linha RRIN/VFI-por-trincas (ver protocolo,
secao 14.5 item 1, e addendum 2026-08-27: ideia adiada em favor da loss
angular/SH da secao 15, retomada depois do bug critico de t_frac corrigido
-- ver utils/gradients.py:spherical_triplet_residual/find_best_bracket_batch
e addendum secao 12).

IDEIA CENTRAL: RRIN3D (model/rrin3d.py) preve uma direcao-alvo a partir de
UM UNICO par (a,b) de direcoes vizinhas -- mas normalmente existem VARIOS
pares candidatos "aceitaveis" pra um mesmo alvo (colineares dentro do teto
de residuo, ver utils/gradients.py:find_star_ensemble_batch), so que a
selecao de par-unico (find_best_bracket_batch) so guarda o de menor
gap_deg. RRIN3DStar recebe ATE M pares DIVERSOS (planos/normais bem
diferentes entre si -- ver find_star_ensemble_batch) para o MESMO alvo, roda
o MESMO pipeline de fluxo+warp+refino (pesos COMPARTILHADOS entre os M pares
-- "siames", nao M redes independentes) para cada um, e funde as M
predicoes por um softmax POR VOXEL sobre um logit de confianca aprendido
por par (PairWeightHead3D) -- generalizacao direta do softmax de selecao de
camada de RRIN3DLayered (model/rrin3d.py): la, as K "camadas" fundidas eram
K hipoteses de fluxo para o MESMO par de entrada; aqui, as M "camadas"
fundidas sao M pares de entrada DIFERENTES (mesma ideia de "deixar a rede
aprender em quem confiar mais", so que a fonte de incerteza e outra --
qual par de vizinhos e mais informativo pra este alvo, nao qual hipotese de
fluxo dentro de um par so).

Por que compartilhar pesos entre os M pares (em vez de M subredes
independentes, ou concatenar os M pares num tensor so de entrada)?
  (a) o numero de pares candidatos disponiveis varia por alvo/sujeito (ver
      "mask" abaixo) -- uma arquitetura com pesos por-posicao-do-feixe nao
      teria como lidar com isso sem redesenhar a cada M diferente;
  (b) fisicamente, cada par (a,b) e uma amostra INDEPENDENTE E
      INTERCAMBIAVEL do mesmo problema ("dado ESTE par, qual e o alvo?") --
      nao ha nenhuma ordem ou identidade privilegiada entre os M pares do
      feixe (ao contrario das K camadas de RRIN3DLayered, onde a rede pode
      aprender uma "especializacao" persistente por indice de camada);
      pesos compartilhados + fusao aprendida e o design correto pra essa
      simetria (mesmo principio de invariancia a permutacao usado em
      arquiteturas tipo DeepSets/attention pooling).

Fonte dos M pares: utils/gradients.py:find_star_ensemble_batch (offline, na
etapa 2b, scripts/02b_build_rrin_triplets.py --ensemble-m) -- este modulo
so consome o resultado ja pronto (vol_a/vol_b/bvec_a/bvec_b/t_frac por
posicao do feixe + uma mascara de padding), nao recalcula geometria.

Requer PyTorch (nao disponivel neste ambiente de desenvolvimento -- revisado
manualmente, testado apenas por compilacao de sintaxe; validar no cluster
com `python -m model.rrin3d_star`, smoke test no fim do arquivo, mesmo
padrao de model/rrin3d.py/model/amt3d.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .rrin3d import FlowNet3D, RefineNet3D, _conv3d, warp3d


class PairWeightHead3D(nn.Module):
    """Pequena rede que prediz um logit de confianca POR VOXEL para UMA
    predicao candidata (um dos M pares do ensemble em estrela), a partir do
    blend inicial (pos-warp, pre-refino) e dos dois volumes de entrada CRUS
    do par -- mesmos 3 canais de entrada de RefineNet3D (ver model/rrin3d.py),
    mesmo espirito do `layer_logit` de FlowNet3DLayered (que decide entre K
    hipoteses de fluxo do MESMO par); aqui decide entre M PARES DIFERENTES.
    Saida SEM sigmoid/softmax (logit cru) -- a normalizacao entre os M pares
    (softmax mascarado) e feita em RRIN3DStar.forward, nao aqui, porque
    precisa enxergar todos os M logits e a mascara de padding ao mesmo tempo."""

    def __init__(self, base_ch: int = 16, norm_type: str = "instance"):
        super().__init__()
        in_ch = 1 + 1 + 1  # blend, vol_a, vol_b
        self.net = nn.Sequential(
            _conv3d(in_ch, base_ch, norm_type=norm_type),
            nn.Conv3d(base_ch, 1, kernel_size=3, padding=1),
        )
        # init zero -- ponto de partida neutro (pesos iguais entre os M
        # pares, ate a rede aprender a diferenciar), mesmo espirito da
        # inicializacao "morna" de FlowNet3D/FlowNet3DLayered (ver
        # model/rrin3d.py). So afeta treinos NOVOS (sem resume).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, blend, vol_a, vol_b):
        x = torch.cat([blend, vol_a, vol_b], dim=1)
        return self.net(x)  # (B,1,D,H,W), logit cru


class RRIN3DStar(nn.Module):
    """Ensemble em estrela: funde ate M predicoes RRIN3D (pesos
    compartilhados) de M pares de entrada candidatos DIFERENTES para o MESMO
    alvo, via um softmax POR VOXEL sobre um logit de confianca aprendido por
    par (PairWeightHead3D). Ver docstring do modulo para a motivacao
    completa e a analogia/diferenca com RRIN3DLayered.

    NAO tem um hiperparametro `num_layers`/`M` fixo na arquitetura -- M e
    simplesmente uma dimensao do tensor de entrada em tempo de execucao
    (quantos pares o feixe do batch atual tem), lida do shape de
    `ensemble_mask`. Isso e deliberado: o mesmo checkpoint funciona pra
    qualquer M >= 1 usado na reconstrucao (ex.: treinar com M=3 e
    reconstruir testando M=1/3/5 pra comparar o efeito do tamanho do
    ensemble), diferente de RRIN3DLayered onde K e fixo no checkpoint.

    IMPORTANTE -- mascara de padding: um alvo pode ter menos de M pares
    REAIS disponiveis (ver utils/gradients.py:find_star_ensemble_batch e
    utils/rrin_dataset.py:RRINTripletDataset.ensemble_m) -- `ensemble_mask`
    (B,M) marca isso; posicoes com mask=False recebem logit de fusao
    -inf ANTES do softmax, entao NUNCA contribuem pra predicao final
    (pesos de fusao dessas posicoes saem exatamente 0, nao so "pequenos").
    Pressuposto (garantido por construcao em find_star_ensemble_batch/
    RRINTripletDataset, ver docstrings la): toda linha de `ensemble_mask`
    tem PELO MENOS uma posicao True -- softmax de uma linha inteiramente
    -inf daria NaN; nunca deveria acontecer na pratica (mesma garantia que
    ja vale hoje para "valid"/"between" de par-unico), mas nao e
    validado defensivamente aqui (custo de compute por chamada nao
    justificado para uma invariante ja garantida rio acima).

    Uso:
        model = RRIN3DStar()
        pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                      quality=quality)
    vol_a, vol_b: (B, M, 1, D, H, W) -- M pares candidatos de entrada.
    bvec_a, bvec_b, bvec_t: (B, M, 3) -- bvec_t e o MESMO alvo repetido nas M
        posicoes (ver utils/rrin_dataset.py:_ensemble_tensors), mas mantido
        com dimensao M aqui so pra simplificar o achatamento (B,M,...) ->
        (B*M,...) do forward, nao porque varie de fato entre posicoes.
    t: (B, M) -- t_frac de cada par (varia entre posicoes, MESMO alvo).
    ensemble_mask: (B, M) bool -- True = par real nesta posicao do feixe.
    quality: (B, M, 2) ou None -- residual_deg/gap_deg normalizados de cada
        par (so usado se use_quality_cond=True, ver model/rrin3d.py).
    retorna: (B, 1, D, H, W) -- direcao-alvo predita (fusao dos <=M pares).
    """

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, use_quality_cond: bool = False,
                 norm_type: str = "instance"):
        super().__init__()
        self.use_quality_cond = use_quality_cond
        self.norm_type = norm_type
        self.flow_net = FlowNet3D(base_ch=base_ch, max_disp=max_disp,
                                   use_quality_cond=use_quality_cond, norm_type=norm_type)
        self.refine_net = RefineNet3D(base_ch=base_ch, norm_type=norm_type)
        self.weight_head = PairWeightHead3D(base_ch=base_ch, norm_type=norm_type)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=None,
                return_pairs=False):
        b, m = vol_a.shape[0], vol_a.shape[1]

        def _flat(x):
            return x.reshape(b * m, *x.shape[2:])

        vol_a_f = _flat(vol_a)
        vol_b_f = _flat(vol_b)
        bvec_a_f = _flat(bvec_a)
        bvec_b_f = _flat(bvec_b)
        bvec_t_f = _flat(bvec_t)
        t_f = t.reshape(b * m)
        quality_f = _flat(quality) if quality is not None else None

        # pipeline RRIN3D compartilhado, rodado UMA VEZ para as B*M
        # "amostras" (mesmo truque de achatamento em batch de
        # scripts/04b_train_rrin.py:_sh_bundle_forward -- equivalente a um
        # loop Python de M chamadas, mas um unico forward batelado).
        flow_a, flow_b, vis_logit = self.flow_net(vol_a_f, vol_b_f, bvec_a_f, bvec_b_f,
                                                    bvec_t_f, t_f, quality=quality_f)
        warped_a = warp3d(vol_a_f, flow_a)
        warped_b = warp3d(vol_b_f, flow_b)
        vis = torch.sigmoid(vis_logit)
        t_map = t_f.view(-1, 1, 1, 1, 1)
        w_a = (1.0 - t_map) * vis
        w_b = t_map * (1.0 - vis)
        denom = (w_a + w_b).clamp(min=1e-6)
        blend_f = (w_a * warped_a + w_b * warped_b) / denom
        residual_f = self.refine_net(blend_f, vol_a_f, vol_b_f)
        pred_f = blend_f + residual_f                          # (B*M,1,D,H,W)
        weight_logit_f = self.weight_head(blend_f, vol_a_f, vol_b_f)  # (B*M,1,D,H,W)

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


def build_star_model(base_ch: int = 16, max_disp: float = 0.5, use_quality_cond: bool = False,
                      norm_type: str = "instance") -> RRIN3DStar:
    """Wrapper trivial (mesmo espirito de build_rrin_model em model/rrin3d.py)
    -- existe so para scripts/04e_train_rrin_star.py e
    scripts/05f_reconstruct_rrin_star.py nao precisarem instanciar a classe
    diretamente, deixando espaco para uma futura variante (ex. K camadas
    tambem dentro do ensemble em estrela) sem mudar a assinatura dos scripts
    que chamam esta funcao."""
    return RRIN3DStar(base_ch=base_ch, max_disp=max_disp, use_quality_cond=use_quality_cond,
                       norm_type=norm_type)


def _smoke_test():
    """Forward pass com tensores pequenos aleatorios -- mesmo padrao de
    model/rrin3d.py/model/amt3d.py. Roda no cluster: python -m model.rrin3d_star"""
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
    ensemble_mask[0, -1] = False  # simula 1o item do batch com so 2/3 pares reais
    expected = (b, 1, d, h, w)

    model = build_star_model(base_ch=8, use_quality_cond=False)
    out = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask)
    assert out.shape == expected, f"shape mismatch: {out.shape} != {expected}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"smoke test OK (use_quality_cond=False), output shape: {tuple(out.shape)}, "
          f"{n_params} parametros")

    out2, extra = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, return_pairs=True)
    assert torch.allclose(out, out2), "forward deveria ser deterministico (sem dropout/RNG)"
    pi = extra["pi"]
    assert pi.shape == (b, m, 1, d, h, w)
    pi_sum = pi.sum(dim=1)
    assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5), \
        "pesos de fusao (pi) nao somam 1 por voxel"
    # posicoes mascaradas (False) devem ter peso EXATAMENTE zero (nao so pequeno)
    masked_pi = pi[0, -1]
    assert torch.allclose(masked_pi, torch.zeros_like(masked_pi)), \
        "posicao mascarada do feixe deveria ter peso de fusao exatamente 0"
    print("OK: pi soma 1 por voxel e posicoes mascaradas tem peso exatamente 0")

    model_q = build_star_model(base_ch=8, use_quality_cond=True)
    quality = torch.rand(b, m, 2)
    out_q = model_q(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=quality)
    assert out_q.shape == expected, f"shape mismatch (com quality): {out_q.shape} != {expected}"
    print(f"smoke test OK (use_quality_cond=True), output shape: {tuple(out_q.shape)}")

    # M=1 (ensemble degenerado a um unico par) deve funcionar normalmente --
    # nao ha nenhum caso especial pra M=1 no codigo (softmax de 1 elemento
    # nao-mascarado da sempre peso 1.0), mas confere explicitamente.
    vol_a1, vol_b1 = vol_a[:, :1], vol_b[:, :1]
    bvec_a1, bvec_b1, bvec_t1 = bvec_a[:, :1], bvec_b[:, :1], bvec_t[:, :1]
    t1 = t[:, :1]
    mask1 = torch.ones(b, 1, dtype=torch.bool)
    out1, extra1 = model(vol_a1, vol_b1, bvec_a1, bvec_b1, bvec_t1, t1, mask1, return_pairs=True)
    assert out1.shape == expected
    assert torch.allclose(extra1["pi"], torch.ones_like(extra1["pi"]))
    print("OK: M=1 funciona (peso de fusao = 1.0, degenerado a RRIN3D de par unico)")

    # linha do batch com TODAS as posicoes mascaradas (menos a 1a, que a
    # garantia de construcao upstream nunca deixa cair) -- so testa que uma
    # linha com exatamente 1 posicao real (as demais False) nao gera NaN.
    mask_single_real = torch.zeros(b, m, dtype=torch.bool)
    mask_single_real[:, 0] = True
    out_single, extra_single = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, mask_single_real,
                                      return_pairs=True)
    assert not torch.isnan(out_single).any(), "saida nao deveria ter NaN"
    assert torch.allclose(extra_single["pi"][:, 0], torch.ones_like(extra_single["pi"][:, 0]))
    print("OK: linha com so 1 posicao real no feixe nao gera NaN (peso todo nela)")


if __name__ == "__main__":
    _smoke_test()