"""
HFD3D -- adaptacao 3D de "Hierarchical Flow Diffusion for Efficient Frame
Interpolation" (Hai et al., CVPR 2025, arXiv:2504.00380,
https://hfd-interpolation.github.io -- SEM codigo publico encontrado, so a
pagina do projeto; a arquitetura abaixo foi reconstruida a partir do texto
do paper/abstract, nao de um repositorio de referencia como foi possivel
para AMT3D -- ver "GRAU DE CONFIANCA" no fim deste docstring) para o mesmo
esquema de trincas (par de direcoes de entrada a,b + direcao-alvo t) usado
por RRIN3D (model/rrin3d.py) e AMT3D (model/amt3d.py).

POR QUE ESTA REDE EXISTE (ver protocolo, addendum 2026-08-27 secao 8): a
RRIN3D e a AMT3D ja mostraram (addendum secao 6) que duas arquiteturas de
fluxo otico de complexidade MUITO diferente (RRIN, simples; AMT, correlacao
all-pairs + multi-field) convergem pro MESMO patamar de erro na MESMA
velocidade de treino, e ambas perdem ~60% de nmse relativo pro baseline_sh
-- evidencia forte de que o gargalo e a PREMISSA de fluxo/correspondencia
espacial entre direcoes de gradiente (que a analogia OLAT, protocolo secao
10, ja preve que nao deveria existir fisicamente), nao a capacidade ou
sofisticacao de nenhuma implementacao especifica de fluxo. HFD3D testa uma
hipotese diferente das duas anteriores: em vez de REGREDIR o fluxo
diretamente (RRIN3D/AMT3D), aqui o fluxo bilateral e GERADO por um processo
iterativo de difusao (denoising), condicionado nas mesmas features de
correlacao all-pairs ja usadas pela AMT3D. Um metodo de kernel-based VFI
(SepConv/AdaCoF/EDSC) foi considerado e DESCARTADO como terceira arquitetura
(ver conversa/protocolo) por compartilhar o mesmo defeito de fundo do fluxo
(assume que o alvo e previsivel a partir de uma vizinhanca ESPACIAL da
entrada, premissa que a analogia OLAT ja contesta) -- nao teria valor
diagnostico independente. A difusao, por outro lado, ainda faz warping (ver
abaixo), mas testa se um processo GERATIVO iterativo (que pode, em
principio, "alucinar" detalhes plausiveis em vez de so regredir uma media)
se sai diferente nos casos geometricamente dificeis -- se HFD3D tambem
falhar do mesmo jeito, e uma terceira confirmacao independente, ainda mais
forte, de que o problema e a premissa de correspondencia espacial em si,
nao o mecanismo de estimacao de fluxo usado.

PIPELINE (baseado no mecanismo central do HFD -- ver "SIMPLIFICACOES
DELIBERADAS" abaixo para o que foi conscientemente reduzido):
  1. FeatureEncoder3D (IMPORTADA de model/amt3d.py, NAO duplicada): mesmo
     encoder siames fino+grosso da AMT3D -- aqui so a escala FINA e usada
     (ver simplificacao de piramide abaixo), a escala grossa retornada e
     descartada.
  2. build_correlation / _corr_lookup_3d (IMPORTADAS de model/amt3d.py):
     correlacao all-pairs entre fine_a e fine_b, calculada UMA VEZ por
     forward (nao uma vez por passo de difusao); o LOOKUP local, sim, e
     recalculado a cada passo de denoising, ao redor da estimativa de
     fluxo ATUAL (ruidosa) daquele passo -- mesmo espirito iterativo de
     RAFT/AMT, agora guiando um denoiser em vez de um regressor direto.
  3. Processo de difusao SOBRE O FLUXO (nao sobre pixels/latentes, ver nota
     de design abaixo): a variavel difundida e x0 = concat(flow_a, flow_b),
     um tensor (B,6,D,H,W) em coordenadas normalizadas -1..1 (MESMA
     convencao de model.rrin3d.warp3d -- fluxo comparavel entre "escalas"
     sem fator de reescala, mesma nota de design ja usada em AMT3D).
     Formulas DDPM/DDIM padrao (Ho et al. 2020; Song et al. 2021 pro DDIM),
     implementadas explicitamente abaixo para permitir conferencia manual
     (nao ha codigo de referencia publico do HFD para comparar linha a
     linha, ao contrario do que foi possivel com AMT3D/github.com/MCG-NKU/AMT):
       - forward (treino):      x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*ruido
       - denoiser preve:        eps_theta(x_t, t_difusao, condicionamento) ~= ruido
       - loss (treino):         MSE(eps_theta, ruido)  -- "simple loss" do DDPM original,
                                 sem ponderacao extra por t_difusao.
       - passo DDIM (inferencia, deterministico, eta=0):
           x0_pred    = (x_t - sqrt(1-alpha_bar_t)*eps_theta) / sqrt(alpha_bar_t)
           x_{t_prev} = sqrt(alpha_bar_{t_prev})*x0_pred + sqrt(1-alpha_bar_{t_prev})*eps_theta
         com alpha_bar_{-1} := 1.0 (passo terminal -> x_{-1} = x0_pred exatamente).
  4. Cabeca de visibilidade DETERMINISTICA (fora do loop de difusao,
     diferente do fluxo): so depois que o fluxo final (denoised) sai do
     loop de amostragem e que uma pequena cabeca conv (`vis_head`/`vis_out`)
     preve a visibilidade de blend -- o HFD tambem separa isso (o
     "flow-guided image synthesizer" do paper e um modulo DETERMINISTICO
     downstream da difusao de fluxo, nao parte do processo de denoising).
  5. warp3d + blend (MESMA formula de model.rrin3d.RRIN3D.forward) +
     RefineNet3D (IMPORTADA de model/rrin3d.py, instancia PROPRIA, pesos
     NAO compartilhados com RRIN3D/AMT3D -- mesmo padrao de "cada modelo
     tem seu proprio refine_net" ja estabelecido entre RRIN3D e AMT3D).

TREINO REQUER UM "PROFESSOR" DE FLUXO PSEUDO-VERDADEIRO (diferente de
RRIN3D/AMT3D, que treinam direto contra o sinal-alvo): o HFD original
supervisiona o estagio de difusao com fluxo bilateral pseudo-verdadeiro de
uma rede de fluxo PRE-TREINADA (pra evitar que o denoiser tenha que
aprender fluxo do zero so a partir de gradiente de pixel, que e um sinal de
treino fraco/indireto pra difusao). Aqui, em vez de importar uma rede de
fluxo de video 2D pre-treinada (nao existe uma pra dMRI 3D, obviamente), a
PROPRIA AMT3D ja treinada deste projeto (model/amt3d.py, a rede de fluxo
mais forte ja validada aqui, ver addendum secao 6) faz esse papel --
`AMT3D.forward(..., return_flow=True)` (adicionado em 2026-08-27
especificamente para isto, ver model/amt3d.py) retorna o fluxo bilateral
final fundido, usado como alvo x0 do denoising. Consequencia pratica
importante: **treinar a HFD3D exige um checkpoint da AMT3D ja treinado**
(`--teacher-checkpoint` em scripts/04d_train_hfd.py) -- nao e um modelo
"do zero" como RRIN3D/AMT3D, e o resultado da HFD3D fica limitado pela
qualidade do fluxo que a AMT3D professora conseguiu aprender (se a AMT3D
falha numa trinca geometricamente invalida, o fluxo pseudo-verdadeiro que
a HFD3D tenta replicar naquela trinca ja e ruim por construcao -- isso e
uma limitacao conhecida do design, nao um bug, e devera ser discutida ao
interpretar os resultados).

SIMPLIFICACOES DELIBERADAS vs. o HFD original (documentadas aqui pra nao
confundir "port inspirado" com "reproducao literal" -- SEM codigo de
referencia publico, o grau de fidelidade aqui e MENOR do que foi possivel
com a AMT3D):
  - Piramide de 1 nivel (so fino), nao 3 como no HFD original: o HFD usa 3
    niveis (1/16 a 1/4 de resolucao de video) porque quadros de video tem
    centenas/milhares de pixels por eixo: comprimir hierarquicamente reduz
    custo. Um patch 3D de 10-24 voxels por eixo e MENOR que o nivel mais
    grosseiro do HFD numa imagem real -- colapsar pra 1 nivel (mesma logica
    ja usada pela AMT3D ao colapsar de 4 pra 2 niveis, aqui levada um passo
    adiante) evita amplificar o custo (K passos de difusao POR NIVEL
    multiplicaria o numero de avaliacoes de rede por nivel extra) sem
    perder capacidade de contexto relevante nessa escala pequena.
  - `num_timesteps=1000` passos de treino (schedule beta linear
    1e-4->0.02, EXATAMENTE os valores canonicos de Ho et al. 2020 DDPM,
    escolhidos por serem o ponto de partida mais testado da literatura, na
    ausencia de um valor especifico publicado pelo HFD no material
    disponivel) mas so `num_sample_steps=6` passos DDIM na inferencia --
    mesma ordem de grandeza relatada pelo HFD (6 passos DDIM, la
    multiplicados por 3 niveis; aqui por 1 nivel so, ver item acima).
  - Rede denoisadora: um bloco conv pequeno (2x `_conv3d` + 1 conv final),
    NAO um U-Net com encoder-decoder proprio como o HFD descreve -- o HFD
    opera em features de imagem em resolucao real (motiva um U-Net); aqui,
    com um volume de ate 24^3 voxels, um bloco conv raso condicionado (nas
    features de correlacao + geometria + timestep de difusao) e a escolha
    mais consistente com o resto deste codebase (RRIN3D/AMT3D tambem usam
    blocos conv rasos, nao U-Nets, pela mesma razao de escala).
  - Sem as 3 fases de treino do HFD (sintetizador sozinho -> denoiser
    sozinho -> fine-tune conjunto): aqui o treino e de UM ESTAGIO SO,
    combinando a loss de difusao (contra o fluxo professor da AMT3D) com a
    loss fotometrica no sinal final (contra o sinal-alvo real), ambas
    otimizadas juntas desde o inicio -- simplificacao aceitavel dado que
    (a) ja existe uma "professora" de fluxo pronta (a AMT3D, que o HFD
    original nao tinha disponivel pra video generico) e (b) o escopo aqui e
    diagnostico, nao alcancar o melhor resultado possivel de HFD.
  - `use_quality_cond` implementado do mesmo jeito que RRIN3D/AMT3D (2
    canais extras de condicionamento), nao ficou como TODO.
  - `norm_type="batch"` funciona de graca por reaproveitar `_conv3d`/
    `_norm3d` de model/rrin3d.py (mesmo artefato de "costura" ja conhecido
    se usado com sliding-window, ver protocolo secao 14.4).

NOTA DE DESIGN -- por que difundir o FLUXO, nao pixels/latentes (mesma
logica do HFD original, ver a pesquisa de literatura registrada no
addendum secao 8): um campo de fluxo e uma variavel de baixa dimensao,
suave e estruturada (poucos canais escalares por voxel), bem mais facil de
gerar por difusao num volume pequeno (10^3-24^3 voxels) do que texturas em
espaco de pixel ou tokens latentes comprimidos -- que dependem de
redundancia espacial em escala de imagem natural (centenas de milhares de
pixels) pra funcionar, redundancia que nao existe num patch 3D tao pequeno.
Essa foi justamente a razao pela qual EDEN (difusao em espaco latente,
CVPR 2025, tambem pesquisado no addendum secao 8) foi DESCARTADO como
candidato a porte: o "tokenizer" dele depende de compressao espacial
calibrada pra imagens reais, sem equivalente validavel aqui.

GRAU DE CONFIANCA NESTA IMPLEMENTACAO (importante, ler antes de comparar
resultados com a literatura): ao contrario da AMT3D (onde havia um
repositorio de referencia, github.com/MCG-NKU/AMT, conferido linha a linha
antes de escrever o port), o HFD NAO tem codigo publico disponivel no
momento desta implementacao (so a pagina do projeto,
hfd-interpolation.github.io, sem repositorio de treino) -- a arquitetura
acima foi reconstruida a partir do texto do abstract/paper (arXiv:2504.00380)
via pesquisa web, nao verificada contra uma implementacao de referencia.
As formulas DDPM/DDIM em si SAO padrao da literatura de difusao (nao
especificas do HFD) e foram implementadas e conferidas manualmente aqui
(ver funcoes `q_sample`/`ddim_step` abaixo). Trate esta rede como "inspirada
no mecanismo central do HFD" (difundir fluxo bilateral, nao pixels/latentes),
nao como uma reproducao fiel do paper -- o valor diagnostico (testar se um
processo gerativo iterativo tambem falha nas trincas geometricamente
invalidas) nao depende de fidelidade arquitetural exata ao HFD, so de
genuinamente NAO ser mais uma regressao direta de fluxo (que RRIN3D/AMT3D
ja cobriram).

Requer PyTorch (nao disponivel neste ambiente de desenvolvimento --
revisado manualmente, testado apenas por compilacao de sintaxe; validar no
cluster com `python -m model.hfd3d`, smoke test no fim do arquivo, mesmo
padrao de model/rcae.py, model/rrin3d.py e model/amt3d.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reaproveita a infra ja validada de RRIN3D/AMT3D em vez de duplicar.
from .rrin3d import _conv3d, _norm3d, _repeat_vec_3d, warp3d, RefineNet3D
from .amt3d import FeatureEncoder3D, build_correlation, _corr_lookup_3d


def make_beta_schedule(num_timesteps: int, beta_start: float = 1e-4,
                        beta_end: float = 0.02) -> torch.Tensor:
    """Schedule beta LINEAR, valores canonicos de Ho et al. 2020 (DDPM) --
    escolhidos por serem o ponto de partida mais testado da literatura na
    ausencia de um valor publicado especificamente pelo HFD (ver docstring
    do modulo, "GRAU DE CONFIANCA"). Retorna (num_timesteps,)."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def diffusion_constants(betas: torch.Tensor):
    """alphas = 1-betas; alpha_bars = cumprod(alphas) -- convencao padrao
    DDPM. alpha_bars[0] proximo de 1 (quase sem ruido), alpha_bars[-1]
    proximo de 0 (quase ruido puro)."""
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars


def q_sample(x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor,
             alpha_bars: torch.Tensor) -> torch.Tensor:
    """Difusao direta (forward process): x_t = sqrt(alpha_bar_t)*x0 +
    sqrt(1-alpha_bar_t)*noise. `t_idx`: (B,) long, indices de timestep de
    difusao em [0, num_timesteps-1], um por item do batch (treino -- cada
    item do batch recebe um timestep de difusao SORTEADO independentemente,
    convencao padrao DDPM). `x0`/`noise`: mesmo shape, (B,C,*spatial)."""
    ab = alpha_bars[t_idx]  # (B,)
    view_shape = (-1,) + (1,) * (x0.dim() - 1)
    ab = ab.view(*view_shape)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def ddim_step(x_t: torch.Tensor, eps_pred: torch.Tensor,
              alpha_bar_t: float, alpha_bar_t_prev: float):
    """Um passo de amostragem DDIM determinstico (eta=0, Song et al. 2021,
    eq. 12 com sigma_t=0). `alpha_bar_t`/`alpha_bar_t_prev`: escalares
    python (MESMO valor de alpha_bar pra todo o batch -- a amostragem aqui
    usa um cronograma de timesteps COMPARTILHADO entre itens do batch,
    convencao padrao de inferencia DDIM, diferente do treino onde cada item
    sorteia seu proprio timestep). `alpha_bar_t_prev=1.0` e o caso terminal
    (t_prev=-1, "antes do inicio" do schedule) -- nesse caso a formula
    reduz exatamente a x_prev = x0_pred (sem ruido residual), o passo final
    da amostragem. Retorna (x_prev, x0_pred)."""
    sqrt_ab_t = math.sqrt(alpha_bar_t)
    sqrt_1m_ab_t = math.sqrt(max(1.0 - alpha_bar_t, 0.0))
    x0_pred = (x_t - sqrt_1m_ab_t * eps_pred) / sqrt_ab_t
    sqrt_ab_prev = math.sqrt(alpha_bar_t_prev)
    sqrt_1m_ab_prev = math.sqrt(max(1.0 - alpha_bar_t_prev, 0.0))
    x_prev = sqrt_ab_prev * x0_pred + sqrt_1m_ab_prev * eps_pred
    return x_prev, x0_pred


def ddim_timesteps(num_timesteps: int, num_sample_steps: int) -> list[int]:
    """Subsequencia DECRESCENTE de `num_sample_steps` indices de timestep
    em [0, num_timesteps-1] pra amostragem DDIM, espacados uniformemente
    (mesma convencao de bibliotecas de referencia como HuggingFace
    diffusers' DDIMScheduler) -- garante que o primeiro indice seja
    `num_timesteps-1` (ruido quase puro, ponto de partida da amostragem) e
    o ultimo seja `0` (quase sem ruido; o passo seguinte, fora desta lista,
    e o terminal com alpha_bar_prev=1.0, ver `ddim_step`)."""
    if num_sample_steps <= 0:
        raise ValueError(f"num_sample_steps deve ser >= 1 (recebido {num_sample_steps})")
    if num_sample_steps == 1:
        return [num_timesteps - 1]
    raw = torch.linspace(num_timesteps - 1, 0, num_sample_steps).round().long().tolist()
    # remove duplicatas preservando ordem decrescente (pode ocorrer se
    # num_sample_steps for proximo de num_timesteps, nao no regime tipico
    # de poucos passos usado aqui, mas protegido por seguranca).
    seen = set()
    ts = []
    for v in raw:
        if v not in seen:
            seen.add(v)
            ts.append(v)
    return ts


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Embedding senoidal padrao (Transformer/DDPM, Ho et al. 2020, sec.
    3.2/apendice) do indice de timestep de DIFUSAO (nao confundir com `t`,
    a fracao de interpolacao entre bvec_a/bvec_b -- ver nomenclatura no
    resto do modulo: `t` sempre se refere a interpolacao, `timestep`/
    `t_idx` sempre ao passo de difusao). `timesteps`: (B,) long ou float.
    Retorna (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class HFD3D(nn.Module):
    """Modelo completo -- ver docstring do modulo para o pipeline detalhado
    (encoder siames fino (AMT3D) -> correlacao all-pairs (AMT3D) -> fluxo
    bilateral GERADO por difusao (DDPM/DDIM, ver `sample_flow`) ->
    visibilidade determinstica -> warp3d+blend -> RefineNet3D (RRIN3D)).

    Uso pra INFERENCIA/RECONSTRUCAO (MESMA assinatura de forward de
    model.rrin3d.RRIN3D.forward e model.amt3d.AMT3D.forward, de proposito):
        model = HFD3D(num_sample_steps=6)
        pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None)

    Uso pra TREINO (metodo separado, `diffusion_loss` -- NAO e o forward,
    porque o treino precisa do fluxo pseudo-verdadeiro de uma AMT3D
    professora ja treinada, que vem de FORA deste modelo, ver docstring do
    modulo e scripts/04d_train_hfd.py):
        loss_diff = model.diffusion_loss(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t,
                                          target_flow_a, target_flow_b, quality=None)
        pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None)  # loss fotometrica
        loss = loss_diff + photometric_loss(pred, target_signal)

    vol_a, vol_b: (B, 1, D, H, W); bvec_a, bvec_b, bvec_t: (B, 3); t: (B,);
    quality: (B, 2) ou None (obrigatorio se use_quality_cond=True).

    num_timesteps (default 1000): passos do schedule de difusao usado no
    TREINO (quantos t_idx possiveis existem) -- nao afeta o custo de
    inferencia (isso e `num_sample_steps`). ARQUITETURAL (afeta o buffer
    `alpha_bars`), mas nao muda nenhum shape de peso da rede -- ainda assim
    tratado como bloqueante em resume por simplicidade/consistencia com o
    resto do treino (mudar o schedule no meio do treino tornaria o
    checkpoint do otimizador/scheduler inconsistente com o novo regime de
    ruido).

    num_sample_steps (default 6): quantos passos DDIM rodar na
    AMOSTRAGEM/inferencia (`sample_flow`/`forward`) -- NAO afeta nenhum
    peso/shape da rede, so o custo/qualidade da amostragem. Pode ser
    mudado livremente entre treino e reconstrucao sem qualquer
    incompatibilidade de checkpoint (avisado, nao bloqueante, ver
    scripts/04d_train_hfd.py).

    corr_radius (default 3): mesmo papel/convencao de model.amt3d.AMT3D
    (raio da janela de lookup de correlacao) -- ARQUITETURAL (muda o
    shape da primeira camada do denoiser), bloqueante em resume."""

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, corr_radius: int = 3,
                 use_quality_cond: bool = False, norm_type: str = "instance",
                 num_timesteps: int = 1000, num_sample_steps: int = 6,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 time_emb_dim: int = 32):
        super().__init__()
        if num_timesteps < 1:
            raise ValueError(f"num_timesteps deve ser >= 1 (recebido {num_timesteps})")
        self.base_ch = base_ch
        self.max_disp = max_disp
        self.corr_radius = corr_radius
        self.use_quality_cond = use_quality_cond
        self.norm_type = norm_type
        self.num_timesteps = num_timesteps
        self.num_sample_steps = num_sample_steps
        self.time_emb_dim = time_emb_dim

        betas = make_beta_schedule(num_timesteps, beta_start, beta_end)
        alphas, alpha_bars = diffusion_constants(betas)
        # buffers (nao parametros treinaveis, mas movem com .to(device) e
        # sao salvos/restaurados pelo state_dict do checkpoint).
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        # Encoder siames (IMPORTADO de model.amt3d) -- so a escala FINA e
        # usada aqui (piramide colapsada pra 1 nivel, ver docstring do
        # modulo); a escala grossa retornada por FeatureEncoder3D.forward
        # e descartada (nao instanciamos uma segunda rede so pra isso).
        self.encoder = FeatureEncoder3D(base_ch=base_ch, norm_type=norm_type)

        cond_ch = 3 + 3 + 3 + 1  # bvec_a, bvec_b, bvec_t, t (fracao de interpolacao)
        if use_quality_cond:
            cond_ch += 2  # residual_norm, gap_norm -- mesma convencao de RRIN3D/AMT3D

        win = (2 * corr_radius + 1) ** 3
        # fine_a, fine_b (cada base_ch) + lookup_ab, lookup_ba (cada win) +
        # fluxo ruidoso atual (6: flow_a+flow_b) + condicionamento
        # geometrico + embedding do timestep de DIFUSAO (broadcast espacial).
        denoiser_in_ch = base_ch * 2 + win * 2 + 6 + cond_ch + time_emb_dim
        self.denoise_head = nn.Sequential(
            _conv3d(denoiser_in_ch, base_ch * 2, norm_type=norm_type),
            _conv3d(base_ch * 2, base_ch * 2, norm_type=norm_type),
        )
        self.denoise_out = nn.Conv3d(base_ch * 2, 6, kernel_size=3, padding=1)
        # zero-init da ultima camada: estabiliza a previsao de ruido no
        # inicio do treino (eps_pred~=0), pratica comum em implementacoes
        # de difusao -- IMPORTANTE: ao contrario do zero-init "neutro" de
        # RRIN3D/AMT3D (que produz uma saida SENSATA/identidade no init),
        # aqui isso NAO torna a amostragem sensata antes do treino -- ver
        # nota no docstring do modulo, "GRAU DE CONFIANCA": um modelo de
        # difusao so produz amostras uteis DEPOIS de treinado, mesmo com
        # este zero-init (e um comportamento esperado da familia de
        # modelos, nao uma regressao em relacao aos outros modelos deste
        # codebase).
        nn.init.zeros_(self.denoise_out.weight)
        nn.init.zeros_(self.denoise_out.bias)

        # Cabeca de visibilidade DETERMINISTICA (fora do loop de difusao,
        # ver docstring do modulo, item 4) -- ve as features finas + o
        # fluxo bilateral FINAL (ja denoised) que saiu da amostragem.
        self.vis_head = nn.Sequential(
            _conv3d(base_ch * 2 + 6, base_ch, norm_type=norm_type),
        )
        self.vis_out = nn.Conv3d(base_ch, 1, kernel_size=3, padding=1)
        # zero-init -> vis=sigmoid(0)=0.5, neutro -- MESMA convencao de
        # warm-start de RRIN3D/AMT3D (aqui sim aplicavel, porque esta
        # cabeca e determinstica, nao parte do processo de difusao).
        nn.init.zeros_(self.vis_out.weight)
        nn.init.zeros_(self.vis_out.bias)

        self.refine_net = RefineNet3D(base_ch=base_ch, norm_type=norm_type)

    def _predict_noise(self, noisy_flow, fine_a, fine_b, corr_ab, corr_ba,
                        bvec_a, bvec_b, bvec_t, t, quality, timestep_idx):
        """Um forward do denoiser: preve o ruido `eps` a partir do fluxo
        ruidoso atual `noisy_flow` (B,6,D,H,W) + condicionamento. Chamado
        tanto no treino (`diffusion_loss`, timestep sorteado por item do
        batch) quanto na amostragem (`sample_flow`, timestep compartilhado
        pelo batch, ver docstring de `ddim_step`)."""
        spatial = fine_a.shape[-3:]
        flow_a_n = noisy_flow[:, 0:3]
        flow_b_n = noisy_flow[:, 3:6]
        lookup_ab = _corr_lookup_3d(corr_ab, flow_a_n, self.corr_radius)
        lookup_ba = _corr_lookup_3d(corr_ba, flow_b_n, self.corr_radius)
        t_col = t.view(-1, 1)
        time_emb = sinusoidal_time_embedding(timestep_idx, self.time_emb_dim)  # (B, time_emb_dim)
        parts = [fine_a, fine_b, lookup_ab, lookup_ba, noisy_flow,
                 _repeat_vec_3d(bvec_a, spatial), _repeat_vec_3d(bvec_b, spatial),
                 _repeat_vec_3d(bvec_t, spatial), _repeat_vec_3d(t_col, spatial),
                 _repeat_vec_3d(time_emb, spatial)]
        if self.use_quality_cond:
            parts.append(_repeat_vec_3d(quality, spatial))
        feat = self.denoise_head(torch.cat(parts, dim=1))
        return self.denoise_out(feat)  # (B, 6, D, H, W)

    def diffusion_loss(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t,
                        target_flow_a, target_flow_b, quality=None):
        """Loss de treino do denoiser (objetivo DDPM "simple", MSE contra o
        ruido, sem ponderacao extra por timestep -- Ho et al. 2020, eq.
        14). `target_flow_a`/`target_flow_b`: (B,3,D,H,W), fluxo
        pseudo-verdadeiro vindo de uma AMT3D professora JA TREINADA e
        CONGELADA (ver scripts/04d_train_hfd.py -- este metodo nao sabe
        nada sobre a professora, so recebe o fluxo-alvo pronto, o que
        mantem este modulo desacoplado de qual rede foi usada como fonte)."""
        if self.use_quality_cond and quality is None:
            raise ValueError("use_quality_cond=True mas `quality` nao foi passado a diffusion_loss")
        fine_a, _ = self.encoder(vol_a)
        fine_b, _ = self.encoder(vol_b)
        corr_ab = build_correlation(fine_a, fine_b)
        corr_ba = build_correlation(fine_b, fine_a)

        x0 = torch.cat([target_flow_a, target_flow_b], dim=1)  # (B,6,D,H,W)
        bsz = x0.shape[0]
        t_idx = torch.randint(0, self.num_timesteps, (bsz,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t_idx, noise, self.alpha_bars)
        eps_pred = self._predict_noise(x_t, fine_a, fine_b, corr_ab, corr_ba,
                                        bvec_a, bvec_b, bvec_t, t, quality, t_idx)
        return F.mse_loss(eps_pred, noise)

    def sample_flow(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None):
        """Amostragem DDIM (determinstica, eta=0) do fluxo bilateral final,
        partindo de ruido gaussiano puro. `num_sample_steps` passos (default
        6, ver __init__). Retorna (flow_a_final, flow_b_final, fine_a,
        fine_b) -- as duas ultimas sao reaproveitadas por `forward` pra
        montar a entrada da cabeca de visibilidade sem recalcular o
        encoder."""
        if self.use_quality_cond and quality is None:
            raise ValueError("use_quality_cond=True mas `quality` nao foi passado a sample_flow")
        fine_a, _ = self.encoder(vol_a)
        fine_b, _ = self.encoder(vol_b)
        corr_ab = build_correlation(fine_a, fine_b)
        corr_ba = build_correlation(fine_b, fine_a)
        spatial = fine_a.shape[-3:]
        bsz = vol_a.shape[0]

        x_t = torch.randn(bsz, 6, *spatial, device=vol_a.device, dtype=vol_a.dtype)
        timesteps = ddim_timesteps(self.num_timesteps, self.num_sample_steps)
        alpha_bars = self.alpha_bars
        safety_bound = 3.0 * self.max_disp  # ver nota no __init__ sobre instabilidade pre-treino

        for i, cur_t in enumerate(timesteps):
            t_idx_tensor = torch.full((bsz,), cur_t, device=vol_a.device, dtype=torch.long)
            eps_pred = self._predict_noise(x_t, fine_a, fine_b, corr_ab, corr_ba,
                                            bvec_a, bvec_b, bvec_t, t, quality, t_idx_tensor)
            ab_t = alpha_bars[cur_t].item()
            ab_prev = alpha_bars[timesteps[i + 1]].item() if i + 1 < len(timesteps) else 1.0
            x_t, _ = ddim_step(x_t, eps_pred, ab_t, ab_prev)
            x_t = torch.clamp(x_t, -safety_bound, safety_bound)

        flow_a_final = x_t[:, 0:3]
        flow_b_final = x_t[:, 3:6]
        return flow_a_final, flow_b_final, fine_a, fine_b

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None, return_flow=False):
        if self.use_quality_cond and quality is None:
            raise ValueError("use_quality_cond=True mas `quality` nao foi passado ao forward")
        flow_a_final, flow_b_final, fine_a, fine_b = self.sample_flow(
            vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=quality)

        vis_in = torch.cat([fine_a, fine_b, flow_a_final, flow_b_final], dim=1)
        vis_logit = self.vis_out(self.vis_head(vis_in))
        vis = torch.sigmoid(vis_logit)

        warped_a = warp3d(vol_a, flow_a_final)
        warped_b = warp3d(vol_b, flow_b_final)
        t_map = t.view(-1, 1, 1, 1, 1)
        # MESMA formula de blend de model.rrin3d.RRIN3D.forward / model.amt3d.AMT3D.forward.
        w_a = (1.0 - t_map) * vis
        w_b = t_map * (1.0 - vis)
        denom = (w_a + w_b).clamp(min=1e-6)
        blend = (w_a * warped_a + w_b * warped_b) / denom

        residual = self.refine_net(blend, vol_a, vol_b)
        pred = blend + residual
        if not return_flow:
            return pred
        return pred, flow_a_final, flow_b_final, vis_logit


def build_hfd_model(base_ch: int = 16, max_disp: float = 0.5, corr_radius: int = 3,
                     use_quality_cond: bool = False, norm_type: str = "instance",
                     num_timesteps: int = 1000, num_sample_steps: int = 6,
                     beta_start: float = 1e-4, beta_end: float = 0.02,
                     time_emb_dim: int = 32) -> HFD3D:
    """Dispatcher, mesmo espirito de model.rrin3d.build_rrin_model e
    model.amt3d.build_amt_model -- usar esta funcao em
    scripts/04d_train_hfd.py e scripts/05e_reconstruct_hfd.py em vez de
    instanciar HFD3D diretamente."""
    return HFD3D(base_ch=base_ch, max_disp=max_disp, corr_radius=corr_radius,
                 use_quality_cond=use_quality_cond, norm_type=norm_type,
                 num_timesteps=num_timesteps, num_sample_steps=num_sample_steps,
                 beta_start=beta_start, beta_end=beta_end, time_emb_dim=time_emb_dim)


def _smoke_test():
    """Forward pass e diffusion_loss com tensores pequenos aleatorios, so
    pra checar shapes -- mesmo padrao de model/rcae.py, model/rrin3d.py e
    model/amt3d.py. Nao executavel neste ambiente de desenvolvimento (sem
    PyTorch) -- rodar no cluster: python -m model.hfd3d"""
    torch.manual_seed(0)
    b, d, h, w = 2, 10, 10, 10
    vol_a = torch.rand(b, 1, d, h, w)
    vol_b = torch.rand(b, 1, d, h, w)
    bvec_a = torch.randn(b, 3); bvec_a = bvec_a / bvec_a.norm(dim=-1, keepdim=True)
    bvec_b = torch.randn(b, 3); bvec_b = bvec_b / bvec_b.norm(dim=-1, keepdim=True)
    bvec_t = torch.randn(b, 3); bvec_t = bvec_t / bvec_t.norm(dim=-1, keepdim=True)
    t = torch.rand(b)
    expected = (b, 1, d, h, w)

    # num_sample_steps pequeno so pra velocidade do smoke test -- nao afeta
    # nenhum shape/peso (ver docstring de HFD3D), so custo de amostragem.
    model = build_hfd_model(base_ch=8, corr_radius=2, num_timesteps=1000, num_sample_steps=2)

    # --- diffusion_loss (treino) ---
    target_flow_a = torch.randn(b, 3, d, h, w) * 0.1
    target_flow_b = torch.randn(b, 3, d, h, w) * 0.1
    loss = model.diffusion_loss(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t,
                                 target_flow_a, target_flow_b)
    assert loss.dim() == 0, f"diffusion_loss deveria ser escalar, veio shape {loss.shape}"
    assert torch.isfinite(loss), "diffusion_loss nao-finita no smoke test"
    print(f"smoke test OK (diffusion_loss), valor: {loss.item():.6f}")

    # --- forward (inferencia/amostragem) ---
    out = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
    assert out.shape == expected, f"shape mismatch (forward): {out.shape} != {expected}"
    assert torch.isfinite(out).all(), "forward produziu valores nao-finitos no smoke test"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"smoke test OK (forward), output shape: {tuple(out.shape)}, {n_params} parametros")

    out_rf, flow_a_f, flow_b_f, vis_logit_f = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t,
                                                     return_flow=True)
    assert out_rf.shape == expected
    assert flow_a_f.shape == (b, 3, d, h, w), f"flow_a_final shape errado: {flow_a_f.shape}"
    assert flow_b_f.shape == (b, 3, d, h, w), f"flow_b_final shape errado: {flow_b_f.shape}"
    assert vis_logit_f.shape == (b, 1, d, h, w), f"vis_logit shape errado: {vis_logit_f.shape}"
    print("OK: return_flow=True retorna (pred, flow_a, flow_b, vis_logit) com shapes corretos")

    # --- use_quality_cond ---
    model_q = build_hfd_model(base_ch=8, corr_radius=2, num_sample_steps=2, use_quality_cond=True)
    quality = torch.rand(b, 2)
    out_q = model_q(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=quality)
    assert out_q.shape == expected, f"shape mismatch (com quality): {out_q.shape} != {expected}"
    loss_q = model_q.diffusion_loss(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t,
                                     target_flow_a, target_flow_b, quality=quality)
    assert torch.isfinite(loss_q)
    print(f"smoke test OK (use_quality_cond=True), output shape: {tuple(out_q.shape)}")

    try:
        model_q(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
        raise AssertionError("deveria ter levantado ValueError sem `quality`")
    except ValueError:
        print("OK: chamar sem `quality` com use_quality_cond=True levanta ValueError, como esperado")

    # --- varias combinacoes de corr_radius/num_sample_steps/norm_type ---
    for radius in (1, 3):
        for k_steps in (1, 4):
            model_k = build_hfd_model(base_ch=8, corr_radius=radius, num_sample_steps=k_steps)
            out_k = model_k(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
            assert out_k.shape == expected, \
                f"shape mismatch (corr_radius={radius}, num_sample_steps={k_steps}): " \
                f"{out_k.shape} != {expected}"
            print(f"smoke test OK (corr_radius={radius}, num_sample_steps={k_steps}), "
                  f"output shape: {tuple(out_k.shape)}")

    model_bn = build_hfd_model(base_ch=8, corr_radius=2, num_sample_steps=2, norm_type="batch")
    assert isinstance(model_bn.encoder.fine1[1], nn.BatchNorm3d)
    out_bn = model_bn(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
    assert out_bn.shape == expected
    print(f"smoke test OK (norm_type=batch), output shape: {tuple(out_bn.shape)}")

    # --- utilitarios de difusao isolados (conferencia direta das formulas) ---
    betas = make_beta_schedule(1000)
    assert betas.shape == (1000,)
    assert abs(betas[0].item() - 1e-4) < 1e-9 and abs(betas[-1].item() - 0.02) < 1e-9
    alphas, alpha_bars = diffusion_constants(betas)
    assert alpha_bars[0].item() < 1.0 and alpha_bars[0].item() > 0.999  # quase sem ruido
    assert alpha_bars[-1].item() < 0.01  # quase ruido puro
    ts = ddim_timesteps(1000, 6)
    assert ts[0] == 999 and ts[-1] == 0 and len(ts) == 6 and ts == sorted(ts, reverse=True)
    assert ddim_timesteps(1000, 1) == [999]
    # passo terminal (alpha_bar_prev=1.0) deve retornar x_prev == x0_pred exatamente
    x_dummy = torch.randn(2, 6, 4, 4, 4)
    eps_dummy = torch.randn(2, 6, 4, 4, 4)
    x_prev, x0_pred = ddim_step(x_dummy, eps_dummy, alpha_bar_t=0.3, alpha_bar_t_prev=1.0)
    assert torch.allclose(x_prev, x0_pred), "passo terminal (alpha_bar_prev=1.0) deveria dar x_prev==x0_pred"
    print("OK: utilitarios de difusao (make_beta_schedule/diffusion_constants/"
          "ddim_timesteps/ddim_step) conferidos")


if __name__ == "__main__":
    _smoke_test()