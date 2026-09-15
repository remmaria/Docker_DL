"""
Representacao angular IMPLICITA e continua para super-resolucao angular em
dMRI -- linha de modelo NOVA, independente do RCAE (model/rcae.py) e do RRIN
(model/rrin3d.py/rrin3d_star.py), nascida da discussao "pq nao implementamos
entao?" sobre paralelos com sintese de vista em imagem natural (ver
addendum, secao 20.10/20.11).

MOTIVACAO (nao repetir isto em cada docstring de classe abaixo -- ver aqui
a ideia central uma vez so):

Toda a linha RRIN/AMT/HFD/estrela trata a super-resolucao angular como um
problema de FLUXO OTICO/correspondencia entre um PAR (ou pequeno feixe) de
direcoes medidas "vizinhas" -- generalizacao direta de VFI (video frame
interpolation) de imagem natural. O diagnostico acumulado ao longo desta
tese (varias secoes do addendum, ver 20.6/20.9) e que essa suposicao
estrutural (correspondencia local entre pares) e o proprio gargalo: mesmo
com arquiteturas mais sofisticadas (AMT3D ~ RRIN3D), mais contexto espacial
(baseline_sh sem NENHUM contexto espacial ja supera as duas), ou ensembles
maiores (M), o ganho estagna -- porque em muitas regioes da esfera (n_level
baixo) NAO EXISTE um par proximo o bastante para "fluxo" fazer sentido
fisicamente (gap_deg entre os pares escolhidos facilmente passa de 50-70
graus, bem longe da suposicao de deslocamento pequeno que sustenta o
warping).

Em imagem natural, o problema analogo -- sintetizar uma vista nova a partir
de vistas esparsas e MUITO distantes entre si, onde correspondencia para de
fazer sentido -- deixou de ser atacado por fluxo otico entre pares ha um
tempo, migrando para REPRESENTACOES IMPLICITAS CONTINUAS condicionadas em
coordenada/direcao:
  - NeRF (Mildenhall et al., ECCV 2020): a cor/densidade de um ponto e uma
    funcao continua aprendida da SUA PROPRIA coordenada 3D + direcao de
    visão -- nunca uma correspondencia entre duas imagens.
  - LIIF -- Local Implicit Image Function (Chen et al., CVPR 2021): uma
    rede aprende f(features_locais_de_uma_CNN, coordenada_alvo) -> valor,
    combinando contexto espacial (encoder CNN, UMA vez so, nao por par) com
    uma consulta em coordenada continua, tambem sem nenhum "segundo frame"
    pareado.

Este modulo adapta essa ideia para a esfera de direcoes de gradiente:
  1. Um encoder por-direcao (`PerDirectionEncoder3D`) processa CADA uma das
     n_level direcoes medidas, condicionado na SUA PROPRIA direcao (via
     harmonicos esfericos reais, ver `sh_positional_encoding` abaixo -- a
     MESMA base que `utils/sh_basis.py` ja usa no baseline_sh, entao
     antipodal-simetrica por construcao, sem precisar de correcao manual de
     sinal v/-v como as linhas de pares fazem em `find_best_bracket_batch`/
     `find_star_ensemble_batch`).
  2. As n_level saidas sao agregadas por MEDIA (pooling permutation-
     invariant, estilo DeepSets -- Zaheer et al. 2017) numa unica
     representacao espacial "estado" -- o encoder e chamado (pesos
     compartilhados) uma vez por direcao de entrada, nao uma vez por PAR.
  3. `SpatialTrunk3D` (pequena U-Net 3D de 2 niveis, reaproveitando os
     mesmos blocos genericos `_conv3d`/`_repeat_vec_3d` ja usados por
     `model/rrin3d.py`) refina esse estado com contexto espacial local.
  4. `ImplicitDecoderHead3D` consulta esse estado numa direcao-alvo
     ARBITRARIA (nao precisa ser uma das medidas, nem estar perto de
     nenhuma): concatena o estado com o codigo SH da direcao-alvo e prediz
     o sinal -- sem NUNCA formar um par explicito nem estimar um campo de
     fluxo.

ESCOPO DESTA VERSAO (decisoes explicitas do usuario, ver addendum secao
20.11): "v2 direto" (com CNN espacial completa via `SpatialTrunk3D`, nao um
"v1" mais barato so-por-voxel primeiro) e entrada = SOMENTE o sinal medido
nas n_level direcoes (sem os coeficientes baseline_sh como feature extra).
Modelo construido do ZERO, sem importar nada de `model/rcae.py` (que tem
sobreposicao conceitual real -- encoder->estado->decoder condicionado em
direcao-alvo -- mas foi mantido deliberadamente como baseline comparativo
PURO, nunca uma dependencia desta linha experimental, ver addendum secao
20.8). Reaproveita apenas blocos GENERICOS de `model/rrin3d.py`
(`_conv3d`/`_norm3d`/`_repeat_vec_3d`), ja usados por mais de um modelo
desta linha (rrin3d.py e rrin3d_star.py), portanto nao especificos do RCAE.

AGREGACAO -- `aggregation="mean"` (default, comportamento ORIGINAL desta
linha, sem nenhuma mudanca) ou `aggregation="attention"` (ADITIVO, ver
addendum 2026-09-03, secao 29): motivado pelo diagnostico de
`scripts/14_diagnose_implicit_pooling.py` (secao 28.1) -- num checkpoint
real (epoca 15), as n_level saidas de `PerDirectionEncoder3D` ja divergem
bastante entre si ANTES da agregacao ("colapso de agregacao" alto, ~0,54),
mas a sensibilidade de remover qualquer direcao da MEDIA e' so' moderada
(~0,30) -- leitura mista, nem forte a favor nem contra trocar a media por
algo aprendido, mas material suficiente pra tentar. `AttentionAggregator3D`
(abaixo) substitui a media simples por uma media PONDERADA, com pesos
aprendidos POR VOXEL (nao um peso global por direcao) via um pequeno "gated
attention pooling" (mesmo espirito de Ilse et al. 2018, e do
`PairWeightHead3D` que `model/rrin3d_star.py` ja usa pra fundir candidatos
do feixe em estrela -- aqui fundindo DIRECOES DE ENTRADA em vez de PARES).
Critico: o escore de cada direcao e' uma funcao SO da propria feature dela
(pesos compartilhados entre direcoes, nenhuma nocao de posicao/ordem) --
permutar a ordem das n_level direcoes de entrada permuta escores e
features JUNTOS, entao a soma ponderada final nao muda -- a propriedade de
permutation-invariance da secao anterior (motivada por DeepSets) e'
preservada por construcao, nao um acidente; ver teste de permutacao para
"attention" em `_smoke_test`, espelhando o teste ja existente pra "mean".

Requer PyTorch (nao disponivel neste ambiente de desenvolvimento -- revisado
manualmente, testado apenas por compilacao de sintaxe; validar no cluster
com `python -m model.implicit_angular`, smoke test no fim do arquivo, mesmo
padrao de model/rcae.py e model/rrin3d.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rrin3d import _conv3d, _repeat_vec_3d
from utils.sh_basis import real_sh_matrix, cart2sphere, max_order_for_n_directions


def sh_dim_for_lmax(l_max: int) -> int:
    """Numero de coeficientes SH (pares, l=0,2,4,...,l_max) -- mesma formula
    de `utils/sh_basis.py:max_order_for_n_directions`, so invertida (dado
    l_max, quantas colunas `real_sh_matrix` produz). Usado para fixar o
    numero de canais de entrada dos blocos condicionados em direcao ANTES
    de construir a rede (o l_max e escolhido uma vez, na construcao do
    modelo -- ver `ImplicitAngularModel3D.__init__` -- e nao pode variar
    por chamada de forward, ja que muda o shape dos pesos)."""
    return (l_max + 1) * (l_max + 2) // 2


def sh_positional_encoding(bvecs: torch.Tensor, l_max: int) -> torch.Tensor:
    """Codificacao posicional de uma direcao de gradiente via harmonicos
    esfericos reais (`utils/sh_basis.py:real_sh_matrix`), o analogo, para a
    esfera de direcoes, da codificacao posicional de coordenada usada em
    NeRF/LIIF (la, funcoes seno/cosseno de frequencia crescente da
    coordenada; aqui, harmonicos esfericos de ordem crescente da direcao --
    a base natural para funcoes definidas na esfera, ja usada pelo proprio
    baseline_sh deste projeto).

    bvecs: (..., 3) tensor de vetores unitarios de gradiente (qualquer
        numero de dimensoes de batch antes do ultimo eixo).
    l_max: ordem par maxima (fixa por chamada -- ver `sh_dim_for_lmax`).
    Retorna: (..., sh_dim_for_lmax(l_max)) tensor, MESMO device/dtype de
        `bvecs`.

    Implementado via um round-trip por numpy (`real_sh_matrix`/
    `cart2sphere` ja existem e sao testados isoladamente em
    utils/sh_basis.py -- reimplementar a matematica de harmonicos esfericos
    em torch so duplicaria codigo). Isso e aceitavel aqui porque bvecs NUNCA
    precisam de gradiente (sao constantes de geometria da aquisicao, nao
    parametros aprendidos nem uma funcao diferenciavel de nada que se
    otimize) -- o roundtrip CPU e barato (poucas dezenas de vetores por
    batch) e roda uma vez por forward, nao por epoca."""
    orig_shape = bvecs.shape[:-1]
    device, dtype = bvecs.device, bvecs.dtype
    flat = bvecs.detach().reshape(-1, 3).cpu().numpy().astype(np.float64)
    theta, phi = cart2sphere(flat)
    B = real_sh_matrix(theta, phi, l_max)  # (N, sh_dim), numpy float64
    out = torch.from_numpy(B.astype(np.float32)).to(device=device, dtype=dtype)
    return out.reshape(*orig_shape, -1)


class PerDirectionEncoder3D(nn.Module):
    """Encoder de PESOS COMPARTILHADOS aplicado independentemente a cada uma
    das n_level direcoes medidas -- nunca a um par. Cada chamada ve APENAS:
    (a) o patch de sinal medido nessa UNICA direcao, (b) o codigo SH da
    PROPRIA direcao dessa medida (concatenado, broadcast espacial, mesma
    ideia de `_repeat_vec_3d`/`_repeat_bvec` ja usada em rrin3d.py/rcae.py).

    Por nao ver a direcao-alvo nem nenhuma outra direcao de entrada, a saida
    desta rede e uma funcao SO da geometria/sinal daquela direcao isolada --
    a informacao relativa entre direcoes so entra depois, na agregacao por
    media (ver `ImplicitAngularModel3D.forward`) e no `SpatialTrunk3D`."""

    def __init__(self, sh_dim: int, base_ch: int = 16, norm_type: str = "instance"):
        super().__init__()
        in_ch = 1 + sh_dim  # sinal medido (1 canal) + codigo SH da direcao
        self.net = nn.Sequential(
            _conv3d(in_ch, base_ch, norm_type=norm_type),
            _conv3d(base_ch, base_ch, norm_type=norm_type),
        )

    def forward(self, vol: torch.Tensor, sh_code: torch.Tensor) -> torch.Tensor:
        """vol: (N, 1, D, H, W) -- N = B*n_level (achatado, ver chamada em
        ImplicitAngularModel3D.forward). sh_code: (N, sh_dim). Retorna
        (N, base_ch, D, H, W)."""
        spatial = vol.shape[-3:]
        code_map = _repeat_vec_3d(sh_code, spatial)
        x = torch.cat([vol, code_map], dim=1)
        return self.net(x)


class AttentionAggregator3D(nn.Module):
    """Agregacao alternativa a media simples de `ImplicitAngularModel3D.
    encode` -- "gated attention pooling" adaptado a 3D+espaco (mesmo
    espirito de Ilse et al. 2018, e do `PairWeightHead3D` que
    `model/rrin3d_star.py` ja usa pra fundir candidatos do feixe em
    estrela -- aqui fundindo DIRECOES DE ENTRADA, nao pares). Ver docstring
    do modulo (secao "AGREGACAO") para a motivacao completa.

    Calcula um escore ESPACIAL (por voxel, nao um escalar global por
    direcao) pra cada uma das n_level direcoes, de pesos COMPARTILHADOS
    (mesma rede aplicada independentemente a cada direcao, igual ao
    `PerDirectionEncoder3D` -- nenhuma direcao ve as demais nesta etapa),
    normaliza por softmax ENTRE AS DIRECOES (nao entre voxels) e agrega por
    media ponderada. PERMUTATION-INVARIANT por construcao: o escore de
    cada direcao depende so' da propria feature dela, entao permutar a
    ordem de entrada permuta escores e features juntos, sem mudar a soma
    ponderada final -- ver teste em `_smoke_test`."""

    def __init__(self, in_ch: int, hidden_ch: int | None = None, norm_type: str = "instance"):
        super().__init__()
        hidden_ch = hidden_ch if hidden_ch is not None else max(in_ch // 2, 4)
        self.net = nn.Sequential(
            _conv3d(in_ch, hidden_ch, norm_type=norm_type),
            nn.Conv3d(hidden_ch, 1, kernel_size=3, padding=1),
        )

    def forward(self, feat: torch.Tensor):
        """feat: (B, n_level, C, D, H, W). Retorna (state, alpha):
        state: (B, C, D, H, W) -- media ponderada entre as n_level
            direcoes (substitui `feat.mean(dim=1)` do caminho "mean").
        alpha: (B, n_level, 1, D, H, W) -- pesos de atencao (softmax sobre
            o eixo das n_level direcoes, POR VOXEL), devolvido pra
            diagnostico/inspecao futura (mesmo espirito de `extra["pi"]`
            em `model/rrin3d_star.py:RRIN3DStar.forward(return_pairs=True)`)."""
        b, n_level, c = feat.shape[0], feat.shape[1], feat.shape[2]
        spatial = feat.shape[-3:]
        feat_flat = feat.reshape(b * n_level, c, *spatial)
        score_flat = self.net(feat_flat)                        # (B*n_level, 1, D,H,W)
        score = score_flat.reshape(b, n_level, 1, *spatial)
        alpha = torch.softmax(score, dim=1)                      # softmax ENTRE DIRECOES
        state = (alpha * feat).sum(dim=1)                        # (B, C, D,H,W)
        return state, alpha


class CrossDirectionAttention3D(nn.Module):
    """Bloco de self-attention ENTRE as n_level direcoes de entrada, aplicado
    POR VOXEL (pesos compartilhados no espaco, mesmo espirito dos blocos
    conv3d desta linha) -- ADITIVO (item 2 da discussao 2026-09-13 sobre por
    que o RCAE bate o `implicit` em producao: `PerDirectionEncoder3D`
    processa cada direcao de forma totalmente independente, e a agregacao
    por media/`AttentionAggregator3D` tambem so calcula um escore POR
    direcao isoladamente -- nenhuma direcao "ve" as outras antes da
    agregacao. Esse bloco insere justamente essa interacao FALTANTE, logo
    apos `PerDirectionEncoder3D` e ANTES de qualquer agregacao (ver
    `ImplicitAngularModel3D.encode`) -- mesmo principio geral do "Set
    Transformer" (Lee et al., ICML 2019): deixar os elementos do CONJUNTO
    trocarem informacao entre si antes de qualquer pooling.

    Motivacao empirica direta: um treino de producao real (M8/res15,
    BATCH_SIZE=32/LR=4e-3, AGGREGATION=attention, INIT_OUTPUT_BIAS_FROM_DATA
    + WARMUP_STEPS ja ativos) saturou em val_loss~0,041-0,042 ja na epoca 3,
    bem acima do platô do RCAE/RRIN3DStar (~0,028-0,030) -- mesmo com
    aggregation="attention" (que ja aprende um peso por voxel/direcao) o
    modelo bate um teto cedo, sinal de que o problema nao e' "que tipo de
    pooling" e sim a falta de interacao ENTRE direcoes antes de qualquer
    pooling.

    Por que self-attention COMPLETA (SAB) em vez da versao "induzida" (ISAB,
    tambem do paper do Set Transformer, O(n) em vez de O(n^2)): aqui
    n_level e' pequeno (tipicamente <=54 direcoes) -- o custo O(n^2) de uma
    SAB comum e' desprezivel comparado ao resto do modelo (a dimensao
    realmente grande e' o espaco: B*D*H*W voxels tratados como "batch"
    desta atencao, nao o eixo das direcoes). ISAB so' compensaria a
    complexidade extra pra n_level grande (centenas+), o que nao e' o caso
    deste dataset.

    PERMUTATION-EQUIVARIANTE por construcao (nao invariante -- a invariancia
    so' vem DEPOIS, na agregacao por media/`AttentionAggregator3D`, que ja
    consome a saida deste bloco): self-attention nao usa nenhuma codificacao
    de POSICAO na sequencia (so' o conteudo de cada direcao, que ja carrega
    seu proprio codigo SH via `PerDirectionEncoder3D`) -- permutar a ordem
    das n_level direcoes de entrada permuta a saida na MESMA ordem, sem
    mudar nenhum valor individual (ver teste de permutacao em
    `_smoke_test`)."""

    def __init__(self, channels: int, num_heads: int = 4, ff_mult: int = 2):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) precisa ser divisivel por num_heads ({num_heads}) -- "
                f"ver --base-ch/--cross-attn-heads em scripts/04f_train_implicit.py.")
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

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, n_level, C, D, H, W). Retorna o MESMO shape, com cada
        direcao atualizada por atencao sobre as demais (por voxel)."""
        b, n_level, c, d, h, w = feat.shape
        # (B,n_level,C,D,H,W) -> (B,D,H,W,n_level,C) -> (B*D*H*W, n_level, C):
        # trata cada posicao espacial como um item de "batch" independente da
        # atencao (pesos compartilhados no espaco, mesmo espirito dos blocos
        # conv3d desta linha -- so' que aqui a "convolucao" e' sobre o eixo
        # das direcoes, nao sobre voxels vizinhos).
        x = feat.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, n_level, c)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x.reshape(b, d, h, w, n_level, c).permute(0, 4, 5, 1, 2, 3)


class SpatialTrunk3D(nn.Module):
    """Pequena U-Net 3D (2 niveis de downsample), MESMA topologia de
    `FlowNet3D.enc1/enc2/enc3/dec2/dec1/head` em model/rrin3d.py -- reusada
    aqui como um refinador de contexto espacial local sobre o "estado"
    agregado (media entre direcoes de `PerDirectionEncoder3D`), em vez de
    predizer fluxo. Generica o bastante (so recebe/devolve um tensor de
    canais fixos) para nao precisar duplicar a classe -- apenas os blocos
    `_conv3d`/`_repeat_vec_3d` (genericos, ja compartilhados entre
    rrin3d.py e rrin3d_star.py) sao reaproveitados; nenhuma classe de
    rrin3d.py e importada ou instanciada aqui."""

    def __init__(self, in_ch: int, base_ch: int = 16, norm_type: str = "instance"):
        super().__init__()
        self.enc1 = _conv3d(in_ch, base_ch, norm_type=norm_type)
        self.enc2 = _conv3d(base_ch, base_ch * 2, stride=2, norm_type=norm_type)
        self.enc3 = _conv3d(base_ch * 2, base_ch * 4, stride=2, norm_type=norm_type)
        self.dec2 = _conv3d(base_ch * 4, base_ch * 2, norm_type=norm_type)
        self.dec1 = _conv3d(base_ch * 2 + base_ch * 2, base_ch, norm_type=norm_type)
        self.head = _conv3d(base_ch + base_ch, base_ch, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d2 = self.dec2(e3)
        d2 = F.interpolate(d2, size=e2.shape[-3:], mode="trilinear", align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = F.interpolate(d1, size=e1.shape[-3:], mode="trilinear", align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)

        return self.head(d1)  # (B, base_ch, D, H, W) -- o "estado" espacial


class ImplicitDecoderHead3D(nn.Module):
    """Decoder implicito: consulta o "estado" espacial numa direcao-alvo
    ARBITRARIA (nunca precisa coincidir com uma direcao medida nem estar
    perto de nenhuma) -- concatena o estado com o codigo SH da direcao-alvo
    (broadcast espacial) e prediz o sinal nessa direcao. Analogo direto do
    "query em coordenada continua" do LIIF (aqui a "coordenada" e uma
    direcao na esfera, nao um pixel 2D).

    ATUALIZACAO (2026-09-14, ver addendum de capacidade): ganhou `depth` e
    `reinject_code`, ADITIVOS. Com os defaults (`depth=1`,
    `reinject_code=False`) a topologia, os NOMES dos parametros (`net.0.*`,
    `net.1.*` -- `nn.ModuleList` registra os filhos pelo indice exatamente
    como `nn.Sequential`) e a contagem de parametros ficam IDENTICOS a
    versao anterior, entao checkpoints ja treinados continuam carregando.

    Motivacao: esta e' a rede que representa a funcao continua sobre a
    esfera -- e' o PROPOSITO do modelo implicito inteiro. Com `base_ch=16`
    ela tinha 13.873 parametros e DUAS convolucoes (uma oculta + a de
    saida), contra 3.723.497 parametros e 10 convolucoes do decoder do
    RCAE (268x mais), que ainda por cima reinjeta o bvec do alvo em TODOS
    os estagios (`model/rcae.py:RepeatBVector`). Aqui o codigo SH da
    direcao-alvo entrava uma unica vez, na concatenacao de entrada, e
    precisava sobreviver a toda a rede a partir dali -- `reinject_code=True`
    corrige isso, seguindo tanto o RCAE quanto a pratica padrao de
    representacoes implicitas (LIIF/NeRF injetam a coordenada repetidamente
    ao longo do MLP, nao so' na primeira camada)."""

    def __init__(self, state_ch: int, sh_dim: int, base_ch: int = 16,
                 norm_type: str = "instance", depth: int = 1,
                 reinject_code: bool = False):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth de ImplicitDecoderHead3D deve ser >= 1 (recebido {depth})")
        self.depth = depth
        self.reinject_code = reinject_code
        self.sh_dim = sh_dim
        in_ch = state_ch + sh_dim
        hidden_in = base_ch + (sh_dim if reinject_code else 0)
        layers = [_conv3d(in_ch, base_ch, norm_type=norm_type)]
        for _ in range(depth - 1):
            layers.append(_conv3d(hidden_in, base_ch, norm_type=norm_type))
        layers.append(nn.Conv3d(base_ch, 1, kernel_size=3, padding=1))
        # ModuleList (nao Sequential) porque o forward precisa reconcatenar o
        # codigo SH entre as camadas quando reinject_code=True. Os nomes dos
        # parametros no state_dict sao os mesmos de um Sequential equivalente.
        self.net = nn.ModuleList(layers)

    def forward(self, state: torch.Tensor, sh_code: torch.Tensor) -> torch.Tensor:
        """state: (N, state_ch, D, H, W) -- N = B*N_out (ja repetido, ver
        ImplicitAngularModel3D.decode). sh_code: (N, sh_dim). Retorna
        (N, 1, D, H, W)."""
        spatial = state.shape[-3:]
        code_map = _repeat_vec_3d(sh_code, spatial)
        x = torch.cat([state, code_map], dim=1)
        for i, layer in enumerate(self.net):
            if self.reinject_code and 0 < i < len(self.net) - 1:
                x = torch.cat([x, code_map], dim=1)
            x = layer(x)
        return x


class ImplicitAngularModel3D(nn.Module):
    """Modelo completo: PerDirectionEncoder3D (pesos compartilhados, uma
    chamada por direcao medida) -> media entre direcoes (agregacao
    permutation-invariant, estilo DeepSets) -> SpatialTrunk3D (contexto
    espacial local) -> ImplicitDecoderHead3D (consulta em direcao-alvo
    continua). Ver docstring do modulo para a motivacao completa.

    l_max: ordem par maxima da base SH usada tanto para codificar as
        direcoes de ENTRADA quanto as direcoes-ALVO (mesma base para as
        duas, mesmo espirito do baseline_sh -- a rede nao precisa de bases
        diferentes para "olhar" e para "consultar"). FIXO na construcao
        (muda o numero de canais de entrada dos blocos condicionados em
        direcao -- nao pode variar por chamada de forward). Se None
        (default), usa `max_order_for_n_directions(n_level)` -- amarra a
        resolucao angular da representacao a quantas direcoes sao
        realmente medidas, mesma convencao do baseline_sh.

    aggregation: "mean" (default, comportamento ORIGINAL desta linha) ou
        "attention" (ADITIVO, ver `AttentionAggregator3D`/docstring do
        modulo, secao "AGREGACAO", e addendum 2026-09-03 secao 29) -- troca
        a media simples entre as n_level direcoes por uma media ponderada
        aprendida, com pesos POR VOXEL. FIXO na construcao (adiciona
        parametros novos ao modelo quando "attention" -- nao pode mudar em
        --resume-checkpoint, mesma logica de l_max/base_ch/norm_type, ver
        scripts/04f_train_implicit.py).

    cross_direction_attention: False (default, comportamento ORIGINAL) ou
        True (ADITIVO, ver `CrossDirectionAttention3D` e addendum
        2026-09-13, item 2) -- insere um bloco de self-attention ENTRE as
        n_level direcoes de entrada logo apos `PerDirectionEncoder3D` e
        ANTES de qualquer agregacao (mean/attention) -- ataca a falta de
        interacao entre direcoes que nem "attention" (que so pondera
        direcoes ja calculadas de forma independente) resolve. Composa com
        QUALQUER `aggregation` (mean ou attention) -- sao etapas
        independentes do pipeline. FIXO na construcao (adiciona parametros
        novos -- nao pode mudar em --resume-checkpoint).
    cross_attn_heads: numero de cabecas da `CrossDirectionAttention3D`
        (default 4). So tem efeito se cross_direction_attention=True. Precisa
        dividir `base_ch` (embed_dim da atencao) sem resto.

    Uso:
        model = build_implicit_model(n_level=16)
        pred = model(input_vols, input_bvecs, target_bvecs)
    input_vols: (B, n_level, 1, D, H, W) -- sinal medido em CADA direcao de
        entrada (nao pares -- todas as n_level direcoes de uma vez, mesmo
        shape que utils/dataset.py:DWIPatchDataset ja produz).
    input_bvecs: (B, n_level, 3).
    target_bvecs: (B, N_out, 3) -- N_out direcoes-alvo, QUALQUER N_out
        (nao precisa ser fixo nem bater com n_level).
    retorna: (B, N_out, 1, D, H, W).
    """

    def __init__(self, n_level: int, l_max: int | None = None, base_ch: int = 16,
                 norm_type: str = "instance", aggregation: str = "mean",
                 cross_direction_attention: bool = False, cross_attn_heads: int = 4,
                 decoder_base_ch: int = None, decoder_depth: int = 1,
                 decoder_reinject_code: bool = False):
        super().__init__()
        if aggregation not in ("mean", "attention"):
            raise ValueError(f"aggregation deve ser 'mean' ou 'attention', recebi {aggregation!r}")
        self.n_level = n_level
        self.l_max = l_max if l_max is not None else max_order_for_n_directions(n_level)
        self.sh_dim = sh_dim_for_lmax(self.l_max)
        self.base_ch = base_ch
        self.norm_type = norm_type
        self.aggregation = aggregation
        self.cross_direction_attention = cross_direction_attention
        self.cross_attn_heads = cross_attn_heads

        self.per_dir_encoder = PerDirectionEncoder3D(self.sh_dim, base_ch=base_ch,
                                                       norm_type=norm_type)
        self.cross_attn = (CrossDirectionAttention3D(base_ch, num_heads=cross_attn_heads)
                            if cross_direction_attention else None)
        self.attn_aggregator = (AttentionAggregator3D(base_ch, norm_type=norm_type)
                                 if aggregation == "attention" else None)
        self.trunk = SpatialTrunk3D(base_ch, base_ch=base_ch, norm_type=norm_type)
        # decoder_base_ch=None => segue base_ch (comportamento historico).
        self.decoder_base_ch = decoder_base_ch if decoder_base_ch is not None else base_ch
        self.decoder_depth = decoder_depth
        self.decoder_reinject_code = decoder_reinject_code
        self.decoder_head = ImplicitDecoderHead3D(base_ch, self.sh_dim,
                                                   base_ch=self.decoder_base_ch,
                                                   depth=decoder_depth,
                                                   reinject_code=decoder_reinject_code,
                                                    norm_type=norm_type)

    def encode(self, input_vols: torch.Tensor, input_bvecs: torch.Tensor,
               return_attn: bool = False):
        """input_vols: (B, n_level, 1, D, H, W); input_bvecs: (B, n_level, 3).
        Retorna o "estado" espacial (B, base_ch, D, H, W) -- ANTES de
        condicionar em qualquer direcao-alvo (mesmo papel do `state` de
        RCAE.encoder(...), ver model/rcae.py e scripts/04_train_rcae.py --
        so por isso o metodo se chama `encode`, para os scripts de
        treino/debug poderem reaproveitar o MESMO padrao de plotar
        input/target/pred/contexto que ja usam para o RCAE, sem duplicar a
        logica de visualizacao).

        return_attn (default False, ADITIVO -- todo chamador existente
        continua recebendo so' o tensor `state`, comportamento identico ao
        de antes do aggregation="attention" existir): quando True, retorna
        `(state, alpha)`, com `alpha=None` no caminho "mean" (nao ha' peso
        nenhum pra devolver) ou o tensor de pesos (B, n_level, 1, D, H, W)
        no caminho "attention" -- ver `AttentionAggregator3D.forward`."""
        b, n_level = input_vols.shape[0], input_vols.shape[1]
        spatial = input_vols.shape[-3:]

        sh_flat = sh_positional_encoding(
            input_bvecs.reshape(b * n_level, 3), self.l_max)          # (B*n_level, sh_dim)
        vols_flat = input_vols.reshape(b * n_level, 1, *spatial)       # (B*n_level, 1, D,H,W)
        feat_flat = self.per_dir_encoder(vols_flat, sh_flat)           # (B*n_level, base_ch, D,H,W)
        feat = feat_flat.reshape(b, n_level, self.base_ch, *spatial)

        if self.cross_attn is not None:
            # interacao ENTRE direcoes (item 2, addendum 2026-09-13) --
            # atualiza cada feature por-direcao com base nas demais, ANTES
            # de qualquer agregacao (equivariante a permutacao, ver
            # docstring de CrossDirectionAttention3D).
            feat = self.cross_attn(feat)

        if self.aggregation == "mean":
            # agregacao PERMUTATION-INVARIANT (media simples, estilo DeepSets
            # -- Zaheer et al. 2017): a ORDEM das n_level direcoes de entrada
            # nunca deveria importar (nao ha nenhuma nocao de
            # "primeira"/"ultima" direcao medida, ao contrario de uma
            # sequencia de video) -- media (ou soma/max) sao as agregacoes
            # canonicas que garantem isso por construcao. Deliberadamente NAO
            # uma ConvLSTM3D (como o encoder do RCAE usa, ver
            # model/rcae.py:ConvLSTM3D) -- alem de manter esta linha
            # independente do RCAE (decisao explicita do usuario, ver
            # addendum secao 20.8), uma LSTM processa a sequencia em ORDEM,
            # deixando de ser estritamente permutation-invariant sem
            # embaralhar a ordem de entrada a cada epoca como paliativo.
            agg = feat.mean(dim=1)                                      # (B, base_ch, D,H,W)
            alpha = None
        else:
            # agregacao aprendida, TAMBEM permutation-invariant por
            # construcao (ver docstring de AttentionAggregator3D/secao
            # "AGREGACAO" do modulo) -- so troca COMO as n_level direcoes sao
            # ponderadas, nao introduz nenhuma nocao de ordem.
            agg, alpha = self.attn_aggregator(feat)                     # (B, base_ch, D,H,W)

        state = self.trunk(agg)                                         # (B, base_ch, D,H,W)
        if return_attn:
            return state, alpha
        return state

    def decode(self, state: torch.Tensor, target_bvecs: torch.Tensor) -> torch.Tensor:
        """state: (B, base_ch, D, H, W) -- saida de `encode`. target_bvecs:
        (B, N_out, 3), N_out arbitrario. Retorna (B, N_out, 1, D, H, W)."""
        b, n_out = target_bvecs.shape[0], target_bvecs.shape[1]
        spatial = state.shape[-3:]

        sh_flat = sh_positional_encoding(
            target_bvecs.reshape(b * n_out, 3), self.l_max)             # (B*N_out, sh_dim)
        # repete o MESMO estado (ja calculado uma unica vez em encode) para
        # cada uma das N_out consultas -- mesma ideia de `_repeat_state` em
        # model/rcae.py, so que aqui via expand+reshape em vez de um helper
        # dedicado (nao vale a pena importar de rcae.py so por isto, ver
        # docstring do modulo sobre independencia).
        state_rep = state.unsqueeze(1).expand(b, n_out, *state.shape[1:])
        state_flat = state_rep.reshape(b * n_out, *state.shape[1:])     # (B*N_out, base_ch, D,H,W)

        pred_flat = self.decoder_head(state_flat, sh_flat)              # (B*N_out, 1, D,H,W)
        return pred_flat.reshape(b, n_out, 1, *spatial)

    def forward(self, input_vols: torch.Tensor, input_bvecs: torch.Tensor,
                target_bvecs: torch.Tensor) -> torch.Tensor:
        state = self.encode(input_vols, input_bvecs)
        return self.decode(state, target_bvecs)


def build_implicit_model(n_level: int, l_max: int | None = None, base_ch: int = 16,
                          norm_type: str = "instance", aggregation: str = "mean",
                          cross_direction_attention: bool = False,
                          cross_attn_heads: int = 4,
                          decoder_base_ch: int = None, decoder_depth: int = 1,
                          decoder_reinject_code: bool = False) -> ImplicitAngularModel3D:
    """Factory unica -- usar em scripts/04f_train_implicit.py e
    scripts/05i_reconstruct_implicit.py em vez de instanciar
    ImplicitAngularModel3D diretamente, mesmo espirito de
    `build_rrin_model`/`build_star_model` (mantem as duas pontas
    sincronizadas; o checkpoint grava l_max/base_ch/norm_type/aggregation/
    cross_direction_attention/cross_attn_heads em `args`, e a reconstrucao
    le de la)."""
    return ImplicitAngularModel3D(n_level=n_level, l_max=l_max, base_ch=base_ch,
                                   norm_type=norm_type, aggregation=aggregation,
                                   cross_direction_attention=cross_direction_attention,
                                   cross_attn_heads=cross_attn_heads,
                                   decoder_base_ch=decoder_base_ch,
                                   decoder_depth=decoder_depth,
                                   decoder_reinject_code=decoder_reinject_code)


def _smoke_test():
    """Forward pass com tensores pequenos aleatorios, so pra checar shapes
    -- mesmo padrao de model/rcae.py e model/rrin3d.py. Rodar no cluster:
    python -m model.implicit_angular"""
    torch.manual_seed(0)
    b, n_level, n_out, d, h, w = 2, 8, 5, 10, 10, 10

    def _rand_unit_bvecs(n):
        v = torch.randn(b, n, 3)
        return v / v.norm(dim=-1, keepdim=True)

    input_vols = torch.rand(b, n_level, 1, d, h, w)
    input_bvecs = _rand_unit_bvecs(n_level)
    target_bvecs = _rand_unit_bvecs(n_out)
    expected = (b, n_out, 1, d, h, w)

    model = build_implicit_model(n_level=n_level, base_ch=8)
    print(f"l_max automatico para n_level={n_level}: {model.l_max} (sh_dim={model.sh_dim})")
    pred = model(input_vols, input_bvecs, target_bvecs)
    assert pred.shape == expected, f"shape mismatch: {pred.shape} != {expected}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"smoke test OK (l_max automatico), output shape: {tuple(pred.shape)}, "
          f"{n_params} parametros")

    # l_max explicito
    model2 = build_implicit_model(n_level=n_level, l_max=2, base_ch=8)
    assert model2.sh_dim == sh_dim_for_lmax(2) == 6
    pred2 = model2(input_vols, input_bvecs, target_bvecs)
    assert pred2.shape == expected
    print(f"smoke test OK (l_max=2 explicito, sh_dim={model2.sh_dim}), "
          f"output shape: {tuple(pred2.shape)}")

    # encode/decode separados devem bater com forward (mesmo estado
    # reaproveitado, ver docstring de encode/decode -- confere que o
    # "atalho" usado pelo script de treino para plotar o contexto de debug
    # nao diverge silenciosamente do forward completo).
    model.eval()
    with torch.no_grad():
        state = model.encode(input_vols, input_bvecs)
        pred_split = model.decode(state, target_bvecs)
        pred_direct = model(input_vols, input_bvecs, target_bvecs)
        assert torch.allclose(pred_split, pred_direct, atol=1e-6), \
            "encode()+decode() deveria ser identico a forward()"
    print("OK: encode()+decode() == forward() (mesmo estado, sem divergencia)")

    # permutation invariance: embaralhar a ORDEM das n_level direcoes de
    # entrada (junto com seus bvecs) nao deveria mudar a predicao -- prova
    # direta de que a agregacao por media (em vez de uma ConvLSTM que
    # processa em ordem) cumpre a propriedade alegada na docstring de
    # `encode`.
    model.eval()
    perm = torch.randperm(n_level)
    with torch.no_grad():
        pred_orig = model(input_vols, input_bvecs, target_bvecs)
        pred_perm = model(input_vols[:, perm], input_bvecs[:, perm], target_bvecs)
    assert torch.allclose(pred_orig, pred_perm, atol=1e-5), \
        "predicao deveria ser invariante a permutacao da ORDEM das direcoes de entrada"
    print("OK: predicao invariante a permutacao das n_level direcoes de entrada "
          "(agregacao por media e permutation-invariant, como esperado)")

    # N_out diferente de n_level, e diferente de 1, ja testado acima
    # (n_out=5 != n_level=8) -- confirma que nao ha nenhum acoplamento
    # estrutural entre os dois (ao contrario de uma abordagem par-a-par).
    for n_out_alt in (1, 3, 12):
        target_bvecs_alt = _rand_unit_bvecs(n_out_alt)
        pred_alt = model(input_vols, input_bvecs, target_bvecs_alt)
        assert pred_alt.shape == (b, n_out_alt, 1, d, h, w)
    print("OK: N_out arbitrario (1, 3, 12) aceito sem reconstruir o modelo")

    # norm_type="batch" (mesma opcao de rrin3d.py -- resolve o artefato de
    # costura de patch-tiling na reconstrucao por sliding-window)
    import torch.nn as _nn
    model_bn = build_implicit_model(n_level=n_level, base_ch=8, norm_type="batch")
    assert isinstance(model_bn.per_dir_encoder.net[0][1], _nn.BatchNorm3d)
    pred_bn = model_bn(input_vols, input_bvecs, target_bvecs)
    assert pred_bn.shape == expected
    print(f"smoke test OK (norm_type=batch), output shape: {tuple(pred_bn.shape)}")

    # aggregation="attention" (ADITIVO, ver addendum 2026-09-03 secao 29) --
    # mesmos testes de shape/permutation-invariance da media, mais o formato
    # do tensor de pesos `alpha` e a checagem de que ele soma 1 entre
    # direcoes (softmax valido) em cada voxel.
    model_attn = build_implicit_model(n_level=n_level, base_ch=8, aggregation="attention")
    assert model_attn.attn_aggregator is not None
    pred_attn = model_attn(input_vols, input_bvecs, target_bvecs)
    assert pred_attn.shape == expected
    print(f"smoke test OK (aggregation=attention), output shape: {tuple(pred_attn.shape)}")

    model_attn.eval()
    with torch.no_grad():
        state_attn, alpha = model_attn.encode(input_vols, input_bvecs, return_attn=True)
        pred_split_attn = model_attn.decode(state_attn, target_bvecs)
        pred_direct_attn = model_attn(input_vols, input_bvecs, target_bvecs)
        assert torch.allclose(pred_split_attn, pred_direct_attn, atol=1e-6), \
            "encode(return_attn=True)+decode() deveria bater com forward() em aggregation=attention"
        assert alpha.shape == (b, n_level, 1, d, h, w)
        alpha_sum = alpha.sum(dim=1)  # deveria ser 1.0 em todo voxel (softmax sobre n_level)
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5), \
            "alpha deveria somar 1 entre as n_level direcoes em cada voxel (softmax valido)"
    print("OK: encode(return_attn=True)+decode() == forward() e alpha e' um softmax valido "
          "(aggregation=attention)")

    # permutation invariance TAMBEM com aggregation="attention" -- prova de
    # que o escore aprendido nao introduz nenhuma dependencia de ORDEM (ver
    # docstring de AttentionAggregator3D: escore de cada direcao depende so'
    # da propria feature dela, pesos compartilhados entre direcoes).
    model_attn.eval()
    with torch.no_grad():
        pred_orig_attn = model_attn(input_vols, input_bvecs, target_bvecs)
        pred_perm_attn = model_attn(input_vols[:, perm], input_bvecs[:, perm], target_bvecs)
    assert torch.allclose(pred_orig_attn, pred_perm_attn, atol=1e-5), \
        "aggregation=attention deveria continuar invariante a permutacao da ORDEM de entrada"
    print("OK: aggregation=attention tambem e' invariante a permutacao das n_level direcoes "
          "de entrada (escore por direcao depende so da propria feature, nao da posicao)")

    # aggregation invalido deve falhar cedo, na construcao (nao silenciosamente
    # cair pra "mean" nem so' falhar tarde dentro de encode/forward).
    try:
        build_implicit_model(n_level=n_level, base_ch=8, aggregation="max")
        raise AssertionError("aggregation invalido deveria levantar ValueError na construcao")
    except ValueError:
        print("OK: aggregation invalido ('max') levanta ValueError na construcao, como esperado")

    # cross_direction_attention=True (ADITIVO, ver CrossDirectionAttention3D,
    # item 2 do addendum 2026-09-13) -- mesmos testes de shape/permutation-
    # invariance, combinado com as duas opcoes de aggregation (mean E
    # attention, ja que sao etapas independentes do pipeline).
    for agg in ("mean", "attention"):
        model_xattn = build_implicit_model(n_level=n_level, base_ch=8, aggregation=agg,
                                            cross_direction_attention=True, cross_attn_heads=2)
        assert model_xattn.cross_attn is not None
        pred_xattn = model_xattn(input_vols, input_bvecs, target_bvecs)
        assert pred_xattn.shape == expected
        print(f"smoke test OK (cross_direction_attention=True, aggregation={agg}), "
              f"output shape: {tuple(pred_xattn.shape)}")

        model_xattn.eval()
        with torch.no_grad():
            pred_orig_x = model_xattn(input_vols, input_bvecs, target_bvecs)
            pred_perm_x = model_xattn(input_vols[:, perm], input_bvecs[:, perm], target_bvecs)
        assert torch.allclose(pred_orig_x, pred_perm_x, atol=1e-4), \
            (f"cross_direction_attention=True (aggregation={agg}) deveria continuar invariante "
             f"a permutacao da ORDEM das n_level direcoes de entrada")
        print(f"OK: cross_direction_attention=True (aggregation={agg}) tambem e' invariante a "
              f"permutacao das n_level direcoes de entrada")

    # numero de cabecas que nao divide base_ch deve falhar cedo, na
    # construcao do bloco de atencao (nao silenciosamente truncar nem falhar
    # tarde dentro de um forward).
    try:
        build_implicit_model(n_level=n_level, base_ch=8, cross_direction_attention=True,
                              cross_attn_heads=3)
        raise AssertionError("cross_attn_heads que nao divide base_ch deveria levantar "
                              "ValueError na construcao")
    except ValueError:
        print("OK: cross_attn_heads=3 com base_ch=8 (nao divisivel) levanta ValueError na "
              "construcao, como esperado")

    # CAPACIDADE DO DECODER (2026-09-14, ver addendum de capacidade):
    # --decoder-base-ch / --decoder-depth / --decoder-reinject-code.
    #
    # (a) GUARDA DE RETROCOMPATIBILIDADE: com os defaults, o decoder precisa
    # manter EXATAMENTE a contagem historica (13.873 em base_ch=16/l_max=4) e
    # os mesmos nomes de parametro no state_dict (`net.0.*`/`net.1.*`) -- a
    # troca de nn.Sequential por nn.ModuleList so' e' segura porque os dois
    # registram os filhos pelo indice. Se isto quebrar, todo checkpoint do
    # implicit ja treinado para de carregar.
    model_def = build_implicit_model(n_level=n_level, l_max=4, base_ch=16)
    n_dec_default = sum(p.numel() for p in model_def.decoder_head.parameters())
    assert model_def.sh_dim == 15, f"esperado sh_dim=15 com l_max=4, obtido {model_def.sh_dim}"
    assert n_dec_default == 13873, \
        (f"decoder com defaults deveria manter os 13.873 parametros historicos, obtido "
         f"{n_dec_default} -- isso INVALIDA checkpoints existentes do implicit")
    dec_keys = sorted(k for k, _ in model_def.decoder_head.named_parameters())
    assert all(k.startswith("net.0.") or k.startswith("net.1.") for k in dec_keys), \
        f"nomes de parametro do decoder mudaram ({dec_keys}) -- quebra checkpoints existentes"
    print(f"OK: decoder com defaults preserva os {n_dec_default} parametros e os nomes "
          f"historicos ({len(dec_keys)} tensores em net.0/net.1)")

    # (b) decoder maior/mais fundo/com reinjecao do codigo
    model_dec = build_implicit_model(n_level=n_level, l_max=4, base_ch=16, decoder_base_ch=32,
                                      decoder_depth=3, decoder_reinject_code=True)
    pred_dec = model_dec(input_vols, input_bvecs, target_bvecs)
    assert pred_dec.shape == (b, n_out, 1, d, h, w), f"shape mismatch: {pred_dec.shape}"
    n_dec_big = sum(p.numel() for p in model_dec.decoder_head.parameters())
    print(f"smoke test OK (decoder_base_ch=32, depth=3, reinject_code=True): decoder com "
          f"{n_dec_big} parametros ({n_dec_big / n_dec_default:.1f}x o default)")

    # (c) a reinjecao tem que REALMENTE mudar o calculo -- comparar contra a
    # mesma topologia sem reinjecao (contagens diferentes ja provam que as
    # camadas ocultas recebem canais a mais).
    model_noreinj = build_implicit_model(n_level=n_level, l_max=4, base_ch=16, decoder_base_ch=32,
                                          decoder_depth=3, decoder_reinject_code=False)
    n_dec_noreinj = sum(p.numel() for p in model_noreinj.decoder_head.parameters())
    assert n_dec_big > n_dec_noreinj, \
        "reinject_code=True deveria adicionar canais de entrada nas camadas ocultas"
    print(f"OK: reinject_code=True adiciona {n_dec_big - n_dec_noreinj} parametros vs. a mesma "
          f"topologia sem reinjecao (o codigo SH volta a entrar em cada camada oculta)")

    # (d) invariancia a permutacao tem que continuar valendo com o decoder novo
    model_dec.eval()
    with torch.no_grad():
        p_orig = model_dec(input_vols, input_bvecs, target_bvecs)
        p_perm = model_dec(input_vols[:, perm], input_bvecs[:, perm], target_bvecs)
    assert torch.allclose(p_orig, p_perm, atol=1e-4), \
        "decoder maior nao pode quebrar a invariancia a permutacao das direcoes de entrada"
    print("OK: decoder maior/mais fundo preserva a invariancia a permutacao das direcoes")

    try:
        build_implicit_model(n_level=n_level, l_max=4, decoder_depth=0)
        raise AssertionError("deveria ter levantado ValueError (decoder_depth=0)")
    except ValueError:
        print("OK: decoder_depth < 1 levanta ValueError cedo")


if __name__ == "__main__":
    _smoke_test()