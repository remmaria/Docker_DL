"""
AMT3D -- adaptacao 3D de "AMT: All-Pairs Multi-Field Transforms for
Efficient Frame Interpolation" (Li et al., CVPR 2023, arXiv:2304.09790,
https://github.com/MCG-NKU/AMT) para o mesmo esquema de trincas
(par de direcoes de entrada a,b + direcao-alvo t) usado por RRIN3D
(model/rrin3d.py) -- ver protocolo, secoes 10/10.3/13.

POR QUE ESTA REDE EXISTE (nao e so "mais um metodo", ver protocolo secao
10.3/13): RRIN3D ja mostrou (secao 10.1) que tratar duas direcoes de
gradiente como "quadros vizinhos" de video e interpolar por fluxo otico +
warping produz resultados ruins fora da faixa "trivial" (par bem alinhado,
gap pequeno) -- mas RRIN e uma rede deliberadamente pequena/simples (ver
docstring de model/rrin3d.py), o que deixa em aberto a duvida "sera que o
problema e a rede fraca, ou a PREMISSA (fluxo otico entre direcoes de
gradiente) que nao se sustenta?". AMT e um dos metodos state-of-the-art de
interpolacao de quadros de video (2D) na epoca de sua publicacao -- muito
mais sofisticado que RRIN/RIFE: correlacao all-pairs (custo explicito de
"olhar" todas as correspondencias possiveis, nao so um deslocamento local
regredido diretamente), estimativa coarse-to-fine SEM GRU (substituida por
convs simples, ver abaixo), e multi-field bilateral (K campos de fluxo
candidatos fundidos de forma aprendida, em vez de um unico fluxo por
direcao). Se AMT3D tambem falhar nos mesmos casos geometricos que RRIN3D
falha (mesma estratificacao aggregate_valid/aggregate_invalid de
scripts/06_evaluate_reconstruction.py), isso FORTALECE o argumento de que a
falha e estrutural (a premissa de fluxo otico entre direcoes de gradiente
nao se sustenta geometricamente fora do regime trivial), nao uma limitacao
de capacidade de uma rede especifica fraca. Se AMT3D performar
sistematicamente melhor que RRIN3D nos casos "dificeis" (gap grande,
residuo alto), isso enfraquece esse argumento e aponta pra capacidade de
rede como fator relevante -- ambas as leituras sao uteis, esta rede e um
diagnostico, nao uma tentativa de "vencer" o RCAE.

POR QUE A CORRELACAO ALL-PAIRS E BARATA AQUI (diferente do AMT original):
o AMT 2D opera em quadros de video em resolucao natural (centenas a
milhares de pixels por eixo) -- uma correlacao all-pairs completa (todo
pixel de a contra todo pixel de b) explodiria em memoria, e por isso o AMT
original (e RAFT, de onde a ideia vem) so calcula a correlacao numa
piramide de resolucoes reduzidas E faz o lookup local via indexacao
esparsa cara. Aqui os "quadros" sao patches 3D pequenos (10^3 voxels por
direcao no default deste pipeline, ate ~24^3 no maximo usado) -- uma
correlacao all-pairs completa em resolucao NATIVA (nao so numa piramide
reduzida) tem N=D*H*W~1000 elementos por lado, ou seja uma matriz
(B,N,N)~(B,1000,1000), poucos MB por batch -- trivial para GPU/CPU
modernas. Isso permite implementar o mecanismo central do AMT (correlacao
all-pairs + lookup local) fielmente, sem precisar aproximar/podar nada por
restricao de memoria -- a unica adaptacao real e dimensional (2D->3D).

PIPELINE (ver AMT, https://github.com/MCG-NKU/AMT/blob/main/docs/method.md):
  1. FeatureEncoder3D: encoder SIAMES (pesos compartilhados) pequeno, duas
     escalas -- fina (resolucao nativa do patch) e grossa (stride 2), para
     vol_a e vol_b.
  2. build_correlation: correlacao all-pairs (produto escalar, escalado por
     1/sqrt(C) -- mesma convencao de atencao/RAFT, sem normalizacao L2
     extra) entre feat_a e feat_b, calculada UMA VEZ por escala (nao
     recalculada a cada iteracao de refinamento -- ver AMT, esse e o ponto
     central que torna o metodo "eficiente" apesar de ver todos os pares).
  3. CorrLookup3D (`_corr_lookup_3d`): dado um campo de fluxo atual (a->b em
     coordenadas normalizadas, mesma convencao de model.rrin3d.warp3d),
     reamostra (trilinear, `F.grid_sample`) uma janela local
     (2*radius+1)^3 ao redor da posicao estimada, por voxel -- os valores
     de correlacao nessa janela viram canais extras de "custo" pro
     decoder.
  4. Coarse-to-fine SEM GRU: o AMT original ja substitui a GRU do
     RAFT/IFRNet por 2 convs simples por escala -- aqui replicamos essa
     simplificacao (nao e uma simplificacao NOSSA, e do proprio AMT). Na
     escala grossa, um lookup quase-global (a grade grossa e pequena o
     bastante que um raio de janela moderado cobre quase tudo, ver
     `coarse_corr_radius`) + as features grossas predizem um fluxo
     bilateral inicial (flow_a->t, flow_b->t) e uma visibilidade inicial,
     via 2 convs. Upsample trilinear pra resolucao fina (SEM reescalar a
     magnitude do fluxo pelo fator de upsampling -- ver nota de design
     abaixo, isso e uma escolha deliberada, nao um esquecimento).
  5. Multi-field na escala fina: em vez de 1 fluxo bilateral so, o decoder
     fino preve K campos CANDIDATOS (flow_a, flow_b, visibilidade cada),
     como deltas somados ao fluxo grosso upsampled. K default 3 -- o
     ablation do AMT original (ver paper, secao de ablation de "number of
     fields") mostra ganho saturando por volta de K~7; K=3 aqui e um
     default mais leve mas alinhado com a curva de saturacao do proprio
     paper (ganho marginal decrescente), nao uma limitacao imposta por
     falta de recurso.
  6. Fusao adaptativa (softmax aprendido, nao media simples): as K
     previsoes candidatas (ja com warp3d + blend por t/visibilidade, MESMA
     formula de model.rrin3d.RRIN3D.forward) sao combinadas por pesos
     softmax por voxel, preditos por mais 2 convs a partir das proprias
     candidatas + seus mapas de visibilidade -- mesmo espirito do
     "adaptive merging" descrito no AMT (e do softmax de selecao de camada
     de model.rrin3d.RRIN3DLayered, reaproveitado aqui como padrao de
     "warm start neutro": logits zerados no init -> softmax uniforme ->
     fusao comeca como MEDIA simples das K candidatas, e so se afasta disso
     conforme o treino achar vantajoso).
  7. RefineNet3D (model/rrin3d.py, IMPORTADA, nao duplicada): mesmo
     refinamento residual do RRIN3D, vendo a fusao + vol_a + vol_b crus
     (exatamente os mesmos 3 argumentos que RRIN3D.forward passa pra
     RefineNet3D -- conferido no codigo-fonte antes de escrever este
     arquivo, nao adivinhado).

SIMPLIFICACOES DELIBERADAS vs. o AMT 2D original (documentadas aqui pra
nao confundir "port fiel" com "reproducao literal"):
  - Sem GRU: NAO e uma simplificacao nossa -- o AMT original ja substitui a
    GRU do RAFT por convs simples ("task-oriented flow update" com 2 convs
    por escala). Replicamos isso, nao removemos algo que o AMT tinha.
  - Piramide de 2 niveis (grosso+fino), nao 4 como no AMT original: dado
    que os patches aqui sao MUITO menores que quadros de video (10-24^3 vs
    centenas/milhares de pixels por eixo), 2 niveis ja bastam pra dar
    contexto global (nivel grosso) + detalhe local (nivel fino) sem
    over-engineering -- adicionar mais niveis a uma piramide sobre um
    volume de 10^3 rapidamente degenera (nivel 3 seria ~3^3, quase sem
    estrutura espacial sobrando).
  - K=3 candidatos (nao os 8-10 tipicos do AMT-L): ver item 5 acima, dentro
    da faixa de saturacao de ganho do proprio paper.
  - use_quality_cond (residual_deg/gap_deg da trinca, ver
    model.rrin3d.RRIN3D) esta implementado aqui do mesmo jeito que no RRIN
    (2 canais extras de condicionamento, concatenados nas cabecas grossa E
    fina) -- barato de adicionar dado que a arquitetura ja concatena
    bvec_a/bvec_b/bvec_t/t da mesma forma, entao NAO ficou como TODO.
  - Termo de loss angular/SH (utils/sh_angular_loss.py, ja portado pro RRIN
    em scripts/04b_train_rrin.py): NAO ficou como TODO -- ja esta totalmente
    portado e ativo em scripts/04c_train_amt.py (--angular-loss-weight,
    reaproveitando a MESMA infra/RRINTripletDataset.sh_q_out sem qualquer
    mudanca de arquitetura aqui). Esta nota antes dizia "TODO"; foi corrigida
    em 2026-08-27 apos verificacao cruzada com o codigo real do script de
    treino (ver ANGULAR_LOSS_WEIGHT em slurm/04c_train_amt.sh).
  - norm_type="batch" (BatchNorm3d, ver model.rrin3d._norm3d e o artefato de
    "costura" entre patches na reconstrucao com sliding-window): a opcao
    JA EXISTE aqui (_norm3d e _conv3d sao IMPORTADOS de model.rrin3d, nao
    duplicados), entao "batch" ja funciona em AMT3D sem trabalho extra --
    nao ficou como TODO.

NOTA DE DESIGN -- fluxo em coordenadas NORMALIZADAS, nao em pixels/voxels:
o AMT original (como RAFT/IFRNet) trabalha com fluxo em unidades de PIXEL,
entao upsample entre escalas exige reescalar a magnitude do fluxo pelo
fator de upsampling (um fluxo de "2 pixels" na escala 1/4 vira "8 pixels"
na escala 1/1). Aqui, seguindo a MESMA convencao de model.rrin3d.warp3d
(fluxo em coordenadas normalizadas -1..1, onde -1..1 cobre o volume INTEIRO
em qualquer resolucao), o fluxo e automaticamente comparavel entre escalas
sem nenhum fator de reescala -- upsample trilinear simples do CAMPO em si
ja basta. Isso e uma escolha deliberada de manter compatibilidade com a
convencao ja estabelecida em model/rrin3d.py (reaproveitando warp3d
diretamente, sem reimplementar), nao um desvio por descuido do "way" que o
AMT original faz o reescalonamento.

Requer PyTorch (nao disponivel neste ambiente de desenvolvimento --
revisado manualmente, testado apenas por compilacao de sintaxe; validar no
cluster com `python -m model.amt3d`, smoke test no fim do arquivo, mesmo
padrao de model/rcae.py e model/rrin3d.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reaproveita a infra ja validada do RRIN3D em vez de duplicar -- helpers de
# conv/norm 3D, broadcast de vetor->mapa espacial, warp trilinear e a rede
# de refinamento residual sao EXATAMENTE os mesmos usados por RRIN3D (ver
# model/rrin3d.py). Nada disso e reimplementado aqui.
from .rrin3d import _conv3d, _norm3d, _repeat_vec_3d, warp3d, RefineNet3D


def build_correlation(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """Correlacao all-pairs entre duas grades de features (B,C,D,H,W) --
    produto escalar por par de posicoes, escalado por 1/sqrt(C) (mesma
    convencao de atencao/RAFT -- estabiliza a magnitude independente do
    numero de canais; NAO normalizamos L2 os vetores de feature, so o
    escalonamento pelo tamanho do canal). Calculada UMA VEZ por escala (ver
    docstring do modulo) -- quem usa (`_corr_lookup_3d`) so faz lookup
    local nela, nunca recalcula o produto escalar.

    feat_a: (B, C, Da, Ha, Wa); feat_b: (B, C, Db, Hb, Wb) -- podem ter
    tamanhos espaciais diferentes (nao precisam ser a mesma grade).
    retorna: (B, Da, Ha, Wa, Db, Hb, Wb)."""
    b, c = feat_a.shape[:2]
    spatial_a = feat_a.shape[2:]
    spatial_b = feat_b.shape[2:]
    n_a = spatial_a[0] * spatial_a[1] * spatial_a[2]
    n_b = spatial_b[0] * spatial_b[1] * spatial_b[2]
    fa = feat_a.reshape(b, c, n_a).permute(0, 2, 1)  # (B, Na, C)
    fb = feat_b.reshape(b, c, n_b)                   # (B, C, Nb)
    corr = torch.matmul(fa, fb) / (c ** 0.5)         # (B, Na, Nb)
    return corr.view(b, *spatial_a, *spatial_b)


def _corr_lookup_3d(corr: torch.Tensor, flow: torch.Tensor, radius: int) -> torch.Tensor:
    """Lookup local de correlacao (analogo ao "correlation lookup" de
    RAFT/AMT): para cada voxel da grade `a`, reamostra uma janela
    (2*radius+1)^3 de valores de correlacao na grade `b`, centrada na
    posicao estimada pelo fluxo atual (base_grid + flow, MESMA convencao de
    model.rrin3d.warp3d -- flow em coordenadas normalizadas -1..1, canais
    na ordem dx,dy,dz = deslocamento em x,y,z).

    corr: (B, Da,Ha,Wa, Db,Hb,Wb) -- ver build_correlation.
    flow: (B, 3, Da,Ha,Wa) -- fluxo de a para a posicao estimada em b.
    retorna: (B, (2*radius+1)**3, Da, Ha, Wa) -- canais extras de "custo"
        pra concatenar com as features do decoder."""
    b, da, ha, wa, db, hb, wb = corr.shape
    n_a = da * ha * wa
    # "achata" o lado a da correlacao no eixo de batch -- cada voxel de a
    # vira um item de batch independente com seu proprio volume escalar
    # (Db,Hb,Wb) sobre o qual faremos o grid_sample local.
    corr_flat = corr.reshape(b * n_a, 1, db, hb, wb)

    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, da, device=corr.device, dtype=corr.dtype),
        torch.linspace(-1, 1, ha, device=corr.device, dtype=corr.dtype),
        torch.linspace(-1, 1, wa, device=corr.device, dtype=corr.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy, zz], dim=-1)               # (Da,Ha,Wa,3), ordem (x,y,z)
    base = base.unsqueeze(0).expand(b, -1, -1, -1, -1)     # (B,Da,Ha,Wa,3)
    flow_p = flow.permute(0, 2, 3, 4, 1)                   # (B,Da,Ha,Wa,3)
    center = torch.clamp(base + flow_p, -1.0, 1.0)         # (B,Da,Ha,Wa,3)
    center_flat = center.reshape(b * n_a, 1, 1, 1, 3)

    r = radius
    coords = torch.arange(-r, r + 1, device=corr.device, dtype=corr.dtype)
    ddz, ddy, ddx = torch.meshgrid(coords, coords, coords, indexing="ij")  # (win,win,win)
    # unidades normalizadas por voxel da grade b (convencao align_corners=True,
    # mesma de warp3d): 2/(dim-1); max(...,1) so protege contra dim=1 (patch
    # degenerado de tamanho 1 num eixo, nao deveria ocorrer na pratica).
    step_x, step_y, step_z = 2.0 / max(wb - 1, 1), 2.0 / max(hb - 1, 1), 2.0 / max(db - 1, 1)
    offset = torch.stack([ddx * step_x, ddy * step_y, ddz * step_z], dim=-1)  # (win,win,win,3)
    offset = offset.unsqueeze(0)  # (1,win,win,win,3), broadcast no batch b*n_a

    grid = torch.clamp(center_flat + offset, -1.0, 1.0)  # (B*Na, win,win,win, 3)
    sampled = F.grid_sample(corr_flat, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)  # (B*Na, 1, win,win,win)
    win = 2 * r + 1
    sampled = sampled.reshape(b, da, ha, wa, win ** 3).permute(0, 4, 1, 2, 3)
    return sampled  # (B, win**3, Da, Ha, Wa)


class FeatureEncoder3D(nn.Module):
    """Encoder siames (pesos compartilhados entre vol_a e vol_b, chamado
    duas vezes com o mesmo modulo -- ver AMT3D.forward) em duas escalas:
    fina (resolucao nativa do patch) e grossa (1 downsample, stride 2).
    Deliberadamente raso (2-3 blocos conv+norm+ativacao por escala, ver
    docstring do modulo) -- so extrai features de conteudo, NAO ve
    bvec/t/quality (o condicionamento geometrico entra so nas cabecas de
    fluxo, coarse_head/fine_head, mesmo padrao de onde RRIN3D concatena
    bvec/t: na entrada da rede que PREVE fluxo, nao no encoder de imagem)."""

    def __init__(self, base_ch: int = 16, norm_type: str = "instance"):
        super().__init__()
        self.fine1 = _conv3d(1, base_ch, norm_type=norm_type)
        self.fine2 = _conv3d(base_ch, base_ch, norm_type=norm_type)
        self.coarse1 = _conv3d(base_ch, base_ch * 2, stride=2, norm_type=norm_type)
        self.coarse2 = _conv3d(base_ch * 2, base_ch * 2, norm_type=norm_type)

    def forward(self, vol: torch.Tensor):
        fine = self.fine2(self.fine1(vol))
        coarse = self.coarse2(self.coarse1(fine))
        return fine, coarse


class AMT3D(nn.Module):
    """Modelo completo -- ver docstring do modulo para o pipeline detalhado
    (encoder siames -> correlacao all-pairs grossa+fina -> estimativa
    bilateral coarse-to-fine sem GRU -> multi-field (K candidatos) ->
    fusao adaptativa (softmax) -> RefineNet3D importada de model/rrin3d.py).

    Uso (MESMA assinatura de forward de model.rrin3d.RRIN3D.forward, de
    proposito -- os scripts de treino/reconstrucao chamam os dois modelos
    de forma intercambiavel):
        model = AMT3D(num_fields=3, use_quality_cond=False)
        pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None)
    vol_a, vol_b: (B, 1, D, H, W); bvec_a, bvec_b, bvec_t: (B, 3); t: (B,);
    quality: (B, 2) ou None (obrigatorio se use_quality_cond=True).
    retorna: (B, 1, D, H, W) -- direcao-alvo predita.

    num_fields (K, default 3): numero de campos de fluxo bilateral
    candidatos preditos na escala fina (ver docstring do modulo, item 5) --
    hiperparametro de ARQUITETURA (muda o numero de canais de saida do
    decoder fino e da rede de fusao), por isso BLOQUEANTE em resume (ver
    scripts/04c_train_amt.py).

    corr_radius (default 3): raio (em voxels da grade "b") da janela local
    de lookup de correlacao na escala FINA (ver `_corr_lookup_3d`) -- NAO
    e um parametro aprendido, so o tamanho fixo da janela de amostragem
    (afeta o numero de canais concatenados nas cabecas, que dependem do
    tamanho da PRIMEIRA camada conv de cada cabeca -- ou seja, MUDA sim o
    shape dos pesos da primeira conv da cabeca grossa/fina, entao tambem e
    tratado como bloqueante em resume, ver scripts/04c_train_amt.py).
    coarse_corr_radius (default = corr_radius): raio equivalente na escala
    GROSSA -- como a grade grossa e pequena (ver docstring do modulo), um
    raio da mesma ordem de grandeza do corr_radius fino ja cobre quase toda
    a grade grossa (comportamento "quase global" mencionado no AMT)."""

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, num_fields: int = 3,
                 corr_radius: int = 3, coarse_corr_radius: int | None = None,
                 use_quality_cond: bool = False, norm_type: str = "instance"):
        super().__init__()
        if num_fields < 1:
            raise ValueError(f"num_fields deve ser >= 1 (recebido {num_fields})")
        self.num_fields = num_fields
        self.max_disp = max_disp
        self.corr_radius = corr_radius
        self.coarse_corr_radius = coarse_corr_radius if coarse_corr_radius is not None else corr_radius
        self.use_quality_cond = use_quality_cond
        self.norm_type = norm_type

        self.encoder = FeatureEncoder3D(base_ch=base_ch, norm_type=norm_type)

        cond_ch = 3 + 3 + 3 + 1  # bvec_a, bvec_b, bvec_t, t
        if use_quality_cond:
            cond_ch += 2  # residual_norm, gap_norm -- mesma convencao de model.rrin3d.RRIN3D

        win_coarse = (2 * self.coarse_corr_radius + 1) ** 3
        coarse_in_ch = (base_ch * 2) * 2 + win_coarse * 2 + cond_ch
        self.coarse_head = nn.Sequential(
            _conv3d(coarse_in_ch, base_ch * 2, norm_type=norm_type),
            _conv3d(base_ch * 2, base_ch * 2, norm_type=norm_type),
        )
        self.coarse_out = nn.Conv3d(base_ch * 2, 7, kernel_size=3, padding=1)  # 3+3+1

        win_fine = (2 * self.corr_radius + 1) ** 3
        # feat_a, feat_b (fino) + lookups a->b/b->a + flow_a/flow_b/vis
        # upsampled da escala grossa + condicionamento geometrico
        fine_in_ch = base_ch * 2 + win_fine * 2 + 3 + 3 + 1 + cond_ch
        self.fine_head = nn.Sequential(
            _conv3d(fine_in_ch, base_ch, norm_type=norm_type),
            _conv3d(base_ch, base_ch, norm_type=norm_type),
        )
        self.fine_out = nn.Conv3d(base_ch, 7 * num_fields, kernel_size=3, padding=1)

        # fusao adaptativa: ve as K candidatas (ja com warp+blend aplicados)
        # + seus K mapas de visibilidade, preve K logits de peso por voxel.
        self.fusion_net = nn.Sequential(
            _conv3d(num_fields * 2, base_ch, norm_type=norm_type),
            _conv3d(base_ch, base_ch, norm_type=norm_type),
        )
        self.fusion_out = nn.Conv3d(base_ch, num_fields, kernel_size=3, padding=1)

        self.refine_net = RefineNet3D(base_ch=base_ch, norm_type=norm_type)

        # inicializacao "morna" (mesma pratica/motivacao de model.rrin3d.py,
        # ver comentario la -- so as ULTIMAS camadas de cada cabeca sao
        # zeradas, nao a rede inteira):
        #   coarse_out=0 -> flow_a=flow_b=0, vis=sigmoid(0)=0.5 (neutro)
        #   fine_out=0   -> delta_flow=0 (fluxo fino comeca IGUAL ao grosso
        #                   upsampled, pras K "copias"), vis_k=0.5 pra toda
        #                   candidata
        #   fusion_out=0 -> logits=0 -> softmax uniforme (1/K) -> fusao
        #                   comeca como MEDIA simples das K candidatas
        #                   (todas identicas no init, ja que fine_out=0
        #                   tambem zera os deltas -- fusao "nao faz nada de
        #                   especial" ate o treino achar vantajoso).
        for head in (self.coarse_out, self.fine_out, self.fusion_out):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=None, return_flow=False):
        """`return_flow=False` (default): comportamento inalterado, retorna
        so a predicao final (B,1,D,H,W) -- 100% compativel com todo call
        site existente (scripts/04c_train_amt.py, scripts/05d_reconstruct_amt.py).

        `return_flow=True` (novo em 2026-08-27, aditivo -- ver model/hfd3d.py):
        retorna `(pred, flow_a_final, flow_b_final, vis_logit_final)`, onde
        os 3 ultimos sao o fluxo bilateral FINAL (escala fina, ja fundido
        pelas K candidatas via a mesma media ponderada por `fusion_weights`
        usada pra fundir a imagem -- ver abaixo) em coordenadas normalizadas
        (mesma convencao de warp3d). Usado pela HFD3D como fonte de fluxo
        "pseudo-verdadeiro" pra supervisionar o denoiser de difusao (a AMT3D
        ja e a rede de fluxo mais forte validada neste projeto, ver secao 6
        do addendum do protocolo -- convem usa-la como professora em vez de
        uma rede de fluxo externa nao validada aqui)."""
        if self.use_quality_cond and quality is None:
            raise ValueError("use_quality_cond=True mas `quality` nao foi passado ao forward")
        spatial_fine = vol_a.shape[-3:]

        fine_a, coarse_a = self.encoder(vol_a)
        fine_b, coarse_b = self.encoder(vol_b)  # mesmo modulo (siames), chamado 2x

        # --- escala grossa: correlacao all-pairs + fluxo bilateral inicial ---
        corr_ab_c = build_correlation(coarse_a, coarse_b)
        corr_ba_c = build_correlation(coarse_b, coarse_a)
        spatial_coarse = coarse_a.shape[-3:]
        zero_flow_c = torch.zeros(coarse_a.shape[0], 3, *spatial_coarse,
                                   device=vol_a.device, dtype=vol_a.dtype)
        lookup_ab_c = _corr_lookup_3d(corr_ab_c, zero_flow_c, self.coarse_corr_radius)
        lookup_ba_c = _corr_lookup_3d(corr_ba_c, zero_flow_c, self.coarse_corr_radius)

        t_col = t.view(-1, 1)
        parts_c = [coarse_a, coarse_b, lookup_ab_c, lookup_ba_c,
                   _repeat_vec_3d(bvec_a, spatial_coarse), _repeat_vec_3d(bvec_b, spatial_coarse),
                   _repeat_vec_3d(bvec_t, spatial_coarse), _repeat_vec_3d(t_col, spatial_coarse)]
        if self.use_quality_cond:
            parts_c.append(_repeat_vec_3d(quality, spatial_coarse))
        feat_c = self.coarse_head(torch.cat(parts_c, dim=1))
        raw_c = self.coarse_out(feat_c)
        flow_a_c = torch.tanh(raw_c[:, 0:3]) * self.max_disp
        flow_b_c = torch.tanh(raw_c[:, 3:6]) * self.max_disp
        vis_logit_c = raw_c[:, 6:7]

        # upsample trilinear pra resolucao fina -- SEM reescalar magnitude
        # (fluxo em coordenadas normalizadas, ver nota de design no docstring
        # do modulo)
        flow_a_up = F.interpolate(flow_a_c, size=spatial_fine, mode="trilinear", align_corners=True)
        flow_b_up = F.interpolate(flow_b_c, size=spatial_fine, mode="trilinear", align_corners=True)
        vis_up = F.interpolate(vis_logit_c, size=spatial_fine, mode="trilinear", align_corners=True)

        # --- escala fina: correlacao all-pairs + refinamento multi-field ---
        corr_ab_f = build_correlation(fine_a, fine_b)
        corr_ba_f = build_correlation(fine_b, fine_a)
        lookup_ab_f = _corr_lookup_3d(corr_ab_f, flow_a_up, self.corr_radius)
        lookup_ba_f = _corr_lookup_3d(corr_ba_f, flow_b_up, self.corr_radius)

        parts_f = [fine_a, fine_b, lookup_ab_f, lookup_ba_f, flow_a_up, flow_b_up, vis_up,
                   _repeat_vec_3d(bvec_a, spatial_fine), _repeat_vec_3d(bvec_b, spatial_fine),
                   _repeat_vec_3d(bvec_t, spatial_fine), _repeat_vec_3d(t_col, spatial_fine)]
        if self.use_quality_cond:
            parts_f.append(_repeat_vec_3d(quality, spatial_fine))
        feat_f = self.fine_head(torch.cat(parts_f, dim=1))
        raw_f = self.fine_out(feat_f)  # (B, 7*K, D,H,W)
        b = raw_f.shape[0]
        raw_f = raw_f.view(b, self.num_fields, 7, *spatial_fine)
        delta_flow_a = torch.tanh(raw_f[:, :, 0:3]) * self.max_disp   # (B,K,3,D,H,W)
        delta_flow_b = torch.tanh(raw_f[:, :, 3:6]) * self.max_disp
        vis_logit_k = raw_f[:, :, 6]                                  # (B,K,D,H,W)

        # cada campo candidato e o fluxo grosso upsampled + um delta proprio
        # (refinamento residual coarse-to-fine, "2a metade" do esquema
        # sem-GRU do AMT) -- flow_a_up/flow_b_up sao (B,3,D,H,W), unsqueeze
        # no eixo K pra somar com os K deltas.
        flow_a_k = flow_a_up.unsqueeze(1) + delta_flow_a  # (B,K,3,D,H,W)
        flow_b_k = flow_b_up.unsqueeze(1) + delta_flow_b

        t_map = t.view(-1, 1, 1, 1, 1)
        candidates, vis_maps = [], []
        for k in range(self.num_fields):
            warped_a_k = warp3d(vol_a, flow_a_k[:, k])
            warped_b_k = warp3d(vol_b, flow_b_k[:, k])
            vis_k = torch.sigmoid(vis_logit_k[:, k:k + 1])
            # MESMA formula de blend de model.rrin3d.RRIN3D.forward (combina
            # a posicao relativa t com a visibilidade aprendida desta
            # candidata especifica).
            w_a = (1.0 - t_map) * vis_k
            w_b = t_map * (1.0 - vis_k)
            denom = (w_a + w_b).clamp(min=1e-6)
            candidates.append((w_a * warped_a_k + w_b * warped_b_k) / denom)
            vis_maps.append(vis_k)
        candidates_t = torch.cat(candidates, dim=1)  # (B,K,D,H,W)
        vis_t = torch.cat(vis_maps, dim=1)           # (B,K,D,H,W)

        # fusao adaptativa (substitui a media simples entre candidatas) --
        # ver docstring do modulo, item 6, e o comentario de init acima
        # (warm start = media uniforme).
        fusion_feat = self.fusion_net(torch.cat([candidates_t, vis_t], dim=1))
        fusion_weights = torch.softmax(self.fusion_out(fusion_feat), dim=1)  # (B,K,D,H,W)
        fused_blend = (fusion_weights * candidates_t).sum(dim=1, keepdim=True)  # (B,1,D,H,W)

        residual = self.refine_net(fused_blend, vol_a, vol_b)
        pred = fused_blend + residual
        if not return_flow:
            return pred

        # Fluxo "final" pra fins de professor/pseudo-GT (return_flow=True):
        # funde os K campos de fluxo/visibilidade candidatos pelos MESMOS
        # pesos (fusion_weights) usados pra fundir a imagem -- ponderacao
        # consistente entre o que a rede "usou de fato" pra montar a imagem
        # final e o fluxo que descrevemos como tendo sido usado.
        fw = fusion_weights.unsqueeze(2)  # (B,K,1,D,H,W)
        flow_a_final = (fw * flow_a_k).sum(dim=1)  # (B,3,D,H,W)
        flow_b_final = (fw * flow_b_k).sum(dim=1)
        vis_logit_final = (fusion_weights * vis_logit_k).sum(dim=1, keepdim=True)  # (B,1,D,H,W)
        return pred, flow_a_final, flow_b_final, vis_logit_final


def build_amt_model(base_ch: int = 16, max_disp: float = 0.5, num_fields: int = 3,
                     corr_radius: int = 3, coarse_corr_radius: int | None = None,
                     use_quality_cond: bool = False, norm_type: str = "instance") -> AMT3D:
    """Dispatcher, mesmo espirito de model.rrin3d.build_rrin_model (usar
    esta funcao em scripts/04c_train_amt.py e scripts/05d_reconstruct_amt.py
    em vez de instanciar AMT3D diretamente, para manter os dois scripts
    sincronizados -- o checkpoint grava `num_fields`/`corr_radius`/
    `norm_type` em `args`, e a reconstrucao le de la). Diferente de
    build_rrin_model, nao ha aqui um caso especial "K=1 usa outra classe" --
    AMT3D com num_fields=1 ja e um caso degenerado valido da MESMA classe
    (fusao com K=1 vira so softmax(escalar)=1, sem-efeito), entao uma unica
    classe cobre todo o intervalo de K sem precisar de uma variante
    separada tipo RRIN3D/RRIN3DLayered."""
    return AMT3D(base_ch=base_ch, max_disp=max_disp, num_fields=num_fields,
                 corr_radius=corr_radius, coarse_corr_radius=coarse_corr_radius,
                 use_quality_cond=use_quality_cond, norm_type=norm_type)


def _smoke_test():
    """Forward pass com tensores pequenos aleatorios, so pra checar shapes
    -- mesmo padrao de model/rcae.py e model/rrin3d.py. Nao executavel neste
    ambiente de desenvolvimento (sem PyTorch) -- rodar no cluster:
    python -m model.amt3d"""
    torch.manual_seed(0)
    b, d, h, w = 2, 10, 10, 10
    vol_a = torch.rand(b, 1, d, h, w)
    vol_b = torch.rand(b, 1, d, h, w)
    bvec_a = torch.randn(b, 3); bvec_a = bvec_a / bvec_a.norm(dim=-1, keepdim=True)
    bvec_b = torch.randn(b, 3); bvec_b = bvec_b / bvec_b.norm(dim=-1, keepdim=True)
    bvec_t = torch.randn(b, 3); bvec_t = bvec_t / bvec_t.norm(dim=-1, keepdim=True)
    t = torch.rand(b)
    expected = (b, 1, d, h, w)

    model = build_amt_model(base_ch=8, num_fields=3, corr_radius=2, use_quality_cond=False)
    out = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
    assert out.shape == expected, f"shape mismatch (sem quality): {out.shape} != {expected}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"smoke test OK (use_quality_cond=False), output shape: {tuple(out.shape)}, "
          f"{n_params} parametros")

    # warm start: no init (sem nenhum treino), a saida deve ser identica
    # pra qualquer K (fusao = media de K candidatas identicas) -- checagem
    # indireta de que o zero-init dos heads propaga como esperado.
    model.eval()
    with torch.no_grad():
        out_eval = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
    assert out_eval.shape == expected

    model_q = build_amt_model(base_ch=8, num_fields=2, corr_radius=2, use_quality_cond=True)
    quality = torch.rand(b, 2)
    out_q = model_q(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, quality=quality)
    assert out_q.shape == expected, f"shape mismatch (com quality): {out_q.shape} != {expected}"
    n_params_q = sum(p.numel() for p in model_q.parameters())
    print(f"smoke test OK (use_quality_cond=True), output shape: {tuple(out_q.shape)}, "
          f"{n_params_q} parametros")

    try:
        model_q(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
        raise AssertionError("deveria ter levantado ValueError sem `quality`")
    except ValueError:
        print("OK: chamar sem `quality` com use_quality_cond=True levanta ValueError, como esperado")

    # varias combinacoes de K/corr_radius/norm_type, so pra checar shapes e
    # que a rede nao quebra com janelas de lookup maiores/menores.
    for k in (1, 2, 5):
        for radius in (1, 3):
            model_k = build_amt_model(base_ch=8, num_fields=k, corr_radius=radius)
            out_k = model_k(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
            assert out_k.shape == expected, \
                f"shape mismatch (K={k}, radius={radius}): {out_k.shape} != {expected}"
            print(f"smoke test OK (num_fields={k}, corr_radius={radius}), "
                  f"output shape: {tuple(out_k.shape)}")

    model_bn = build_amt_model(base_ch=8, num_fields=3, norm_type="batch")
    assert isinstance(model_bn.encoder.fine1[1], nn.BatchNorm3d)
    out_bn = model_bn(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t)
    assert out_bn.shape == expected
    print(f"smoke test OK (norm_type=batch), output shape: {tuple(out_bn.shape)}")

    try:
        build_amt_model(num_fields=0)
        raise AssertionError("num_fields=0 deveria levantar ValueError")
    except ValueError:
        print("OK: num_fields=0 levanta ValueError, como esperado")

    # return_flow=True (novo, 2026-08-27, ver model/hfd3d.py): checa shapes
    # do fluxo/visibilidade "professor" retornado, sem mudar o retorno
    # default (return_flow=False, ja coberto acima).
    out_rf, flow_a_f, flow_b_f, vis_f = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t,
                                               return_flow=True)
    assert out_rf.shape == expected
    assert flow_a_f.shape == (b, 3, d, h, w), f"flow_a_final shape errado: {flow_a_f.shape}"
    assert flow_b_f.shape == (b, 3, d, h, w), f"flow_b_final shape errado: {flow_b_f.shape}"
    assert vis_f.shape == (b, 1, d, h, w), f"vis_logit_final shape errado: {vis_f.shape}"
    assert torch.allclose(out_rf, out), "return_flow=True nao deveria mudar a predicao"
    print("OK: return_flow=True retorna (pred, flow_a, flow_b, vis_logit) com shapes corretos "
          "e pred identica ao return_flow=False")


if __name__ == "__main__":
    _smoke_test()