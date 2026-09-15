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

ATUALIZACAO (discussao pos-secao 20.9 do addendum -- diagnostico do glifo
"torto" do star610 mostrou que, quando NENHUM par do feixe tem `gap_deg`
baixo pra um alvo/voxel, a fusao aprendida acaba fazendo uma media
razoavelmente equilibrada entre candidatos igualmente medianos, borrando
estrutura angular fina; ver secao 20.6): `PairWeightHead3D`/`RRIN3DStar`
ganharam um `use_quality_cond`/`weight_quality_cond` PROPRIO, ADITIVO e
DESACOPLADO do `use_quality_cond` do FlowNet3D -- alimenta `residual_deg`/
`gap_deg` de cada par DIRETO na cabeca de fusao, em vez de deixar a cabeca
inferir confiabilidade so pelo conteudo de imagem. Paralelo com VFI de
imagem natural: e o mesmo principio do "Z-metric"/importancia conhecida do
Softmax Splatting (Niklaus & Liu, CVPR 2020) -- alimentar um sinal de
qualidade JA CONHECIDO no mecanismo de peso, em vez de exigir que ele seja
redescoberto do zero a partir de pixels. Nao resolve o problema de fundo
(pool sem candidato bom, ver secao 20.6/20.9) -- so barateia o trabalho da
cabeca de fusao pra usar o sinal geometrico de forma mais direta/estavel.

ATUALIZACAO (2026-09-14 -- item 1 da discussao 2026-09-13 sobre por que o
RCAE bate RRIN3DStar/PairFlowStar/implicit, ver addendum
2026-09-13_hipotese_rcae_vs_fluxo_residual_l2_quality_cond.md): a
`PairWeightHead3D` sempre calculou o logit de confianca de CADA candidato
de forma totalmente INDEPENDENTE dos outros M-1 candidatos do mesmo feixe
-- so a normalizacao final (softmax mascarado) via os M logits juntos, nao
o CALCULO de cada logit. `PairWeightHead3D`/`RRIN3DStar` ganharam um
`cross_candidate_attention`/`cross_candidate_attn_heads` PROPRIOS, ADITIVOS
(default desligado): quando ligados, insere-se um bloco de self-attention
completa (`CrossCandidateAttention3D`, MESMO principio geral do
`CrossDirectionAttention3D` do item 2 em model/implicit_angular.py) entre
os M candidatos, POR VOXEL, logo apos a projecao inicial da
`PairWeightHead3D` e ANTES do `out_conv` que produz o logit -- assim a
cabeca de fusao pode julgar "quao bom e este candidato" de forma RELATIVA
aos outros M-1, nao so absoluta a partir do proprio par. Posicoes de
PADDING do feixe (`ensemble_mask=False`) sao mascaradas via
`key_padding_mask` de `nn.MultiheadAttention`, entao nunca contaminam o
logit dos candidatos REAIS (ver teste de independencia no `_smoke_test`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .rrin3d import FlowNet3D, RefineNet3D, _conv3d, _repeat_vec_3d, warp3d


# Numero de canais do vetor de condicionamento da RefineNet3D quando
# `refine_cond=True` (ver RRIN3DStar.__init__ e model/rrin3d.py:RefineNet3D):
# bvec_a (3) + bvec_b (3) + bvec_t (3) + t_frac (1) + quality (2) = 12.
# Ordem FIXA -- e a mesma montada por `_refine_cond_vector` abaixo e a mesma
# assumida por qualquer checkpoint treinado com a flag ligada; mudar a ordem
# ou o tamanho invalidaria checkpoints existentes em silencio.
REFINE_COND_CH = 12


def _refine_cond_vector(bvec_a, bvec_b, bvec_t, t, quality):
    """Monta o vetor de condicionamento (B*M, REFINE_COND_CH) da RefineNet3D
    a partir dos tensores JA ACHATADOS do feixe. `quality` pode ser None (o
    treino so' a repassa quando alguma flag de qualidade esta ligada, ver
    `need_quality` nos scripts de treino) -- nesse caso os 2 canais de
    qualidade entram ZERADOS, mantendo `cond_ch` fixo em 12
    independentemente das outras flags (o shape dos pesos nao pode depender
    de qual flag de qualidade esta ligada, senao um mesmo checkpoint nao
    carregaria em configuracoes diferentes).

    Por que o modelo ganha com isso: hoje a RefineNet3D nao recebe NENHUMA
    informacao de QUAL direcao ela esta predizendo nem de quao confiavel e
    a geometria do par -- ela ve so' o blend e os dois volumes crus. O
    `gap_deg`/`residual_deg` (em `quality`) e exatamente o sinal de "quanto
    desconfiar do warp" que decide o tamanho da correcao necessaria."""
    parts = [bvec_a, bvec_b, bvec_t, t.view(-1, 1)]
    if quality is not None:
        parts.append(quality)
    else:
        parts.append(torch.zeros(bvec_a.shape[0], 2, dtype=bvec_a.dtype, device=bvec_a.device))
    return torch.cat(parts, dim=1)


class CrossCandidateAttention3D(nn.Module):
    """Bloco de self-attention ENTRE os M candidatos (pares) do ensemble em
    estrela, aplicado POR VOXEL (pesos compartilhados no espaco) -- ADITIVO
    (item 1 da discussao 2026-09-13 sobre por que o RCAE bate RRIN3DStar/
    PairFlowStar/implicit): `PairWeightHead3D` sempre calculou o logit de
    confianca de cada candidato de forma totalmente INDEPENDENTE dos outros
    M-1 candidatos do mesmo feixe (mesmo alvo) -- so a normalizacao final
    (softmax mascarado sobre os M logits) acontece depois, em
    RRIN3DStar.forward. Este bloco insere a interacao FALTANTE ANTES do
    calculo do logit (ver PairWeightHead3D.forward): cada candidato passa a
    ser atualizado por atencao sobre as features dos OUTROS candidatos do
    mesmo feixe, permitindo que a cabeca de fusao julgue "quao bom e este
    candidato" de forma RELATIVA aos demais, nao so absoluta -- mesmo
    principio geral do "Set Transformer" (Lee et al., ICML 2019) ja usado
    pelo `CrossDirectionAttention3D` do item 2 (model/implicit_angular.py),
    aqui adaptado pra M candidatos em vez de n_level direcoes.

    Por que self-attention COMPLETA (SAB) em vez de ISAB: M e tipicamente
    pequeno (3-8, ver find_star_ensemble_batch), custo O(M^2) desprezivel
    (mesmo raciocinio de CrossDirectionAttention3D, so que a dimensao
    "grande" aqui tambem e o espaco B*D*H*W, nao o eixo dos candidatos).

    PERMUTATION-EQUIVARIANTE por construcao (sem nenhuma codificacao de
    posicao de sequencia) -- permutar a ordem dos M candidatos de entrada
    permuta a saida na MESMA ordem, sem mudar nenhum valor individual (ver
    teste de permutacao em _smoke_test); isso preserva a invariancia a
    ordem/identidade dos M pares que o design original do ensemble em
    estrela ja exige (ver docstring do modulo).

    Posicoes de PADDING (`ensemble_mask=False`) sao mascaradas via
    `key_padding_mask` de `nn.MultiheadAttention` -- do contrario, um
    candidato REAL atenderia sobre features de um par inexistente (lixo/
    zeros sem significado fisico), contaminando seu logit de confianca (ver
    teste de independencia em _smoke_test)."""

    def __init__(self, channels: int, num_heads: int = 4, ff_mult: int = 2):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) precisa ser divisivel por num_heads ({num_heads}) -- "
                f"ver --base-ch/--cross-candidate-attn-heads em scripts/04e_train_rrin_star.py.")
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
        """feat: (B, M, C, D, H, W). ensemble_mask: (B, M) bool ou None
        (None == todos os M candidatos tratados como reais). Retorna o
        MESMO shape, com cada candidato atualizado por atencao sobre os
        demais (por voxel)."""
        b, m, c, d, h, w = feat.shape
        # (B,M,C,D,H,W) -> (B,D,H,W,M,C) -> (B*D*H*W, M, C): mesmo truque de
        # reshape/permute de CrossDirectionAttention3D (model/
        # implicit_angular.py), so trocando o eixo das n_level direcoes pelo
        # eixo dos M candidatos.
        x = feat.permute(0, 3, 4, 5, 1, 2).reshape(b * d * h * w, m, c)
        key_padding_mask = None
        if ensemble_mask is not None:
            # nn.MultiheadAttention espera True = IGNORAR essa posicao --
            # convencao INVERTIDA da nossa `ensemble_mask` (True = candidato
            # REAL) -- inverte aqui.
            pad_mask = (~ensemble_mask).unsqueeze(1).expand(b, d * h * w, m)
            key_padding_mask = pad_mask.reshape(b * d * h * w, m)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x.reshape(b, d, h, w, m, c).permute(0, 4, 5, 1, 2, 3)


class PairWeightHead3D(nn.Module):
    """Pequena rede que prediz um logit de confianca POR VOXEL para UMA
    predicao candidata (um dos M pares do ensemble em estrela), a partir do
    blend inicial (pos-warp, pre-refino) e dos dois volumes de entrada CRUS
    do par -- mesmos 3 canais de entrada de RefineNet3D (ver model/rrin3d.py),
    mesmo espirito do `layer_logit` de FlowNet3DLayered (que decide entre K
    hipoteses de fluxo do MESMO par); aqui decide entre M PARES DIFERENTES.
    Saida SEM sigmoid/softmax (logit cru) -- a normalizacao entre os M pares
    (softmax mascarado) e feita em RRIN3DStar.forward, nao aqui, porque
    precisa enxergar todos os M logits e a mascara de padding ao mesmo tempo.

    `use_quality_cond` (default False, ADITIVO -- discussao pos-secao 20.9 do
    addendum, "alimentar gap_deg/residual_deg como feature explicita na
    PairWeightHead3D"): quando True, concatena `quality` (residual_deg/90,
    gap_deg/90 -- MESMA normalizacao/convencao ja usada por
    RRIN3D/FlowNet3D.use_quality_cond, ver model/rrin3d.py) como 2 canais
    constantes extras, via `_repeat_vec_3d` (mesmo mecanismo de broadcast
    ja usado la). Motivacao: em vez de a cabeca inferir a confiabilidade de
    cada par so pelo conteudo de imagem, ela recebe tambem o sinal
    GEOMETRICO conhecido que a secao 20.6 do addendum ja mostrou
    correlacionar com o peso aprendido (par de menor `residual_deg`/
    `gap_deg` tende a ser mais confiavel) -- parametro INDEPENDENTE do
    `use_quality_cond` do FlowNet3D (podem ligar/desligar em combinacoes
    quaisquer; por isso o nome do parametro aqui e o mesmo, mas o dono e
    outro modulo -- ver RRIN3DStar.weight_quality_cond, que os desacopla).

    `cross_candidate_attention`/`cross_candidate_attn_heads` (default False/4,
    ADITIVO -- item 1, ver CrossCandidateAttention3D acima e a ATUALIZACAO
    2026-09-14 na docstring do modulo): quando True, insere um bloco de
    self-attention entre os M candidatos logo apos a primeira camada
    (`self.net[0]`) e ANTES da camada de saida (`self.net[1]`) que produz o
    logit. Como `self.net[-1]` continua zero-init (ver abaixo), a saida de
    um modelo RECEM-CRIADO continua identicamente 0 (pi uniforme)
    independente desta flag -- so muda o que a rede PODE aprender depois,
    nao o ponto de partida (mesmo espirito do zero-init de sempre).

    REGRESSAO DE 2026-09-14, CORRIGIDA (registrar para nao repetir): ao
    implementar o item 1 esta classe foi reescrita trocando o
    `self.net = nn.Sequential(...)` por dois atributos nomeados
    (`self.proj`/`self.out_conv`). Isso RENOMEOU os pesos no state_dict
    (`net.0.*`/`net.1.*` -> `proj.*`/`out_conv.*`) e quebrou o
    carregamento de TODOS os checkpoints star ja treinados -- descoberto
    quando `scripts/15_diagnose_star_fusion_ceiling.py` falhou com
    "Missing key(s) ... weight_head.proj.0.weight / Unexpected key(s) ...
    weight_head.net.0.0.weight" num checkpoint de 99 epocas. A mesma
    disciplina de preservar nomes tinha sido aplicada corretamente em
    `RefineNet3D` e em `ImplicitDecoderHead3D` (que usam `nn.ModuleList`
    justamente para manter as chaves `net.N.*`), e falhou so aqui. O fix:
    voltar a `self.net`, agora como `nn.ModuleList` (mesmas chaves de um
    `Sequential`), mais um `_load_from_state_dict` que remapeia os
    checkpoints gravados na janela curta em que a versao quebrada esteve
    em uso. Ha teste de regressao no `_smoke_test` conferindo o CONJUNTO
    EXATO de chaves do state_dict desta classe.

    MUDANCA DE CONVENCAO (2026-09-14, item 1): antes, `forward` recebia UM
    candidato de cada vez, achatado em B*M pelo chamador (RRIN3DStar.forward
    fazia o achatamento ANTES de chamar esta classe). Agora recebe o feixe
    INTEIRO (B,M,...) de uma vez, porque cross_candidate_attention precisa
    enxergar todos os M candidatos simultaneamente -- RRIN3DStar.forward foi
    atualizado pra fazer o achatamento/desachatamento em volta desta
    chamada em vez de dentro dela (nenhum outro chamador direto desta
    classe existe no projeto, ver grep de PairWeightHead3D)."""

    def __init__(self, base_ch: int = 16, norm_type: str = "instance",
                 use_quality_cond: bool = False,
                 cross_candidate_attention: bool = False,
                 cross_candidate_attn_heads: int = 4):
        super().__init__()
        self.use_quality_cond = use_quality_cond
        self.cross_candidate_attention = cross_candidate_attention
        in_ch = 1 + 1 + 1  # blend, vol_a, vol_b
        if use_quality_cond:
            in_ch += 2  # residual_norm, gap_norm (mesma convencao de RRIN3D/FlowNet3D)
        # ModuleList (nao Sequential) porque o forward precisa inserir a
        # atencao entre as duas camadas quando cross_candidate_attention=True.
        # CRITICO: `nn.ModuleList` registra os filhos pelo indice EXATAMENTE
        # como `nn.Sequential`, entao as chaves do state_dict continuam
        # `net.0.*`/`net.1.*` -- identicas as da versao historica desta
        # classe. Ver "REGRESSAO DE 2026-09-14" na docstring acima: uma
        # versao intermediaria usou `self.proj`/`self.out_conv`, o que
        # renomeou os pesos e quebrou TODOS os checkpoints star existentes.
        self.net = nn.ModuleList([
            _conv3d(in_ch, base_ch, norm_type=norm_type),
            nn.Conv3d(base_ch, 1, kernel_size=3, padding=1),
        ])
        self.cross_attn = None
        if cross_candidate_attention:
            self.cross_attn = CrossCandidateAttention3D(base_ch, num_heads=cross_candidate_attn_heads)
        # init zero -- ponto de partida neutro (pesos iguais entre os M
        # pares, ate a rede aprender a diferenciar), mesmo espirito da
        # inicializacao "morna" de FlowNet3D/FlowNet3DLayered (ver
        # model/rrin3d.py). So afeta treinos NOVOS (sem resume). Continua
        # valendo com cross_candidate_attention=True (ver docstring acima).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Compatibilidade com os checkpoints salvos pela versao
        intermediaria de 2026-09-14 (entre o item 1 e o fix), que gravou
        `proj.*`/`out_conv.*` em vez de `net.0.*`/`net.1.*`. Remapeia na
        leitura, entao tanto os checkpoints HISTORICOS quanto os dessa
        janela curta carregam sem intervencao manual."""
        for old, new in (("proj.", "net.0."), ("out_conv.", "net.1.")):
            stale = [k for k in state_dict if k.startswith(prefix + old)]
            for k in stale:
                state_dict[prefix + new + k[len(prefix + old):]] = state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, blend, vol_a, vol_b, quality=None, ensemble_mask=None):
        """blend, vol_a, vol_b: (B,M,1,D,H,W) -- o feixe INTEIRO de
        candidatos de uma vez (ver "MUDANCA DE CONVENCAO" na docstring da
        classe). quality: (B,M,2) ou None. ensemble_mask: (B,M) bool ou
        None -- usado so se cross_candidate_attention=True (ver
        CrossCandidateAttention3D). Retorna (B,M,1,D,H,W), logit cru (sem
        sigmoid/softmax -- normalizacao entre os M candidatos e feita em
        RRIN3DStar.forward)."""
        b, m = blend.shape[0], blend.shape[1]
        parts = [blend, vol_a, vol_b]
        if self.use_quality_cond:
            if quality is None:
                raise ValueError(
                    "PairWeightHead3D.use_quality_cond=True mas `quality` nao foi passado ao "
                    "forward -- ver RRIN3DStar.weight_quality_cond/docstring do modulo.")
            spatial = blend.shape[-3:]
            quality_flat = quality.reshape(b * m, quality.shape[-1])
            quality_map = _repeat_vec_3d(quality_flat, spatial).reshape(
                b, m, quality.shape[-1], *spatial)  # (B,2) -> (B*M,2,D,H,W) -> (B,M,2,D,H,W)
            parts.append(quality_map)
        x = torch.cat(parts, dim=2)                          # (B,M,in_ch,D,H,W)
        x_flat = x.reshape(b * m, x.shape[2], *x.shape[3:])
        feat_flat = self.net[0](x_flat)                        # (B*M,base_ch,D,H,W)
        if self.cross_attn is not None:
            feat = feat_flat.reshape(b, m, feat_flat.shape[1], *feat_flat.shape[2:])
            feat = self.cross_attn(feat, ensemble_mask=ensemble_mask)
            feat_flat = feat.reshape(b * m, feat.shape[2], *feat.shape[3:])
        logit_flat = self.net[1](feat_flat)                    # (B*M,1,D,H,W)
        return logit_flat.reshape(b, m, 1, *logit_flat.shape[2:])


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
        par (usado se use_quality_cond=True e/ou weight_quality_cond=True --
        MESMO tensor de entrada serve aos dois, ver docstring de
        weight_quality_cond abaixo).
    retorna: (B, 1, D, H, W) -- direcao-alvo predita (fusao dos <=M pares).

    weight_quality_cond (default False, ADITIVO -- ver PairWeightHead3D):
        DESACOPLADO de `use_quality_cond` -- este ultimo condiciona o
        FlowNet3D (a estimativa de fluxo em si, ja existente desde a
        primeira versao desta classe); `weight_quality_cond` condiciona a
        `PairWeightHead3D` (a escolha de QUANTO CONFIAR em cada par na
        fusao). Sao ortogonais por design -- qualquer uma das 4 combinacoes
        (False/False, True/False, False/True, True/True) e valida, dado
        que o `quality` de entrada (mesmo tensor, ja calculado por
        `utils/rrin_dataset.py:RRINTripletDataset._ensemble_tensors`
        independente de qualquer uma das duas flags) e passado para o
        forward sempre que QUALQUER uma das duas estiver ligada.
    """

    def __init__(self, base_ch: int = 16, max_disp: float = 0.5, use_quality_cond: bool = False,
                 norm_type: str = "instance", weight_quality_cond: bool = False,
                 cross_candidate_attention: bool = False, cross_candidate_attn_heads: int = 4,
                 refine_base_ch: int = None, refine_depth: int = 2, refine_cond: bool = False):
        super().__init__()
        self.use_quality_cond = use_quality_cond
        self.weight_quality_cond = weight_quality_cond
        self.norm_type = norm_type
        self.cross_candidate_attention = cross_candidate_attention
        self.cross_candidate_attn_heads = cross_candidate_attn_heads
        # refine_base_ch=None => segue base_ch (comportamento historico, byte-a-byte).
        self.refine_base_ch = refine_base_ch if refine_base_ch is not None else base_ch
        self.refine_depth = refine_depth
        self.refine_cond = refine_cond
        self.flow_net = FlowNet3D(base_ch=base_ch, max_disp=max_disp,
                                   use_quality_cond=use_quality_cond, norm_type=norm_type)
        self.refine_net = RefineNet3D(base_ch=self.refine_base_ch, norm_type=norm_type,
                                       depth=refine_depth,
                                       cond_ch=REFINE_COND_CH if refine_cond else 0)
        self.weight_head = PairWeightHead3D(
            base_ch=base_ch, norm_type=norm_type, use_quality_cond=weight_quality_cond,
            cross_candidate_attention=cross_candidate_attention,
            cross_candidate_attn_heads=cross_candidate_attn_heads)

    def forward(self, vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=None,
                return_pairs=False):
        b, m = vol_a.shape[0], vol_a.shape[1]

        def _flat(x):
            return x.reshape(b * m, *x.shape[2:])

        def _unflat(x):
            return x.reshape(b, m, *x.shape[1:])

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
        refine_cond_f = None
        if self.refine_cond:
            refine_cond_f = _refine_cond_vector(bvec_a_f, bvec_b_f, bvec_t_f, t_f, quality_f)
        residual_f = self.refine_net(blend_f, vol_a_f, vol_b_f, cond=refine_cond_f)
        pred_f = blend_f + residual_f                          # (B*M,1,D,H,W)
        pred = _unflat(pred_f)                                 # (B,M,1,D,H,W)

        # weight_head agora recebe o feixe INTEIRO (B,M,...) de uma vez (ver
        # "MUDANCA DE CONVENCAO" em PairWeightHead3D.__doc__, item 1,
        # 2026-09-14) -- precisa ver todos os M candidatos simultaneamente
        # pra poder rodar cross_candidate_attention entre eles. `quality`
        # (nao `quality_f`) e passado direto, ja no formato (B,M,2) que a
        # nova assinatura espera.
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
            # --residual-l2-weight em scripts/04e_train_rrin_star.py (addendum
            # 2026-09-13): exposto aqui so quando return_pairs=True, entao
            # nenhum chamador existente (que usa o default False) e afetado.
            return out, {"pred": pred, "pi": pi, "residual": residual}
        return out


def build_star_model(base_ch: int = 16, max_disp: float = 0.5, use_quality_cond: bool = False,
                      norm_type: str = "instance", weight_quality_cond: bool = False,
                      cross_candidate_attention: bool = False,
                      cross_candidate_attn_heads: int = 4,
                      refine_base_ch: int = None, refine_depth: int = 2,
                      refine_cond: bool = False) -> RRIN3DStar:
    """Wrapper trivial (mesmo espirito de build_rrin_model em model/rrin3d.py)
    -- existe so para scripts/04e_train_rrin_star.py e
    scripts/05f_reconstruct_rrin_star.py nao precisarem instanciar a classe
    diretamente, deixando espaco para uma futura variante (ex. K camadas
    tambem dentro do ensemble em estrela) sem mudar a assinatura dos scripts
    que chamam esta funcao."""
    return RRIN3DStar(base_ch=base_ch, max_disp=max_disp, use_quality_cond=use_quality_cond,
                       norm_type=norm_type, weight_quality_cond=weight_quality_cond,
                       cross_candidate_attention=cross_candidate_attention,
                       cross_candidate_attn_heads=cross_candidate_attn_heads,
                       refine_base_ch=refine_base_ch, refine_depth=refine_depth,
                       refine_cond=refine_cond)


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

    # weight_quality_cond (novo, DESACOPLADO de use_quality_cond -- ver docstring do
    # modulo/RRIN3DStar): testa as 4 combinacoes de (use_quality_cond, weight_quality_cond),
    # incluindo as 2 mistas (uma ligada, outra nao) pra confirmar que sao independentes.
    for uqc in (False, True):
        for wqc in (False, True):
            model_mix = build_star_model(base_ch=8, use_quality_cond=uqc,
                                          weight_quality_cond=wqc)
            out_mix = model_mix(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                                 quality=quality)
            assert out_mix.shape == expected, \
                f"shape mismatch (use_quality_cond={uqc}, weight_quality_cond={wqc}): " \
                f"{out_mix.shape} != {expected}"
    print("OK: as 4 combinacoes de (use_quality_cond, weight_quality_cond) rodam sem erro")

    # weight_quality_cond=True mas esquecendo `quality` no forward deve levantar erro
    # claro (mesmo padrao de FlowNet3D.use_quality_cond em model/rrin3d.py), nao um
    # erro criptico de shape dentro da PairWeightHead3D.
    model_wqc = build_star_model(base_ch=8, use_quality_cond=False, weight_quality_cond=True)
    try:
        model_wqc(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=None)
        raise AssertionError("deveria ter levantado ValueError sem `quality`")
    except ValueError:
        print("OK: weight_quality_cond=True sem `quality` levanta ValueError, como esperado")

    # zero-init da PairWeightHead3D (mesmo espirito do FlowNet3D, ver docstring da classe):
    # com weight_quality_cond=True mas pesos recem-inicializados, os 2 canais extras de
    # quality nao deveriam ainda influenciar a saida (logit continua 0 em toda parte,
    # pi uniforme) -- confere que a mudanca de quality nao muda pi num modelo NOVO.
    model_wqc2 = build_star_model(base_ch=8, use_quality_cond=False, weight_quality_cond=True)
    _, extra_a = model_wqc2(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                             quality=torch.zeros(b, m, 2), return_pairs=True)
    _, extra_b = model_wqc2(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                             quality=torch.ones(b, m, 2), return_pairs=True)
    assert torch.allclose(extra_a["pi"], extra_b["pi"]), \
        "com a ultima camada da PairWeightHead3D zero-init, pi nao deveria mudar com quality"
    print("OK: zero-init da PairWeightHead3D preserva pi uniforme mesmo com quality diferente")

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

    # ITEM 1 (2026-09-14): cross_candidate_attention -- self-attention entre
    # os M candidatos ANTES do logit de confianca (ver PairWeightHead3D/
    # CrossCandidateAttention3D). Testa: (a) shape, (b) zero-init do
    # out_conv continua preservando pi uniforme/mascara exata mesmo com a
    # atencao ligada, (c) equivariancia a permutacao dos M candidatos, (d)
    # independencia (uma posicao de PADDING nao pode vazar pro logit dos
    # candidatos REAIS), (e) ValueError cedo se cross_candidate_attn_heads
    # nao divide base_ch.
    model_cattn = build_star_model(base_ch=8, cross_candidate_attention=True,
                                    cross_candidate_attn_heads=4)
    out_cattn = model_cattn(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask)
    assert out_cattn.shape == expected, \
        f"shape mismatch (cross_candidate_attention=True): {out_cattn.shape} != {expected}"
    print(f"smoke test OK (cross_candidate_attention=True), output shape: {tuple(out_cattn.shape)}")

    _, extra_cattn = model_cattn(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                                  return_pairs=True)
    pi_cattn = extra_cattn["pi"]
    real_pi = pi_cattn[0, :2]  # 2 posicoes reais do 1o item do batch (a 3a e mascarada)
    assert torch.allclose(real_pi, real_pi[:1].expand_as(real_pi), atol=1e-5), \
        "com net[-1] zero-init, pi deveria ficar uniforme entre candidatos reais mesmo com " \
        "cross_candidate_attention=True"
    masked_pi_cattn = pi_cattn[0, -1]
    assert torch.allclose(masked_pi_cattn, torch.zeros_like(masked_pi_cattn)), \
        "posicao mascarada do feixe deveria ter peso de fusao exatamente 0 mesmo com " \
        "cross_candidate_attention=True"
    print("OK: cross_candidate_attention=True com net[-1] zero-init preserva pi uniforme "
          "entre candidatos reais e zero na posicao mascarada")

    # equivariancia a permutacao: permutar a ordem dos M candidatos de
    # entrada deve permutar pi/pred na MESMA ordem, sem mudar o valor de
    # "out" (fusao final) -- ativa out_conv com pesos NAO-zero pra este
    # teste ser sensivel a bugs de reshape/permute (com zero-init, tudo
    # daria 0/uniforme independente de qualquer bug de reshape).
    torch.manual_seed(1)
    model_perm = build_star_model(base_ch=8, cross_candidate_attention=True,
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

    # independencia entre candidatos: perturbar o CONTEUDO de uma posicao
    # MASCARADA (padding) do feixe nao pode mudar o logit/pi dos candidatos
    # REAIS -- bug classico se o key_padding_mask nao for aplicado
    # corretamente dentro de CrossCandidateAttention3D.
    vol_a_pert = vol_a.clone()
    vol_a_pert[0, -1] += 100.0  # so a posicao mascarada do 1o item do batch
    _, extra_pert = model_perm(vol_a_pert, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                                return_pairs=True)
    assert torch.allclose(extra_orig["pi"][0, :2], extra_pert["pi"][0, :2], atol=1e-4), \
        "perturbar o conteudo de uma posicao MASCARADA do feixe nao deveria mudar pi dos " \
        "candidatos reais (key_padding_mask deveria impedir essa fuga de informacao)"
    print("OK: perturbar uma posicao mascarada (padding) nao vaza pro pi dos candidatos reais")

    try:
        build_star_model(base_ch=8, cross_candidate_attention=True,
                          cross_candidate_attn_heads=3)
        raise AssertionError("deveria ter levantado ValueError (8 nao e divisivel por 3)")
    except ValueError:
        print("OK: cross_candidate_attn_heads que nao divide base_ch levanta ValueError cedo")


    # GUARDA DE NOMES DO state_dict (2026-09-14, apos a regressao descrita na
    # docstring de PairWeightHead3D): a lista abaixo e o conjunto EXATO de
    # chaves que essa cabeca sempre teve. Se alguem reestruturar a classe e
    # renomear os pesos, TODOS os checkpoints star ja treinados param de
    # carregar -- e o erro so aparece la na frente, num script de
    # reconstrucao/diagnostico, nao aqui. Este teste transforma isso numa
    # falha imediata e obvia.
    head_default = build_star_model(base_ch=16).weight_head
    got = sorted(k for k, _ in head_default.named_parameters())
    esperado = sorted(["net.0.0.weight", "net.0.0.bias", "net.0.1.weight", "net.0.1.bias",
                        "net.1.weight", "net.1.bias"])
    assert got == esperado, (
        f"as chaves do state_dict de PairWeightHead3D mudaram!\n"
        f"  esperado: {esperado}\n  obtido:   {got}\n"
        f"Isso INVALIDA todos os checkpoints star ja treinados -- ver a nota de regressao "
        f"na docstring da classe antes de mudar qualquer coisa aqui.")
    print(f"OK: PairWeightHead3D preserva as {len(got)} chaves historicas do state_dict")

    # e o remapeamento de compatibilidade da janela quebrada funciona?
    quebrado = {}
    for k, v in head_default.state_dict().items():
        novo = k.replace("net.0.", "proj.").replace("net.1.", "out_conv.")
        quebrado[novo] = v
    head_novo = build_star_model(base_ch=16).weight_head
    head_novo.load_state_dict(quebrado)   # nao pode levantar
    for k, v in head_default.state_dict().items():
        assert torch.allclose(head_novo.state_dict()[k], v), f"remapeamento errou em {k}"
    print("OK: checkpoints da janela quebrada (proj./out_conv.) sao remapeados no load")

    # CAPACIDADE DO REFINO (2026-09-14, ver addendum de capacidade): testa
    # --refine-base-ch/--refine-depth/--refine-cond.
    #
    # (a) GUARDA DE RETROCOMPATIBILIDADE -- com os defaults, a RefineNet3D
    # precisa ter EXATAMENTE os 8.737 parametros historicos em base_ch=16.
    # Se este teste quebrar, todo checkpoint ja treinado (desta linha, da
    # AMT3D, da HFD3D e da PairFlow) para de carregar.
    model_default = build_star_model(base_ch=16)
    n_refine_default = sum(p.numel() for p in model_default.refine_net.parameters())
    assert n_refine_default == 8737, \
        (f"RefineNet3D com defaults deveria manter os 8.737 parametros historicos em "
         f"base_ch=16, obtido {n_refine_default} -- isso INVALIDA checkpoints existentes")
    print(f"OK: RefineNet3D com defaults preserva os {n_refine_default} parametros historicos")

    # (b) forward com refino maior/mais fundo/condicionado
    model_refine = build_star_model(base_ch=8, refine_base_ch=32, refine_depth=4,
                                     refine_cond=True)
    out_refine = model_refine(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask,
                               quality=quality)
    assert out_refine.shape == expected, f"shape mismatch (refino maior): {out_refine.shape}"
    n_refine_big = sum(p.numel() for p in model_refine.refine_net.parameters())
    assert n_refine_big > 10 * n_refine_default, \
        "refine_base_ch=32/depth=4 deveria aumentar bastante a capacidade do refino"
    print(f"smoke test OK (refine_base_ch=32, refine_depth=4, refine_cond=True): "
          f"refine_net com {n_refine_big} parametros ({n_refine_big / n_refine_default:.1f}x o default)")

    # (c) refine_cond=True sem `quality` NAO deve quebrar (os 2 canais de
    # qualidade entram zerados -- o shape dos pesos nao pode depender de qual
    # flag de qualidade esta ligada, ver _refine_cond_vector).
    out_nq = model_refine(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t, ensemble_mask, quality=None)
    assert out_nq.shape == expected
    print("OK: refine_cond=True sem `quality` roda (canais de qualidade entram zerados)")

    # (d) o condicionamento precisa REALMENTE chegar na saida -- mudar so' o
    # bvec do alvo tem que mudar a predicao. Sem isso, a flag estaria ligada
    # mas inerte (classe de bug ja vista nesta linha: flag "ativada" no log
    # sem efeito real, ver addendum 2026-09-04 secoes 33.21/33.24).
    bvec_t2 = rand_bvecs((b, 1, 3)).expand(b, m, 3).contiguous()
    out_t2 = model_refine(vol_a, vol_b, bvec_a, bvec_b, bvec_t2, t, ensemble_mask,
                           quality=quality)
    assert not torch.allclose(out_refine, out_t2, atol=1e-6), \
        "com refine_cond=True, mudar bvec_t deveria mudar a predicao (condicionamento inerte?)"
    # e a mesma troca NAO pode mudar nada num modelo com refine_cond=False
    # alem do que o proprio flow_net ja faz -- aqui so' confirmamos que o
    # caminho novo e o unico responsavel pela diferenca acima.
    print("OK: refine_cond=True realmente altera a saida ao mudar bvec_t (condicionamento ativo)")

    try:
        build_star_model(base_ch=8, refine_depth=0)
        raise AssertionError("deveria ter levantado ValueError (refine_depth=0)")
    except ValueError:
        print("OK: refine_depth < 1 levanta ValueError cedo")

    print("Todos os smoke tests de model/rrin3d_star.py passaram.")


if __name__ == "__main__":
    _smoke_test()