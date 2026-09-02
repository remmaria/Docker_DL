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
    direcao na esfera, nao um pixel 2D)."""

    def __init__(self, state_ch: int, sh_dim: int, base_ch: int = 16,
                 norm_type: str = "instance"):
        super().__init__()
        in_ch = state_ch + sh_dim
        self.net = nn.Sequential(
            _conv3d(in_ch, base_ch, norm_type=norm_type),
            nn.Conv3d(base_ch, 1, kernel_size=3, padding=1),
        )

    def forward(self, state: torch.Tensor, sh_code: torch.Tensor) -> torch.Tensor:
        """state: (N, state_ch, D, H, W) -- N = B*N_out (ja repetido, ver
        ImplicitAngularModel3D.decode). sh_code: (N, sh_dim). Retorna
        (N, 1, D, H, W)."""
        spatial = state.shape[-3:]
        code_map = _repeat_vec_3d(sh_code, spatial)
        x = torch.cat([state, code_map], dim=1)
        return self.net(x)


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
                 norm_type: str = "instance"):
        super().__init__()
        self.n_level = n_level
        self.l_max = l_max if l_max is not None else max_order_for_n_directions(n_level)
        self.sh_dim = sh_dim_for_lmax(self.l_max)
        self.base_ch = base_ch
        self.norm_type = norm_type

        self.per_dir_encoder = PerDirectionEncoder3D(self.sh_dim, base_ch=base_ch,
                                                       norm_type=norm_type)
        self.trunk = SpatialTrunk3D(base_ch, base_ch=base_ch, norm_type=norm_type)
        self.decoder_head = ImplicitDecoderHead3D(base_ch, self.sh_dim, base_ch=base_ch,
                                                    norm_type=norm_type)

    def encode(self, input_vols: torch.Tensor, input_bvecs: torch.Tensor) -> torch.Tensor:
        """input_vols: (B, n_level, 1, D, H, W); input_bvecs: (B, n_level, 3).
        Retorna o "estado" espacial (B, base_ch, D, H, W) -- ANTES de
        condicionar em qualquer direcao-alvo (mesmo papel do `state` de
        RCAE.encoder(...), ver model/rcae.py e scripts/04_train_rcae.py --
        so por isso o metodo se chama `encode`, para os scripts de
        treino/debug poderem reaproveitar o MESMO padrao de plotar
        input/target/pred/contexto que ja usam para o RCAE, sem duplicar a
        logica de visualizacao)."""
        b, n_level = input_vols.shape[0], input_vols.shape[1]
        spatial = input_vols.shape[-3:]

        sh_flat = sh_positional_encoding(
            input_bvecs.reshape(b * n_level, 3), self.l_max)          # (B*n_level, sh_dim)
        vols_flat = input_vols.reshape(b * n_level, 1, *spatial)       # (B*n_level, 1, D,H,W)
        feat_flat = self.per_dir_encoder(vols_flat, sh_flat)           # (B*n_level, base_ch, D,H,W)
        feat = feat_flat.reshape(b, n_level, self.base_ch, *spatial)

        # agregacao PERMUTATION-INVARIANT (media simples, estilo DeepSets --
        # Zaheer et al. 2017): a ORDEM das n_level direcoes de entrada nunca
        # deveria importar (nao ha nenhuma nocao de "primeira"/"ultima"
        # direcao medida, ao contrario de uma sequencia de video) -- media
        # (ou soma/max) sao as agregacoes canonicas que garantem isso por
        # construcao. Deliberadamente NAO uma ConvLSTM3D (como o encoder do
        # RCAE usa, ver model/rcae.py:ConvLSTM3D) -- alem de manter esta
        # linha independente do RCAE (decisao explicita do usuario, ver
        # addendum secao 20.8), uma LSTM processa a sequencia em ORDEM,
        # deixando de ser estritamente permutation-invariant sem embaralhar
        # a ordem de entrada a cada epoca como paliativo.
        agg = feat.mean(dim=1)                                          # (B, base_ch, D,H,W)

        return self.trunk(agg)                                          # (B, base_ch, D,H,W)

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
                          norm_type: str = "instance") -> ImplicitAngularModel3D:
    """Factory unica -- usar em scripts/04f_train_implicit.py e
    scripts/05i_reconstruct_implicit.py em vez de instanciar
    ImplicitAngularModel3D diretamente, mesmo espirito de
    `build_rrin_model`/`build_star_model` (mantem as duas pontas
    sincronizadas; o checkpoint grava l_max/base_ch/norm_type em `args`, e a
    reconstrucao le de la)."""
    return ImplicitAngularModel3D(n_level=n_level, l_max=l_max, base_ch=base_ch,
                                   norm_type=norm_type)


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


if __name__ == "__main__":
    _smoke_test()