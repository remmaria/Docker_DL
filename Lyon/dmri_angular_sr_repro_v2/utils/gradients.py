"""
Utilidades para manipulacao de esquemas de gradiente (bvals/bvecs) em dMRI.

Nao depende de dipy/nibabel para a logica central (farthest-point sampling),
apenas numpy, para poder ser testado de forma isolada. As funcoes que leem
arquivos NIfTI/bval/bvec usam nibabel/dipy e ficam isoladas no fim do arquivo.
"""
from __future__ import annotations

import numpy as np


B0_THRESHOLD = 50.0  # s/mm^2, abaixo disso consideramos volume b0


def split_shells(bvals: np.ndarray, tol: float = 100.0):
    """Agrupa bvals em shells (clusters de b-value proximos).

    Retorna dict {b_nominal: array_de_indices}. b=0 (b0s) fica na chave 0.
    """
    bvals = np.asarray(bvals, dtype=float)
    order = np.argsort(bvals)
    shells = {}
    current_key = None
    for idx in order:
        b = bvals[idx]
        if b <= B0_THRESHOLD:
            shells.setdefault(0, []).append(idx)
            continue
        if current_key is None or abs(b - current_key) > tol:
            current_key = b
        shells.setdefault(current_key, []).append(idx)
    # normaliza chaves para o valor medio de cada shell (exceto b0)
    normalized = {}
    for key, idxs in shells.items():
        idxs = np.array(idxs, dtype=int)
        if key == 0:
            normalized[0] = idxs
        else:
            mean_b = float(np.round(np.mean(bvals[idxs]), -1))  # arredonda pra dezena
            normalized[mean_b] = idxs
    return normalized


def farthest_point_sampling(bvecs: np.ndarray, n_select: int, seed_idx: int = 0,
                             sort: bool = True):
    """Seleciona um subconjunto de direcoes (indices) maximizando dispersao angular.

    bvecs: (N, 3) vetores unitarios de uma unica shell (ja filtrados, sem b0).
    n_select: quantas direcoes manter.
    seed_idx: indice inicial (dentro do array bvecs local, nao do array global).

    Usa distancia angular (1 - |cos theta|) porque direcoes antipodais (v e -v)
    sao equivalentes em dMRI (o sinal e simetrico no q-space).

    sort: quando True (default, compatibilidade com todas as chamadas
    existentes -- subsample_shell/build_subsampling_scheme), devolve os
    indices em ordem numerica crescente (nao importa a ordem pra quem so
    quer "o conjunto de entrada"). Quando False, devolve na ORDEM DE
    SELECAO (o seed primeiro, depois cada ponto mais distante do conjunto
    ja escolhido) -- usado pelo split dinamico de treino em
    utils/dataset.py, que precisa separar essa ordem em "primeiros N_in =
    entrada" / "seguintes N_out = alvo" (replicando o ShellReorder do
    paper: reamostra a divisao entrada/alvo a cada exemplo, nao so uma vez
    no dataset inteiro).
    """
    bvecs = np.asarray(bvecs, dtype=float)
    n = bvecs.shape[0]
    if n_select >= n:
        order = np.arange(n)
        if sort:
            return order
        # ainda assim tenta comecar do seed_idx pra manter alguma nocao de
        # "ordem de selecao" mesmo no caso degenerado n_select >= n
        rest = [i for i in range(n) if i != seed_idx]
        return np.array([seed_idx] + rest)
    if n_select < 1:
        raise ValueError("n_select deve ser >= 1")

    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = bvecs / norms

    selected = [seed_idx]
    # distancia minima de cada ponto ao conjunto selecionado (usando |cos| para
    # tratar antipodais como identicos)
    cos_to_seed = np.abs(unit @ unit[seed_idx])
    min_dist = 1.0 - cos_to_seed

    while len(selected) < n_select:
        min_dist[selected[-1]] = -np.inf  # nunca reescolher
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        cos_new = np.abs(unit @ unit[next_idx])
        dist_new = 1.0 - cos_new
        min_dist = np.minimum(min_dist, dist_new)

    return np.array(sorted(selected)) if sort else np.array(selected)


def subsample_shell(bvals: np.ndarray, bvecs: np.ndarray, shell_indices: np.ndarray,
                     n_select: int, seed_idx: int = 0):
    """Aplica farthest_point_sampling dentro de uma shell especifica (indices globais).

    Retorna os indices GLOBAIS (relativos ao array bvals/bvecs completo) selecionados.
    """
    local_bvecs = bvecs[shell_indices]
    local_selected = farthest_point_sampling(local_bvecs, n_select, seed_idx=seed_idx)
    return shell_indices[local_selected]


def build_subsampling_scheme(bvals: np.ndarray, bvecs: np.ndarray, n_levels: list[int],
                              tol: float = 100.0, seed_idx: int = 0):
    """Gera, para cada shell (exceto b0) e cada nivel em n_levels, os indices
    globais de direcoes de entrada (subamostradas) e o complemento (alvo/held-out).

    Retorna dict:
      {shell_b: {n_level: {"input_idx": arr, "target_idx": arr, "n_available": int}}}
    b0s sao sempre incluidos integralmente no input (nao entram na subamostragem).
    """
    shells = split_shells(bvals, tol=tol)
    scheme = {}
    for b_key, idxs in shells.items():
        if b_key == 0:
            continue
        n_available = len(idxs)
        scheme[b_key] = {}
        for n_level in n_levels:
            if n_level > n_available:
                # nivel nao aplicavel a essa shell; sinaliza para o script pular
                scheme[b_key][n_level] = {
                    "input_idx": None,
                    "target_idx": None,
                    "n_available": n_available,
                }
                continue
            input_idx = subsample_shell(bvals, bvecs, idxs, n_level, seed_idx=seed_idx)
            target_idx = np.setdiff1d(idxs, input_idx)
            scheme[b_key][n_level] = {
                "input_idx": input_idx,
                "target_idx": target_idx,
                "n_available": n_available,
            }
    return scheme


def spherical_triplet_residual(v_a: np.ndarray, v_b: np.ndarray, v_t: np.ndarray):
    """Mede o quanto v_t esta "entre" v_a e v_b num arco geodesico comum,
    tratando antipodais (v e -v) como identicos (simetria de dMRI).

    Usado para a linha RRIN/VFI-por-trincas (ver protocolo, secao 10.1):
    RRIN assume que a direcao-alvo e uma interpolacao temporal entre duas
    direcoes "vizinhas" (como quadro do meio em video), mas em q-space isso
    so faz sentido geometricamente se as tres direcoes forem aproximadamente
    colineares num grande circulo da esfera -- ao contrario de quadros de
    video, tres direcoes de gradiente quaisquer normalmente NAO sao.

    O "residuo" retornado e a distancia angular PERPENDICULAR de v_t ate o
    PLANO do grande circulo que passa por v_a e v_b (a circunferencia que
    se obtem cortando a esfera por esse plano) -- ou seja, literalmente "a
    quantos graus esse ponto esta desta linha", nao uma aproximacao. Formula:
    se n = v_a x v_b (normal ao plano, invariante a trocar o sinal de v_a
    ou v_b -- flipar qualquer um dos dois so troca o sinal de n, nao o
    plano em si, entao a simetria antipodal de a/b ja sai de graca), entao
    sin(residuo) = |v_t . n_hat|. Isto SUBSTITUI uma versao anterior deste
    calculo que usava o "excesso" da desigualdade triangular esferica
    (ang(a,t)+ang(t,b)-ang(a,b)) como proxy -- essa proxy SUBESTIMA bastante
    o desvio perpendicular real quanto maior o angulo entre a e b (ex.:
    com gap(a,b)=90 graus, um desvio perpendicular real de 20 graus dava um
    "residuo" antigo de so ~6.7 graus) -- exatamente o regime em que
    normalmente operamos aqui (gap(a,b) tipico observado nos dados: ~70-90
    graus, ver protocolo secao 10.1), entao a proxy antiga tornava
    --max-residual-deg bem mais permissivo do que o numero sugeria. Corrigido
    pra a distancia perpendicular exata, que e o que o nome do parametro
    sempre pretendeu dizer.

    v_t tambem precisa estar "entre" a e b ao longo do arco (nao so no MESMO
    grande circulo, que sozinho nao garante isso -- um ponto no lado oposto
    do circulo tambem teria residuo perpendicular zero) -- isso e checado
    separadamente via t_frac (ver abaixo): t_frac fora de aproximadamente
    [0,1] indica que v_t, mesmo perto do plano, NAO esta no arco menor entre
    v_a e v_b. Quem usa esta funcao (find_best_bracket) nao filtra por
    t_frac hoje -- ver ressalva no docstring de find_best_bracket.

    Retorna (residual_rad, ang_ab_rad, t_frac), onde t_frac e a posicao
    relativa (COM SINAL) de v_t no arco geodesico a->b (0 = coincide com
    v_a, 1 = com v_b, negativo = do lado errado de v_a, >1 = alem de v_b) --
    usada como o parametro de tempo `t` da interpolacao, quando a rede de
    VFI usada suportar `t` arbitrario.

    **CORRIGIDO em 2026-08-27 (bug real, nao so de visualizacao -- achado ao
    depurar scripts/07_visualize_triplet.py, ver addendum do projeto):** a
    versao anterior calculava `t_frac = arccos(dot(a,t)) / ang_ab`, um
    angulo SEM SINAL entre a e t. Isso NAO diferencia "t esta do lado de b"
    de "t esta do lado OPOSTO de b" -- um alvo a, digamos, 60 graus de `a`
    na direcao CONTRARIA a `b` (com ang(a,b)=84 graus) dava
    `t_frac=60/84=0.71`, dentro de [0,1], e portanto `between=True` mesmo
    NAO estando geometricamente entre a e b (confirmado numericamente:
    a=(1,0,0), b=(cos84,sin84,0), t=(cos60,-sin60,0) da residual=0,
    t_frac=0.714, quando t esta do lado oposto de b). Corrigido calculando
    o angulo COM SINAL de t em relacao a a, medido no MESMO sentido de
    rotacao de a para b (base ortonormal (a, e2) no plano do grande
    circulo, e2 = componente de b perpendicular a a): `theta_t =
    atan2(proj_e2(t), proj_a(t))`, `t_frac = theta_t/ang_ab`. Para
    qualquer t genuinamente entre a e b (o caso que a formula antiga JA
    acertava), o valor numerico de t_frac fica identico ao de antes --
    a correcao so muda o resultado exatamente nos casos que antes eram
    classificados errado.
    """
    a = np.asarray(v_a, dtype=float)
    b = np.asarray(v_b, dtype=float)
    t = np.asarray(v_t, dtype=float)
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    t = t / (np.linalg.norm(t) or 1.0)

    if np.dot(a, b) < 0:
        b = -b
    if np.dot(a, t) < 0:
        t = -t

    ang_ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))

    # distancia perpendicular real de t ao PLANO que passa por a e b (e pela
    # origem) -- n = a x b e a normal desse plano; sin(residuo) = |t . n_hat|.
    # Caso degenerado (a e b quase paralelos/coincidentes, |n|~0): o "plano
    # que passa por a e b" fica mal definido (infinitos planos contem dois
    # pontos quase coincidentes) -- trata como residuo maximo (90 graus),
    # sinalizando "este par nao serve de referencia geometrica" em vez de
    # dividir por quase-zero.
    n = np.cross(a, b)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        residual = np.pi / 2
        t_frac = 0.0
    else:
        n_hat = n / n_norm
        residual = np.arcsin(np.clip(abs(np.dot(t, n_hat)), 0.0, 1.0))

        # base ortonormal (a, e2) no plano do grande circulo a-b: e2 e a
        # componente de b perpendicular a a, normalizada -- por construcao
        # b = cos(ang_ab)*a + sin(ang_ab)*e2 com sin(ang_ab)>=0 (ang_ab em
        # [0,pi/2] apos o sign-fix acima), entao "andar de a em direcao a
        # b" e sempre o sentido POSITIVO de e2.
        e2 = np.cross(n_hat, a)
        e2_norm = np.linalg.norm(e2)
        if e2_norm > 1e-12:
            e2 = e2 / e2_norm
        # projeta t no plano (remove a componente fora do plano -- pequena
        # se residual for baixo, e o unico caso em que t_frac importa de
        # verdade) e mede o angulo COM SINAL a partir de a, no sentido de b.
        t_in_plane = t - np.dot(t, n_hat) * n_hat
        theta_t = np.arctan2(np.dot(t_in_plane, e2), np.dot(t_in_plane, a))
        t_frac = float(theta_t / ang_ab) if ang_ab > 1e-8 else 0.0

    return float(residual), float(ang_ab), t_frac


def find_best_bracket(candidate_bvecs: np.ndarray, target_bvec: np.ndarray,
                       max_residual_deg: float | None = None,
                       require_between: bool = True):
    """Entre todos os pares (i,j) de candidate_bvecs, acha o par que melhor
    "abraca" target_bvec num arco geodesico comum (baixo residuo de
    colinearidade, ver spherical_triplet_residual) -- e, ENTRE os pares
    aceitaveis, o mais "parecido com vizinhos de video" (menor gap_deg).

    candidate_bvecs: (M,3), tipicamente as direcoes de ENTRADA disponiveis
    (input_idx de um scheme.npz) para uma dada shell/n_level -- ou seja, a
    mesma informacao que o RCAE recebe nesse nivel de subamostragem, para
    manter a comparacao justa entre metodos.

    max_residual_deg: quando informado, a selecao vira em DUAS etapas: (1)
    filtra os pares com residual_deg <= max_residual_deg (os "validos" pelo
    mesmo criterio usado por scripts/02b_build_rrin_triplets.py); (2) entre
    esses, escolhe o de MENOR gap_deg (par mais proximo entre si), nao mais
    o de menor residuo absoluto. Motivacao (ver protocolo secao 10.1): um
    levantamento nos dados reais mostrou que, mesmo entre trincas "validas"
    (residuo baixo), o gap_deg mediano fica perto do maximo teorico (~90,
    limite da simetria antipodal), quase nao caindo com n_level maior --
    ou seja, so minimizar residuo tende a escolher pares bem AFASTADOS
    (que por acaso caem quase colineares com o alvo), nao pares "vizinhos"
    no sentido de video (deslocamento pequeno). Preferir o menor gap_deg
    ENTRE os validos da a hipotese de fluxo a melhor chance disponivel nos
    dados -- se mesmo assim o gap tipico continuar alto, e porque pares
    realmente proximos e colineares com o alvo raramente existem nesse
    esquema de gradiente, nao um artefato da escolha de selecao. Se nenhum
    par passar no teto, cai no fallback de minimizar o residuo global
    (mesmo comportamento de antes, sem o parametro) -- quem chama decide
    se trata isso como invalido (e o que 02b_build_rrin_triplets.py faz).

    require_between: (default True -- ATENCAO, muda o comportamento default
    em relacao a versoes anteriores desta funcao) exige, entre os pares
    aceitaveis pelo teto de residuo, que o alvo esteja de fato ENTRE a e b
    no arco (0 <= t_frac <= 1), nao so no mesmo plano/grande circulo.
    Motivo: residuo baixo garante colinearidade (mesmo plano), mas NAO
    garante que o alvo esteja "no meio" do par -- um alvo fora do arco
    (t_frac<0 ou >1) esta sendo EXTRAPOLADO a partir do par, nao
    interpolado entre eles, o que quebra a premissa de "quadro do meio"
    que da sentido a analogia com VFI (RRIN/RIFE sao treinadas para
    interpolar t em [0,1], nao extrapolar). Achado empirico que motivou
    isso (ver protocolo secao 10.2): ao minimizar gap_deg apenas entre
    pares colineares (sem checar t_frac), a selecao converge para pares
    CADA VEZ MAIS PROXIMOS conforme n_level cresce (bom), mas o t_frac
    mediano do par escolhido ULTRAPASSA 1 ja em n_level>=15 e chega a ~3 em
    n_level=50 -- ou seja, a maioria dos "melhores pares apertados"
    encontrados dessa forma na verdade extrapolam bem para fora do
    segmento (a,b), nao interpolam. Com require_between=True, a busca
    prioriza: (1) pares com residuo <= teto E 0<=t_frac<=1 (interpolacao
    genuina), escolhendo entre esses o de menor gap_deg; (2) se nenhum
    pareamento colinear tiver o alvo entre os dois candidatos, cai para o
    mesmo fallback de antes (menor gap_deg entre os aceitaveis por
    residuo, mesmo que extrapole) e marca isso no campo "between"=False do
    retorno, para quem consome poder filtrar/reportar separadamente. Passe
    False para reproduzir o comportamento anterior (so residuo+gap, sem
    checar betweenness) -- usado por quem quiser comparar as duas versoes.

    Retorna dict com indices LOCAIS i,j (relativos a candidate_bvecs),
    residual_deg, gap_deg (=ang_ab em graus), t_frac e between (bool,
    0<=t_frac<=1 do par retornado -- SEMPRE presente, independente de
    require_between ter encontrado um par "between" ou caido no
    fallback). Levanta ValueError se candidate_bvecs tiver menos de 2
    direcoes.
    """
    candidate_bvecs = np.asarray(candidate_bvecs, dtype=float)
    m = candidate_bvecs.shape[0]
    if m < 2:
        raise ValueError("find_best_bracket precisa de pelo menos 2 direcoes candidatas")

    candidates = []
    for i in range(m):
        for j in range(i + 1, m):
            residual, ang_ab, t_frac = spherical_triplet_residual(
                candidate_bvecs[i], candidate_bvecs[j], target_bvec)
            candidates.append((residual, i, j, ang_ab, t_frac))

    def _is_between(c):
        return 0.0 <= c[4] <= 1.0

    if max_residual_deg is not None:
        max_residual_rad = np.radians(max_residual_deg)
        acceptable = [c for c in candidates if c[0] <= max_residual_rad]
        if acceptable:
            chosen_pool = acceptable
            if require_between:
                between_pool = [c for c in acceptable if _is_between(c)]
                if between_pool:
                    chosen_pool = between_pool
                # senao (nenhum aceitavel tem o alvo entre os dois): cai
                # para todos os aceitaveis mesmo, so pra ter uma resposta
                # (marcado between=False abaixo).
            # entre o pool escolhido, menor gap_deg (ang_ab); empate
            # quebrado pelo menor residuo.
            residual, i, j, ang_ab, t_frac = min(chosen_pool, key=lambda c: (c[3], c[0]))
        else:
            # nenhum par passa no teto -- fallback: menor residuo global
            # (comportamento identico ao de antes deste parametro existir),
            # pra quem chama ainda ter o "menos pior" e poder marcar invalido.
            residual, i, j, ang_ab, t_frac = min(candidates, key=lambda c: c[0])
    else:
        residual, i, j, ang_ab, t_frac = min(candidates, key=lambda c: c[0])

    return {
        "i": i, "j": j,
        "residual_deg": float(np.degrees(residual)),
        "gap_deg": float(np.degrees(ang_ab)),
        "t_frac": t_frac,
        "between": bool(0.0 <= t_frac <= 1.0),
    }


def find_best_bracket_batch(candidate_bvecs: np.ndarray, target_bvecs: np.ndarray,
                             max_residual_deg: float | None = None,
                             require_between: bool = True):
    """Equivalente VETORIZADO de chamar find_best_bracket uma vez por linha
    de target_bvecs (mesmos candidate_bvecs para todos os alvos) -- usado
    por scripts/02b_build_rrin_triplets.py, que precisava disso pra cada
    (shell,n_level,sujeito) de um dataset de ~1000 sujeitos e ficava lento
    de mais (o loop Python duplo -- pares x alvos -- de find_best_bracket
    reavaliava do zero, PARA CADA ALVO, todo par (i,j), incluindo os
    produtos vetoriais/normalizacoes de spherical_triplet_residual, que sao
    baratos individualmente mas o overhead de chamada numpy em vetores de
    3 elementos domina quando repetido milhoes de vezes).

    Ideia da vetorizacao: com a convencao de sinal antipodal usada em
    spherical_triplet_residual, da pra mostrar que:
      - ang_ab (angulo entre os dois candidatos de um par) NAO depende do
        alvo -- calculavel UMA VEZ para todos os pares (i,j), i<j.
      - o plano/normal de cada par (usado no residuo E na base do t_frac
        com sinal, ver abaixo) tambem nao depende do alvo -- so o produto
        escalar final com o alvo muda.
      - dot(a_par, alvo) (usado no t_frac) so depende do candidato i (nao
        do par completo nem de j) -- calculavel de uma vez via `U @ alvos.T`
        e depois indexado por `iu`.
    Ou seja, os unicos termos que realmente cruzam pares x alvos sao
    produtos escalares -- viram produtos de matrizes (numpy BLAS) em vez
    de milhoes de chamadas Python. Depois disso, a escolha do melhor par
    por alvo e so indexacao/argmin em arrays ja prontos.

    candidate_bvecs: (M,3). target_bvecs: (K,3) -- um ou mais alvos, MESMO
    conjunto de candidatos para todos.

    Retorna dict de arrays, cada um com shape (K,): "i", "j" (indices
    LOCAIS em candidate_bvecs, um inteiro por alvo), "residual_deg",
    "gap_deg", "t_frac", "between" -- exatamente os mesmos campos e a
    MESMA semantica de find_best_bracket.

    **CORRIGIDO em 2026-08-27 junto com spherical_triplet_residual (ver
    docstring la para o bug/contraexemplo completo):** o t_frac aqui agora
    tambem usa o angulo COM SINAL (base ortonormal (a_par, e2) no plano do
    par), nao mais `arccos(dot)/ang_ab` sem sinal -- reverificado por
    equivalencia numerica contra `find_best_bracket` chamado par-a-par em
    dados aleatorios (200 conjuntos x 5 alvos, 0 divergencias) mais o
    contraexemplo especifico que expos o bug original.
    """
    candidate_bvecs = np.asarray(candidate_bvecs, dtype=float)
    target_bvecs = np.atleast_2d(np.asarray(target_bvecs, dtype=float))
    m = candidate_bvecs.shape[0]
    if m < 2:
        raise ValueError("find_best_bracket_batch precisa de pelo menos 2 direcoes candidatas")
    k_targets = target_bvecs.shape[0]

    u_norm = np.linalg.norm(candidate_bvecs, axis=1, keepdims=True)
    u_norm[u_norm == 0] = 1.0
    U = candidate_bvecs / u_norm

    t_norm = np.linalg.norm(target_bvecs, axis=1, keepdims=True)
    t_norm[t_norm == 0] = 1.0
    T = target_bvecs / t_norm

    iu, ju = np.triu_indices(m, k=1)  # mesma ordem de enumeracao de find_best_bracket
    n_pairs = iu.shape[0]

    a_pairs = U[iu]  # (n_pairs, 3) -- papel de "a" = candidato de indice menor, como no loop original
    b_raw = U[ju]    # (n_pairs, 3)
    dot_ij = np.sum(a_pairs * b_raw, axis=1)
    ang_ab = np.arccos(np.clip(np.abs(dot_ij), 0.0, 1.0))  # (n_pairs,) -- nao depende do alvo

    # b SIGN-FIXADO (dot(a,b)>=0) -- precisamos do vetor de verdade (nao so
    # do angulo sem sinal) pra montar a base ortonormal do plano usada no
    # t_frac com sinal abaixo (ver nota no docstring do modulo, correcao de
    # 2026-08-27).
    sign_b = np.where(dot_ij >= 0.0, 1.0, -1.0)
    b_pairs = b_raw * sign_b[:, None]

    cross_ij = np.cross(a_pairs, b_pairs)  # (n_pairs, 3) -- consistente com b sign-fixado
    cross_norm = np.linalg.norm(cross_ij, axis=1)
    degenerate = cross_norm < 1e-8
    n_hat = np.zeros_like(cross_ij)
    ok = ~degenerate
    n_hat[ok] = cross_ij[ok] / cross_norm[ok, None]

    # e2 = componente de b perpendicular a a, normalizada -- junto com a,
    # forma a base ortonormal do plano do grande circulo em que "andar de a
    # para b" e sempre o sentido positivo de e2 (mesma construcao de
    # spherical_triplet_residual, ver docstring la para o bug que isso
    # corrige e o contraexemplo numerico).
    e2 = np.cross(n_hat, a_pairs)  # (n_pairs, 3)
    e2_norm = np.linalg.norm(e2, axis=1)
    e2_ok = e2_norm > 1e-12
    e2_safe = np.zeros_like(e2)
    e2_safe[e2_ok] = e2[e2_ok] / e2_norm[e2_ok, None]
    e2 = e2_safe

    dot_it = U @ T.T  # (M, K) -- dot(candidato_i_bruto, alvo_bruto)
    dot_at = dot_it[iu, :]  # (n_pairs, K) -- dot(a_pairs, alvo_bruto), a_pairs = U[iu]
    sign_t = np.where(dot_at >= 0.0, 1.0, -1.0)  # sign-fix do alvo relativo a a_pairs
    comp_a = np.abs(dot_at)  # dot(alvo_sign-fixado, a_pairs) -- sempre >=0 por construcao

    dot_te_raw = e2 @ T.T  # (n_pairs, K) -- dot(e2, alvo_bruto)
    comp_e2 = sign_t * dot_te_raw  # dot(alvo_sign-fixado, e2)

    theta_t = np.arctan2(comp_e2, comp_a)  # (n_pairs, K) -- angulo COM SINAL de a para o alvo
    ang_ab_col = ang_ab[:, None]
    t_frac = np.divide(theta_t, ang_ab_col, out=np.zeros_like(theta_t), where=ang_ab_col > 1e-8)
    t_frac[degenerate, :] = 0.0  # mesmo fallback do par degenerado usado em spherical_triplet_residual

    dot_tn = n_hat @ T.T  # (n_pairs, K)
    residual = np.arcsin(np.clip(np.abs(dot_tn), 0.0, 1.0))
    residual[degenerate, :] = np.pi / 2.0  # par degenerado (i~=j): sem plano bem definido

    between = (t_frac >= 0.0) & (t_frac <= 1.0)
    gap_deg_pairs = np.degrees(ang_ab)
    residual_deg = np.degrees(residual)

    out_i = np.empty(k_targets, dtype=int)
    out_j = np.empty(k_targets, dtype=int)
    out_residual = np.empty(k_targets)
    out_gap = np.empty(k_targets)
    out_tfrac = np.empty(k_targets)
    out_between = np.empty(k_targets, dtype=bool)

    acceptable = residual_deg <= max_residual_deg if max_residual_deg is not None else None

    for k in range(k_targets):
        if acceptable is not None:
            acc_mask = acceptable[:, k]
            if acc_mask.any():
                pool_idx = np.nonzero(acc_mask)[0]
                if require_between:
                    bw_idx = pool_idx[between[pool_idx, k]]
                    if bw_idx.size:
                        pool_idx = bw_idx
                # entre o pool, menor gap_deg; empate quebrado pelo menor residuo
                order = np.lexsort((residual_deg[pool_idx, k], gap_deg_pairs[pool_idx]))
                best = pool_idx[order[0]]
            else:
                best = int(np.argmin(residual_deg[:, k]))
        else:
            best = int(np.argmin(residual_deg[:, k]))

        out_i[k] = iu[best]
        out_j[k] = ju[best]
        out_residual[k] = residual_deg[best, k]
        out_gap[k] = gap_deg_pairs[best]
        out_tfrac[k] = t_frac[best, k]
        out_between[k] = between[best, k]

    return {
        "i": out_i, "j": out_j,
        "residual_deg": out_residual,
        "gap_deg": out_gap,
        "t_frac": out_tfrac,
        "between": out_between,
    }


def find_star_ensemble_batch(candidate_bvecs: np.ndarray, target_bvecs: np.ndarray,
                              m: int, max_residual_deg: float | None = None,
                              require_between: bool = True):
    """"Ensemble em estrela" (ver protocolo secao 14.5, item 1 -- ideia
    adiada em favor da loss angular/SH da secao 15, retomada em 2026-08-27
    depois do bug critico de t_frac corrigido, ver addendum secao 12):
    em vez de devolver so o MELHOR par (a,b) por alvo (find_best_bracket_batch),
    devolve ate `m` pares DIVERSOS entre si, para depois serem combinados
    (blend/fusao aprendida, ver model/rrin3d_star.py) numa unica predicao
    por alvo -- a ideia sendo que pares com planos/normais bem diferentes
    carregam informacao geometrica mais independente sobre o alvo do que um
    unico par (ou vários pares quase-duplicados no mesmo grande circulo).

    Selecao por alvo, em duas etapas:
      1. Monta o mesmo "pool aceitavel" de find_best_bracket_batch (pares
         com residual_deg<=max_residual_deg e, se require_between, tambem
         0<=t_frac<=1 -- ou o pool so-por-residuo se nenhum pool "between"
         existir, mesmo fallback de find_best_bracket_batch). Se NENHUM par
         passar no teto de residuo, cai no MESMO fallback de
         find_best_bracket_batch (menor residuo global) preenchendo so a
         1a posicao do feixe (mask=[True, False, ..., False]) -- quem
         consome trata isso como alvo "invalido" (mesmo criterio de
         sempre, ver "mask"/"between" abaixo).
      2. Dentro do pool aceitavel, ordena por gap_deg crescente (empate:
         menor residual_deg) -- a MESMA ordem/criterio que
         find_best_bracket_batch usaria para escolher um unico par -- e
         usa o 1o (melhor gap_deg) como SEMENTE de uma
         farthest_point_sampling (ver acima) aplicada as NORMAIS dos pares
         do pool (n_i = a_i x b_i, ja calculadas aqui para o
         residuo/t_frac -- nao aos bvecs brutos): os `m` pares escolhidos
         sao o de melhor gap_deg mais os `m-1` com normais mais dispersas
         entre as ja escolhidas. Se o pool tiver <= m pares aceitaveis,
         devolve todos eles (sem FPS, nada a escolher) e marca o resto do
         feixe como padding (mask=False).

    Consequencia direta desta construcao: com m=1, o resultado e
    IDENTICO (mesmo par, mesmos campos) ao de find_best_bracket_batch
    chamada com os mesmos argumentos -- verificado numericamente (ver
    utils/gradients.py, secao de testes do modulo/addendum do projeto).

    candidate_bvecs: (M_cand,3). target_bvecs: (K,3).

    Retorna dict de arrays, todos com shape (K, m):
      "i", "j": indices LOCAIS em candidate_bvecs do par nessa posicao do
          feixe (-1 nas posicoes de padding, quando `mask` e False la).
      "residual_deg", "gap_deg", "t_frac", "between": mesmos campos e
          semantica de find_best_bracket_batch, por posicao do feixe (0.0/
          False nas posicoes de padding).
      "mask": bool, True nas posicoes com par real. SEMPRE tem pelo menos
          uma posicao True por linha (mask[:,0].all() == True) -- ou um
          par aceitavel de verdade, ou o fallback de menor residuo global
          (que quem consome deve tratar como alvo invalido do mesmo jeito
          que ja trata hoje via "valid"/"between" de find_best_bracket_batch).

    Levanta ValueError se candidate_bvecs tiver menos de 2 direcoes ou
    m < 1.
    """
    candidate_bvecs = np.asarray(candidate_bvecs, dtype=float)
    target_bvecs = np.atleast_2d(np.asarray(target_bvecs, dtype=float))
    n_cand = candidate_bvecs.shape[0]
    if n_cand < 2:
        raise ValueError("find_star_ensemble_batch precisa de pelo menos 2 direcoes candidatas")
    if m < 1:
        raise ValueError("m deve ser >= 1")
    k_targets = target_bvecs.shape[0]

    # ---- geometria pairwise identica a find_best_bracket_batch (nao repetimos
    # a explicacao aqui -- ver docstring/comentarios la, mesma formula exata). ----
    u_norm = np.linalg.norm(candidate_bvecs, axis=1, keepdims=True)
    u_norm[u_norm == 0] = 1.0
    U = candidate_bvecs / u_norm
    t_norm = np.linalg.norm(target_bvecs, axis=1, keepdims=True)
    t_norm[t_norm == 0] = 1.0
    T = target_bvecs / t_norm

    iu, ju = np.triu_indices(n_cand, k=1)
    n_pairs = iu.shape[0]
    a_pairs = U[iu]
    b_raw = U[ju]
    dot_ij = np.sum(a_pairs * b_raw, axis=1)
    ang_ab = np.arccos(np.clip(np.abs(dot_ij), 0.0, 1.0))
    sign_b = np.where(dot_ij >= 0.0, 1.0, -1.0)
    b_pairs = b_raw * sign_b[:, None]
    cross_ij = np.cross(a_pairs, b_pairs)
    cross_norm = np.linalg.norm(cross_ij, axis=1)
    degenerate = cross_norm < 1e-8
    n_hat = np.zeros_like(cross_ij)
    ok = ~degenerate
    n_hat[ok] = cross_ij[ok] / cross_norm[ok, None]
    e2 = np.cross(n_hat, a_pairs)
    e2_norm = np.linalg.norm(e2, axis=1)
    e2_ok = e2_norm > 1e-12
    e2_safe = np.zeros_like(e2)
    e2_safe[e2_ok] = e2[e2_ok] / e2_norm[e2_ok, None]
    e2 = e2_safe

    dot_it = U @ T.T
    dot_at = dot_it[iu, :]
    sign_t = np.where(dot_at >= 0.0, 1.0, -1.0)
    comp_a = np.abs(dot_at)
    dot_te_raw = e2 @ T.T
    comp_e2 = sign_t * dot_te_raw
    theta_t = np.arctan2(comp_e2, comp_a)
    ang_ab_col = ang_ab[:, None]
    t_frac = np.divide(theta_t, ang_ab_col, out=np.zeros_like(theta_t), where=ang_ab_col > 1e-8)
    t_frac[degenerate, :] = 0.0

    dot_tn = n_hat @ T.T
    residual = np.arcsin(np.clip(np.abs(dot_tn), 0.0, 1.0))
    residual[degenerate, :] = np.pi / 2.0

    between = (t_frac >= 0.0) & (t_frac <= 1.0)
    gap_deg_pairs = np.degrees(ang_ab)      # (n_pairs,) -- nao depende do alvo
    residual_deg = np.degrees(residual)     # (n_pairs, K)

    acceptable = residual_deg <= max_residual_deg if max_residual_deg is not None else None

    out_i = np.full((k_targets, m), -1, dtype=int)
    out_j = np.full((k_targets, m), -1, dtype=int)
    out_residual = np.zeros((k_targets, m))
    out_gap = np.zeros((k_targets, m))
    out_tfrac = np.zeros((k_targets, m))
    out_between = np.zeros((k_targets, m), dtype=bool)
    out_mask = np.zeros((k_targets, m), dtype=bool)

    for k in range(k_targets):
        if acceptable is not None:
            acc_mask = acceptable[:, k]
            if acc_mask.any():
                pool_idx = np.nonzero(acc_mask)[0]
                if require_between:
                    bw_idx = pool_idx[between[pool_idx, k]]
                    if bw_idx.size:
                        pool_idx = bw_idx
                # ordena o pool por gap_deg crescente (empate: menor residuo) --
                # 1o elemento = par que find_best_bracket_batch teria escolhido
                # sozinho (mesmo criterio, so que aqui como SEMENTE do feixe).
                order = np.lexsort((residual_deg[pool_idx, k], gap_deg_pairs[pool_idx]))
                pool_sorted = pool_idx[order]
            else:
                # nenhum par passa no teto -- mesmo fallback de
                # find_best_bracket_batch: so a 1a posicao do feixe e
                # preenchida (menor residuo global), resto fica padding
                # (quem consome trata isso como alvo invalido, mesmo
                # criterio de sempre).
                best = int(np.argmin(residual_deg[:, k]))
                out_i[k, 0] = iu[best]
                out_j[k, 0] = ju[best]
                out_residual[k, 0] = residual_deg[best, k]
                out_gap[k, 0] = gap_deg_pairs[best]
                out_tfrac[k, 0] = t_frac[best, k]
                out_between[k, 0] = between[best, k]
                out_mask[k, 0] = True
                continue
        else:
            # sem teto (max_residual_deg=None) -- mesmo criterio de
            # find_best_bracket_batch nesse caso: ordena TODOS os pares por
            # residuo (gap so faz sentido como criterio SECUNDARIO dentro de
            # um pool ja filtrado por teto -- sem teto, o "pool" e tudo, e o
            # criterio primario vira minimizar o proprio residuo, igual ao
            # `best = argmin(residual_deg)` do fallback de
            # find_best_bracket_batch/find_best_bracket).
            pool_idx = np.arange(n_pairs)
            order = np.argsort(residual_deg[pool_idx, k])
            pool_sorted = pool_idx[order]

        if pool_sorted.size <= m:
            chosen = pool_sorted
        else:
            normals_pool = n_hat[pool_sorted]  # (P,3)
            fps_local = farthest_point_sampling(normals_pool, m, seed_idx=0, sort=False)
            chosen = pool_sorted[fps_local]

        n_chosen = chosen.size
        out_i[k, :n_chosen] = iu[chosen]
        out_j[k, :n_chosen] = ju[chosen]
        out_residual[k, :n_chosen] = residual_deg[chosen, k]
        out_gap[k, :n_chosen] = gap_deg_pairs[chosen]
        out_tfrac[k, :n_chosen] = t_frac[chosen, k]
        out_between[k, :n_chosen] = between[chosen, k]
        out_mask[k, :n_chosen] = True

    return {
        "i": out_i, "j": out_j,
        "residual_deg": out_residual, "gap_deg": out_gap,
        "t_frac": out_tfrac, "between": out_between,
        "mask": out_mask,
    }


# ---------------------------------------------------------------------------
# I/O (depende de nibabel; import isolado para nao quebrar testes unitarios
# que só exercitam a logica numpy acima)
# ---------------------------------------------------------------------------

def load_bval_bvec(bval_path: str, bvec_path: str):
    bvals = np.loadtxt(bval_path).reshape(-1)
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    return bvals, bvecs


def load_dwi(nifti_path: str):
    import nibabel as nib  # import local: so necessario aqui
    img = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header