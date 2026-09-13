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


def _pairwise_electrostatic_energy(unit_vecs: np.ndarray) -> np.ndarray:
    """Matriz (n,n) de energia eletrostatica de Jones et al. 1999 entre pares
    de direcoes unitarias, tratando v e -v como equivalentes (antipodal-
    simetrico por construcao -- soma o termo de repulsao contra v_j E
    contra -v_j, em vez de escolher um sinal canonico como o metodo legado
    fazia, que e a fonte do bug de formula inconsistente descrito nas notas
    do projeto). Diagonal fica zero (auto-energia nao definida/nao usada).

    Porte fiel de pairwise_energy_matrix em select_pair_rep_centroid_v2.py
    (mesma formula: e = 1/|v_i-v_j|^2 + 1/|v_i+v_j|^2).
    """
    n = unit_vecs.shape[0]
    E = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d_minus = np.linalg.norm(unit_vecs[i] - unit_vecs[j])
            d_plus = np.linalg.norm(unit_vecs[i] + unit_vecs[j])
            e = 1.0 / (d_minus ** 2 + 1e-12) + 1.0 / (d_plus ** 2 + 1e-12)
            E[i, j] = E[j, i] = e
    return E


def _subset_energy(E: np.ndarray, idx_list) -> float:
    idx = list(idx_list)
    return E[np.ix_(idx, idx)].sum() / 2.0


def direction_set_quality(bvecs: np.ndarray) -> dict:
    """Diagnostico geometrico de um subconjunto de direcoes JA SELECIONADO
    (nao faz selecao nenhuma -- so mede a qualidade do que foi passado).
    Mesmas 3 metricas que select_pair_rep_centroid_v2.py imprimia no log
    (angulo minimo, |centro de massa|), mais a energia eletrostatica do
    subconjunto -- usado por scripts/02d_diagnose_subsampling.py para
    auditar esquemas ja gerados (por fps OU electrostatic), sujeito a
    sujeito ou agregado no dataset inteiro.

    bvecs: (N,3), N>=2. Nao precisam vir normalizados.

    Retorna dict: {"min_angle_deg", "mean_nn_angle_deg", "centroid_norm",
    "electrostatic_energy"}.
      - min_angle_deg: menor angulo (antipodal-aware, ou seja tratando v e
        -v como identicos) entre qualquer par do subconjunto -- quanto
        maior, melhor (mais espalhado); compare contra o teto real do
        protocolo de origem (ver notas do projeto -- ex.: ~14,3 graus no
        esquema de 64 direcoes usado aqui), nao contra uma expectativa de
        esfera uniforme de livro-texto.
      - mean_nn_angle_deg: media do angulo ate o VIZINHO MAIS PROXIMO de
        cada direcao (nao a media de todos os pares) -- menos dominado por
        outliers que "angulo minimo" sozinho, mais sensivel a aglomeracoes
        localizadas que um unico min_angle_deg pode esconder.
      - centroid_norm: |media dos vetores unitarios do subconjunto| -- 0 =
        perfeitamente balanceado ao redor da esfera, valores maiores
        indicam vies de direcao (todas puxando pra um lado).
      - electrostatic_energy: mesma formula de _pairwise_electrostatic_energy
        (Jones et al. 1999), somada sobre o subconjunto -- quanto menor,
        melhor (mais disperso); permite comparar fps vs electrostatic no
        MESMO criterio que o metodo electrostatic otimiza diretamente
        (fps nao otimiza isso, entao e esperado que saia pior aqui).
    """
    bvecs = np.asarray(bvecs, dtype=float)
    n = bvecs.shape[0]
    if n < 2:
        raise ValueError("direction_set_quality precisa de pelo menos 2 direcoes")
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = bvecs / norms

    cos_abs = np.abs(unit @ unit.T)
    cos_abs = np.clip(cos_abs, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_abs))
    np.fill_diagonal(ang, np.inf)  # nunca contar auto-angulo (0 graus)

    min_angle_deg = float(ang.min())
    mean_nn_angle_deg = float(ang.min(axis=1).mean())
    centroid_norm = float(np.linalg.norm(unit.mean(axis=0)))
    energy = float(_subset_energy(_pairwise_electrostatic_energy(unit), range(n)))

    return {
        "min_angle_deg": min_angle_deg,
        "mean_nn_angle_deg": mean_nn_angle_deg,
        "centroid_norm": centroid_norm,
        "electrostatic_energy": energy,
    }


def _greedy_energy_init(E: np.ndarray, n_select: int, seed_idx: int) -> list[int]:
    n = E.shape[0]
    selected = [seed_idx]
    remaining = set(range(n)) - {seed_idx}
    while len(selected) < n_select:
        best_cand, best_cost = None, np.inf
        for c in remaining:
            cost = sum(E[c, s] for s in selected)
            if cost < best_cost:
                best_cost, best_cand = cost, c
        selected.append(best_cand)
        remaining.remove(best_cand)
    return selected


def _local_search_polish(E: np.ndarray, selected, n_total: int, max_iter: int = 150):
    """Busca local 2-opt (swap 1-a-1, primeira melhora): tenta trocar cada
    ponto selecionado por cada ponto nao-selecionado, aceita a primeira
    troca que reduz a energia total do subconjunto, reinicia a varredura;
    para quando uma passada completa nao encontra melhora nenhuma ou o
    teto de iteracoes e atingido.
    """
    selected = set(selected)
    not_selected = set(range(n_total)) - selected
    best_energy = _subset_energy(E, selected)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for out_i in list(selected):
            trial_base = selected - {out_i}
            for in_i in list(not_selected):
                candidate = trial_base | {in_i}
                e = _subset_energy(E, candidate)
                if e < best_energy - 1e-9:
                    selected = candidate
                    not_selected = (not_selected - {in_i}) | {out_i}
                    best_energy = e
                    improved = True
                    break
            if improved:
                break
    return sorted(selected), best_energy, it


def electrostatic_repulsion_sampling(bvecs: np.ndarray, n_select: int,
                                      n_starts: int = 20, max_local_iter: int = 150,
                                      seed: int = 42, sort: bool = True):
    """Seleciona um subconjunto de direcoes minimizando energia eletrostatica
    de Jones et al. 1999 (repulsao par-a-par, antipodal-simetrica por
    construcao) -- porte de select_directions em select_pair_rep_centroid_v2.py
    (usado no pre-processamento clinico via run_pipeline.sh --sub_qspace),
    trazido para este repo para que o pipeline de TREINO (build_subsampling_scheme)
    possa usar o mesmo criterio de selecao, evitando o domain gap geometrico
    descrito nas notas do projeto "Subamostragem angular e pipeline de
    treino da rede" (farthest_point_sampling, o metodo historico deste
    modulo, e sistematicamente pior: pra N=16 a partir de um esquema real
    de 64 direcoes, o v2 alcanca ~28,4 graus de angulo minimo contra o teto
    do proprio protocolo de aquisicao de ~14,3 graus -- quase o dobro).

    bvecs: (N, 3) vetores de uma unica shell (ja filtrados, sem b0) --
    nao precisam vir normalizados, este metodo normaliza internamente.
    n_select: quantas direcoes manter.
    n_starts: quantas sementes aleatorias tentar (greedy_init + polimento
    2-opt cada uma, mantendo a de menor energia final) -- 20 e o default
    atual (reduzido de 80 em 2026-09 apos calibracao automatica, ver
    scripts/02e_calibrate_electrostatic_nstarts.py e
    electrostatic_nstarts_calibration.csv): testado em 38 combinacoes de
    shell (500-2000) x N (6-54) com 5 sujeitos cada, ja com n_starts=10 a
    energia mediana ficava a <1% (media 0,16%) da energia obtida com
    n_starts=160 -- ou seja, a convergencia e rapida mesmo pros N mais
    altos (onde o espaco de busca e maior), sem necessidade de mais
    sementes. 20 mantem margem de seguranca de 2x sobre o menor valor
    testado (10) que ja bastava, a um custo bem menor que 80 (speedup de
    ~6-8x em N=48/54 no teste). O pequeno gap de min_angle_deg observado
    contra farthest_point_sampling em N alto (mediana <1,3 grau, ver
    subsampling_quality.csv vs. subsampling_quality_fps.csv) PERSISTE
    mesmo com n_starts alto -- e um efeito geometrico do regime
    quase-saturado (poucas direcoes de fora pra trocar quando N esta
    perto do total disponivel), nao um artefato de sub-otimizacao.
    Revalidar com scripts/02e_calibrate_electrostatic_nstarts.py se o
    dataset ou os niveis de N mudarem muito.
    max_local_iter: teto de iteracoes da busca local 2-opt por semente --
    100-150 valida (convergencia tipica em ~11 iteracoes para N=16,
    independente de tetos ate 300 testados nas notas do projeto).
    seed: semente do gerador de numeros aleatorios que escolhe as sementes
    de n_starts (reprodutibilidade).
    sort: quando True (default, mesma convencao de farthest_point_sampling),
    devolve os indices selecionados em ordem numerica crescente. Quando
    False, devolve na ordem interna do algoritmo (sem significado de
    "ordem de selecao" aqui, ao contrario de farthest_point_sampling, ja
    que a busca local pode trocar qualquer posicao a qualquer momento).

    Retorna array de indices LOCAIS (relativos a bvecs). Note: ao contrario
    de select_pair_rep_centroid_v2.py (que tambem imprime energia/angulo
    minimo/centro de massa para diagnostico), esta funcao devolve so os
    indices -- quem chamar pode recalcular esses diagnosticos a partir do
    subconjunto devolvido, se quiser (ver exemplo em
    scripts/02_subsample_directions.py).
    """
    bvecs = np.asarray(bvecs, dtype=float)
    n = bvecs.shape[0]
    if n_select > n:
        raise ValueError(f"n_select ({n_select}) nao pode exceder o numero de direcoes disponiveis ({n})")
    if n_select < 1:
        raise ValueError("n_select deve ser >= 1")
    if n_select == n:
        order = np.arange(n)
        return order if sort else order

    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = bvecs / norms

    E = _pairwise_electrostatic_energy(unit)
    rng = np.random.default_rng(seed)
    seeds_tried = rng.choice(n, size=min(n_starts, n), replace=False)

    best_selection, best_energy = None, np.inf
    for s in seeds_tried:
        init_sel = _greedy_energy_init(E, n_select, int(s))
        polished_sel, polished_energy, _n_iter = _local_search_polish(E, init_sel, n, max_local_iter)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_selection = polished_sel

    selected = np.array(best_selection)
    return np.array(sorted(selected)) if sort else selected


def subsample_shell(bvals: np.ndarray, bvecs: np.ndarray, shell_indices: np.ndarray,
                     n_select: int, seed_idx: int = 0, method: str = "fps",
                     n_starts: int = 20, max_local_iter: int = 150, seed: int = 42):
    """Aplica a selecao de direcoes dentro de uma shell especifica (indices globais).

    method: "fps" (default, farthest_point_sampling -- comportamento
    historico deste modulo, usado por todo o pipeline de treino ate
    2026-09) ou "electrostatic" (energia eletrostatica de Jones et al.
    1999 + multi-start + polimento por busca local 2-opt, ver
    electrostatic_repulsion_sampling -- porte de select_pair_rep_centroid_v2.py,
    usado no pipeline de pre-processamento clinico via --sub_qspace). Ver
    docstring de electrostatic_repulsion_sampling para o porque de preferir
    esse metodo (dispersao angular bem melhor -- ex.: ~28,4 graus de angulo
    minimo vs. o teto do proprio protocolo de ~14,3 graus, para N=16 a
    partir de um esquema real de 64 direcoes) e para o cuidado de manter
    treino e avaliacao com o MESMO metodo (evitar um domain gap geometrico
    entre os dois, alem do domain gap de SNR ja documentado no protocolo).

    n_starts/max_local_iter/seed: repassados a electrostatic_repulsion_sampling
    quando method="electrostatic" (ignorados para method="fps").

    Retorna os indices GLOBAIS (relativos ao array bvals/bvecs completo) selecionados.
    """
    local_bvecs = bvecs[shell_indices]
    if method == "fps":
        local_selected = farthest_point_sampling(local_bvecs, n_select, seed_idx=seed_idx)
    elif method == "electrostatic":
        local_selected = electrostatic_repulsion_sampling(
            local_bvecs, n_select, n_starts=n_starts, max_local_iter=max_local_iter, seed=seed)
    else:
        raise ValueError(f"method desconhecido: {method!r} (use 'fps' ou 'electrostatic')")
    return shell_indices[local_selected]


def build_subsampling_scheme(bvals: np.ndarray, bvecs: np.ndarray, n_levels: list[int],
                              tol: float = 100.0, seed_idx: int = 0, method: str = "fps",
                              n_starts: int = 20, max_local_iter: int = 150, seed: int = 42):
    """Gera, para cada shell (exceto b0) e cada nivel em n_levels, os indices
    globais de direcoes de entrada (subamostradas) e o complemento (alvo/held-out).

    method/n_starts/max_local_iter/seed: ver subsample_shell -- "fps"
    (default, comportamento historico) ou "electrostatic" (recomendado a
    partir de 2026-09, ver notas do projeto "Subamostragem angular e
    pipeline de treino da rede": usar o MESMO metodo do pre-processamento
    clinico (--sub_qspace/select_pair_rep_centroid_v2.py) tambem para gerar
    os pares de treino, para nao introduzir um domain gap geometrico entre
    o que a rede viu no treino e o que ve na validacao/producao).

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
            input_idx = subsample_shell(bvals, bvecs, idxs, n_level, seed_idx=seed_idx,
                                         method=method, n_starts=n_starts,
                                         max_local_iter=max_local_iter, seed=seed)
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


def _pairwise_bracket_geometry(candidate_bvecs: np.ndarray, target_bvecs: np.ndarray):
    """Geometria compartilhada entre find_best_bracket_batch e
    find_star_ensemble_batch -- fatorada em 2026-09-11 porque as duas
    funcoes recomputavam, de forma independente, EXATAMENTE o mesmo
    calculo O(n_pares x n_alvos) (mesmos dot(a,b), plano/normal, t_frac
    com sinal, residuo) sempre que scripts/02b_build_rrin_triplets.py
    chamava as duas no mesmo (candidate_bvecs, target_bvecs) -- ou seja,
    toda vez que --ensemble-m > 0, a geometria pairwise de um combo
    (shell, n_level, sujeito) era paga em dobro. Agora
    02b_build_rrin_triplets.py so chama find_star_ensemble_batch quando o
    ensemble esta ligado (a posicao 0 do feixe e' sempre, por construcao,
    o mesmo par que find_best_bracket_batch devolveria sozinha -- ver
    docstring de find_star_ensemble_batch), entao o calculo pairwise so e'
    feito uma vez por combo em qualquer dos dois casos. Numericamente
    IDENTICA ao que cada funcao computava inline antes desta refatoracao
    -- so extraida, nenhuma formula mudou.

    Retorna dict com, para n_pairs = M*(M-1)/2 pares (i,j), i<j, de
    candidate_bvecs (M,3), e K alvos de target_bvecs (K,3):
      "iu", "ju": (n_pairs,) indices LOCAIS de cada par em candidate_bvecs.
      "gap_deg_pairs": (n_pairs,) angulo entre os dois candidatos do par,
          em graus -- nao depende do alvo.
      "n_hat": (n_pairs,3) normal do plano de cada par.
      "t_frac": (n_pairs,K) posicao com sinal do alvo no arco a->b.
      "residual_deg": (n_pairs,K) desvio de colinearidade, em graus.
      "between": (n_pairs,K) bool, 0<=t_frac<=1.
      "centrality": (n_pairs,K) abs(t_frac-0.5) -- 0 = alvo exatamente no
          meio do par (interpolacao mais "genuina"), 0.5 = alvo coincide
          com uma das pontas do par (quase-extrapolacao) -- usado so
          quando prefer_central_t_frac=True nas funcoes que chamam este
          helper (ver find_best_bracket_batch/find_star_ensemble_batch).
      "degenerate": (n_pairs,) bool, par quase-paralelo (sem plano bem
          definido -- residuo tratado como maximo, ver spherical_triplet_residual).
    """
    candidate_bvecs = np.asarray(candidate_bvecs, dtype=float)
    target_bvecs = np.atleast_2d(np.asarray(target_bvecs, dtype=float))
    m = candidate_bvecs.shape[0]
    if m < 2:
        raise ValueError("geometria de pares precisa de pelo menos 2 direcoes candidatas")

    u_norm = np.linalg.norm(candidate_bvecs, axis=1, keepdims=True)
    u_norm[u_norm == 0] = 1.0
    U = candidate_bvecs / u_norm

    t_norm = np.linalg.norm(target_bvecs, axis=1, keepdims=True)
    t_norm[t_norm == 0] = 1.0
    T = target_bvecs / t_norm

    iu, ju = np.triu_indices(m, k=1)  # mesma ordem de enumeracao de find_best_bracket

    a_pairs = U[iu]  # (n_pairs, 3) -- papel de "a" = candidato de indice menor
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
    # forma a base ortonormal do plano do grande circulo (mesma construcao
    # de spherical_triplet_residual).
    e2 = np.cross(n_hat, a_pairs)  # (n_pairs, 3)
    e2_norm = np.linalg.norm(e2, axis=1)
    e2_ok = e2_norm > 1e-12
    e2_safe = np.zeros_like(e2)
    e2_safe[e2_ok] = e2[e2_ok] / e2_norm[e2_ok, None]
    e2 = e2_safe

    dot_it = U @ T.T  # (M, K) -- dot(candidato_i_bruto, alvo_bruto)
    dot_at = dot_it[iu, :]  # (n_pairs, K) -- dot(a_pairs, alvo_bruto)
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
    centrality = np.abs(t_frac - 0.5)

    return {
        "iu": iu, "ju": ju,
        "gap_deg_pairs": gap_deg_pairs,
        "n_hat": n_hat,
        "t_frac": t_frac,
        "residual_deg": residual_deg,
        "between": between,
        "centrality": centrality,
        "degenerate": degenerate,
    }


def _fps_avoid_shared_anchor(unit_vecs: np.ndarray, anchors_i: np.ndarray, anchors_j: np.ndarray,
                              n_select: int, seed_idx: int = 0) -> np.ndarray:
    """Variante de farthest_point_sampling usada por find_star_ensemble_batch
    quando avoid_shared_anchor=True (ver docstring la): mesma selecao
    gulosa por distancia maxima (aqui sobre as NORMAIS dos pares do pool,
    nao bvecs brutos), mas em cada passo PREFERE, entre os candidatos, um
    cuja "ancora" (indice do candidato i OU j do proprio par) ainda nao
    apareca em nenhum par ja escolhido do feixe -- evita que duas posicoes
    do mesmo feixe reusem a mesma direcao de entrada como uma das pontas,
    o que daria a rede duas "evidencias" menos independentes entre si do
    que a diversidade de plano (FPS por normal) sozinha sugere. Cai de
    volta no candidato de maior distancia SEM essa restricao quando
    nenhum candidato remanescente tem ancora livre (mesmo padrao de
    fallback ja usado por max_gap_deg -- nunca reduz quantos pares o feixe
    tem, so influencia QUAIS sao escolhidos quando ha opcao de sobra).

    unit_vecs: (P,3) normais dos pares do pool (ja unitarias).
    anchors_i, anchors_j: (P,) indices LOCAIS (em candidate_bvecs) dos dois
    candidatos de cada par do pool, na MESMA ordem/indexacao de unit_vecs.
    """
    unit_vecs = np.asarray(unit_vecs, dtype=float)
    n = unit_vecs.shape[0]
    if n_select >= n:
        rest = [i for i in range(n) if i != seed_idx]
        return np.array([seed_idx] + rest)
    if n_select < 1:
        raise ValueError("n_select deve ser >= 1")

    selected = [seed_idx]
    used_anchors = {int(anchors_i[seed_idx]), int(anchors_j[seed_idx])}
    cos_to_seed = np.abs(unit_vecs @ unit_vecs[seed_idx])
    min_dist = 1.0 - cos_to_seed

    while len(selected) < n_select:
        min_dist[selected[-1]] = -np.inf
        order = np.argsort(-min_dist)  # do mais distante ao menos distante
        picked = None
        for idx in order:
            if min_dist[idx] == -np.inf:
                break  # resto ja e' tudo selecionado antes (distancia -inf)
            if int(anchors_i[idx]) not in used_anchors and int(anchors_j[idx]) not in used_anchors:
                picked = int(idx)
                break
        if picked is None:
            # nenhum candidato remanescente tem ancora livre -- cai para o
            # de maior distancia mesmo (comportamento sem esta restricao).
            picked = int(np.argmax(min_dist))
        selected.append(picked)
        used_anchors.add(int(anchors_i[picked]))
        used_anchors.add(int(anchors_j[picked]))
        cos_new = np.abs(unit_vecs @ unit_vecs[picked])
        dist_new = 1.0 - cos_new
        min_dist = np.minimum(min_dist, dist_new)

    return np.array(selected)


def find_best_bracket_batch(candidate_bvecs: np.ndarray, target_bvecs: np.ndarray,
                             max_residual_deg: float | None = None,
                             require_between: bool = True,
                             prefer_central_t_frac: bool = False):
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

    prefer_central_t_frac (default False, flag ADITIVA -- ver revisao de
    codigo 2026-09-11): quando True, usa abs(t_frac-0.5) como criterio
    TERCIARIO de desempate (depois de gap_deg minimo e residual_deg
    minimo, os dois criterios de sempre) entre pares aceitaveis com o
    MESMO gap_deg e residual_deg -- prefere o par em que o alvo cai mais
    perto do meio do arco (interpolacao "genuina") a um em que o alvo
    quase coincide com uma das pontas do par (t_frac perto de 0 ou 1,
    quase-extrapolacao mesmo estando dentro de [0,1]). Na pratica, como
    gap_deg/residual_deg raramente empatam exatamente entre pares
    distintos, este criterio so decide poucos casos -- pense nele como
    uma preferencia leve, nao uma mudanca de regime na selecao. Default
    False preserva o comportamento de sempre bit-a-bit.

    **CORRIGIDO em 2026-08-27 junto com spherical_triplet_residual (ver
    docstring la para o bug/contraexemplo completo):** o t_frac aqui agora
    tambem usa o angulo COM SINAL (base ortonormal (a_par, e2) no plano do
    par), nao mais `arccos(dot)/ang_ab` sem sinal -- reverificado por
    equivalencia numerica contra `find_best_bracket` chamado par-a-par em
    dados aleatorios (200 conjuntos x 5 alvos, 0 divergencias) mais o
    contraexemplo especifico que expos o bug original.
    """
    geo = _pairwise_bracket_geometry(candidate_bvecs, target_bvecs)
    iu, ju = geo["iu"], geo["ju"]
    gap_deg_pairs = geo["gap_deg_pairs"]
    t_frac = geo["t_frac"]
    residual_deg = geo["residual_deg"]
    between = geo["between"]
    centrality = geo["centrality"]
    k_targets = t_frac.shape[1]

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
                # entre o pool, menor gap_deg; empate quebrado pelo menor
                # residuo e, se prefer_central_t_frac, por ultimo pela
                # centralidade do alvo no arco (ver docstring do parametro).
                if prefer_central_t_frac:
                    order = np.lexsort((centrality[pool_idx, k], residual_deg[pool_idx, k],
                                         gap_deg_pairs[pool_idx]))
                else:
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
                              require_between: bool = True,
                              max_gap_deg: float | None = None,
                              prefer_central_t_frac: bool = False,
                              avoid_shared_anchor: bool = False):
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

    max_gap_deg (opcional, default None = comportamento acima, inalterado):
    a diversidade geometrica pura (FPS por normais, passo 2 acima) nao
    prioriza pares com `gap_deg` pequeno -- ela pode escolher, entre os
    `m-1` pares alem da semente, candidatos bem dispersos mas todos com
    separacao angular grande entre `a`/`b`, regime onde a premissa de fluxo
    optico (que RRIN3DStar usa para "esticar" um par ate o alvo, ver
    model/rrin3d_star.py) e mais fragil (analogia OLAT, ver protocolo/
    addendum secao 10/12). Achado que motivou este parametro: diagnostico
    de pesos de fusao por voxel (addendum secao 20.6) mostrou que, quando
    NENHUM dos `m` candidatos de um alvo tem `gap_deg` pequeno, a fusao
    aprendida (PairWeightHead3D) nao tem um "vencedor" obvio pra confiar e
    acaba fazendo uma media quase-uniforme entre varios candidatos
    mediocres -- borrando estrutura angular fina (mesmo mecanismo do
    `naive_ensemble_blend`, ver secao 20.4, so que dentro da fusao
    aprendida). Quando `max_gap_deg` e dado, a escolha dos `m-1` pares
    ALEM da semente (que continua sendo sempre o par de menor `gap_deg` do
    pool, goste ou nao do teto) fica restrita, quando possivel, ao
    subconjunto do pool com `gap_deg<=max_gap_deg` -- ainda maximizando
    dispersao de normais DENTRO desse subconjunto (nao abre mao da
    diversidade, so a busca dentro de um universo mais "amigavel" ao
    warp). Se esse subconjunto nao tiver membros suficientes pra preencher
    o feixe (menos de `m` no total, incluindo a semente), o teto e
    ignorado PARA ESSE ALVO ESPECIFICO e cai no comportamento antigo (FPS
    sobre o pool inteiro) -- o teto nunca reduz quantos pares reais o
    feixe tem, so influencia QUAIS sao escolhidos quando ha opcao de
    sobra. `None` (default) preserva o comportamento de sempre bit-a-bit,
    incluindo para m=1 (a checagem de equivalencia com
    find_best_bracket_batch nao usa este parametro).

    prefer_central_t_frac (default False -- ver mesmo parametro em
    find_best_bracket_batch): usa abs(t_frac-0.5) como desempate
    TERCIARIO (depois de gap_deg e residual_deg) tanto para escolher a
    SEMENTE do feixe quanto, quando max_residual_deg=None, para a ordem
    "so por residuo" usada nesse caso -- mesma semantica exata do
    parametro homonimo em find_best_bracket_batch (inclusive a garantia
    de que, com m=1, o resultado continua identico a
    find_best_bracket_batch chamada com o mesmo valor deste parametro).

    avoid_shared_anchor (default False, flag ADITIVA -- ver revisao de
    codigo 2026-09-11): quando True, a escolha dos m-1 pares ALEM da
    semente (passo 2 acima) usa _fps_avoid_shared_anchor em vez de
    farthest_point_sampling puro -- entre candidatos com diversidade de
    plano equivalente, prefere pares cujas duas pontas (indices i e j em
    candidate_bvecs) ainda NAO apareceram em nenhum outro par ja
    escolhido do MESMO feixe, para que os `m` pares carreguem evidencia
    geometrica mais independente entre si (dois pares que compartilham
    uma ponta, mesmo com normais bem diferentes, reusam metade da mesma
    informacao de entrada). Nunca reduz quantos pares reais o feixe tem
    -- se nenhum candidato remanescente tiver ancora livre, cai de volta
    no candidato de maior distancia sem essa restricao (mesmo padrao de
    fallback de max_gap_deg). Default False preserva o comportamento de
    sempre bit-a-bit, incluindo a equivalencia com find_best_bracket_batch
    em m=1 (fallback do feixe com m=1 nunca aciona o passo de FPS).

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
    n_cand = candidate_bvecs.shape[0]
    if n_cand < 2:
        raise ValueError("find_star_ensemble_batch precisa de pelo menos 2 direcoes candidatas")
    if m < 1:
        raise ValueError("m deve ser >= 1")

    # ---- geometria pairwise: fatorada em _pairwise_bracket_geometry (2026-09-11)
    # -- era recomputada aqui do zero, identica a de find_best_bracket_batch,
    # toda vez que o script de triplets chamava as duas funcoes no mesmo
    # (candidate_bvecs, target_bvecs); ver docstring do helper. ----
    geo = _pairwise_bracket_geometry(candidate_bvecs, target_bvecs)
    iu, ju = geo["iu"], geo["ju"]
    n_pairs = iu.shape[0]
    n_hat = geo["n_hat"]
    gap_deg_pairs = geo["gap_deg_pairs"]      # (n_pairs,) -- nao depende do alvo
    t_frac = geo["t_frac"]                    # (n_pairs, K)
    residual_deg = geo["residual_deg"]        # (n_pairs, K)
    between = geo["between"]
    centrality = geo["centrality"]
    k_targets = t_frac.shape[1]

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
                # ordena o pool por gap_deg crescente (empate: menor residuo,
                # e se prefer_central_t_frac, por ultimo a centralidade) --
                # 1o elemento = par que find_best_bracket_batch teria escolhido
                # sozinho (mesmo criterio, so que aqui como SEMENTE do feixe).
                if prefer_central_t_frac:
                    order = np.lexsort((centrality[pool_idx, k], residual_deg[pool_idx, k],
                                         gap_deg_pairs[pool_idx]))
                else:
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
            if prefer_central_t_frac:
                order = np.lexsort((centrality[pool_idx, k], residual_deg[pool_idx, k]))
            else:
                order = np.argsort(residual_deg[pool_idx, k])
            pool_sorted = pool_idx[order]

        if pool_sorted.size <= m:
            chosen = pool_sorted
        elif max_gap_deg is not None:
            # restringe os m-1 pares ALEM da semente (pool_sorted[0], que
            # continua sendo sempre o de menor gap_deg do pool) a um
            # subconjunto com gap_deg<=max_gap_deg, quando esse subconjunto
            # tiver membros suficientes pra preencher o feixe -- ver
            # docstring ("max_gap_deg") para a motivacao completa.
            preferred_mask = gap_deg_pairs[pool_sorted] <= max_gap_deg
            preferred_mask[0] = True  # semente sempre incluida, mesmo que o
                                        # proprio gap dela exceda o teto (e' o
                                        # melhor disponivel de qualquer jeito)
            preferred_idx = np.nonzero(preferred_mask)[0]
            if preferred_idx.size >= m:
                # indice 0 e' sempre o menor valor possivel em preferred_idx
                # (pool_sorted[0] sempre marcado True acima), entao fica na
                # posicao 0 do array ordenado -- semente da FPS sem precisar
                # localizar a posicao.
                sub_pool = pool_sorted[preferred_idx]
                normals_sub = n_hat[sub_pool]
                if avoid_shared_anchor:
                    fps_local = _fps_avoid_shared_anchor(normals_sub, iu[sub_pool], ju[sub_pool],
                                                           m, seed_idx=0)
                else:
                    fps_local = farthest_point_sampling(normals_sub, m, seed_idx=0, sort=False)
                chosen = sub_pool[fps_local]
            else:
                # nao ha candidatos suficientes dentro do teto de gap pra
                # preencher o feixe com diversidade -- cai no comportamento
                # antigo (FPS sobre o pool inteiro) so PARA ESTE ALVO.
                normals_pool = n_hat[pool_sorted]
                if avoid_shared_anchor:
                    fps_local = _fps_avoid_shared_anchor(normals_pool, iu[pool_sorted], ju[pool_sorted],
                                                           m, seed_idx=0)
                else:
                    fps_local = farthest_point_sampling(normals_pool, m, seed_idx=0, sort=False)
                chosen = pool_sorted[fps_local]
        else:
            normals_pool = n_hat[pool_sorted]  # (P,3)
            if avoid_shared_anchor:
                fps_local = _fps_avoid_shared_anchor(normals_pool, iu[pool_sorted], ju[pool_sorted],
                                                       m, seed_idx=0)
            else:
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