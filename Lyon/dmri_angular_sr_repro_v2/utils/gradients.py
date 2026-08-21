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