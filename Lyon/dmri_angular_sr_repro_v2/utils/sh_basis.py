"""
Base de harmonicos esfericos reais e simetricos (SH par) para dMRI, usada no
baseline de interpolacao (sem rede neural).

Implementacao propria com numpy/scipy (nao depende de dipy), para poder ser
testada isoladamente neste ambiente. A convencao de base (real, simetrica,
ordem par l=0,2,4,...) e equivalente a usada por dipy.reconst.shm com
`descoteaux07`/`tournier07`; ao integrar com dipy no cluster, prefira
`dipy.reconst.shm.sph_harm_basis` para consistencia com o resto do
pipeline (CSD, etc.) -- este modulo serve como fallback e para deixar a
matematica explicita no protocolo.
"""
from __future__ import annotations

import numpy as np
from scipy.special import lpmv, factorial


def max_order_for_n_directions(n_dirs: int) -> int:
    """Maior ordem par l_max tal que o numero de coeficientes SH,
    R = (l_max+1)(l_max+2)/2, seja <= n_dirs (sistema nao subdeterminado).
    """
    l_max = 0
    while True:
        next_l = l_max + 2
        n_coef = (next_l + 1) * (next_l + 2) // 2
        if n_coef > n_dirs:
            break
        l_max = next_l
    return l_max


def real_sh_matrix(theta: np.ndarray, phi: np.ndarray, l_max: int) -> np.ndarray:
    """Constroi a matriz B (n_dirs x n_coef) de harmonicos esfericos reais,
    simetricos (apenas l par), na convencao usada em dMRI (Descoteaux 2007).

    theta: colatitude (0..pi), phi: azimute (0..2pi), arrays (n_dirs,).
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    n_dirs = theta.shape[0]

    cols = []
    for l in range(0, l_max + 1, 2):
        for m in range(-l, l + 1):
            cols.append(_real_sh(l, m, theta, phi))
    B = np.stack(cols, axis=1)
    assert B.shape == (n_dirs, (l_max + 1) * (l_max + 2) // 2)
    return B


def _real_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Harmonico esferico real de grau l, ordem m (convencao Descoteaux 2007)."""
    abs_m = abs(m)
    norm = np.sqrt(
        (2 * l + 1) / (4 * np.pi) * factorial(l - abs_m) / factorial(l + abs_m)
    )
    plm = lpmv(abs_m, l, np.cos(theta))
    if m > 0:
        return np.sqrt(2) * norm * plm * np.cos(abs_m * phi)
    elif m < 0:
        return np.sqrt(2) * norm * plm * np.sin(abs_m * phi)
    else:
        return norm * plm


def cart2sphere(bvecs: np.ndarray):
    """Converte vetores cartesianos unitarios (N,3) em (theta, phi)."""
    x, y, z = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]
    r = np.linalg.norm(bvecs, axis=1)
    r_safe = np.where(r == 0, 1.0, r)
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x) % (2 * np.pi)
    return theta, phi


def laplace_beltrami_diag(l_max: int) -> np.ndarray:
    """Diagonal do regularizador de Laplace-Beltrami: l^2 (l+1)^2 por coeficiente,
    na mesma ordem de colunas usada em real_sh_matrix.
    """
    diag = []
    for l in range(0, l_max + 1, 2):
        for _m in range(-l, l + 1):
            diag.append((l * (l + 1)) ** 2)
    return np.array(diag, dtype=np.float64)


def fit_sh(signal: np.ndarray, bvecs: np.ndarray, l_max: int | None = None,
           lambda_reg: float = 0.006):
    """Ajusta coeficientes SH (regressao regularizada, Descoteaux 2007) a um
    sinal de uma unica shell.

    signal: (..., n_dirs) -- sinal DWI normalizado (dividido pelo b0) na shell.
    bvecs: (n_dirs, 3) direcoes dessa shell (ja filtradas, sem b0).
    Retorna (coef, B, l_max) onde coef tem shape (..., n_coef).
    """
    n_dirs = bvecs.shape[0]
    if l_max is None:
        l_max = max_order_for_n_directions(n_dirs)
    theta, phi = cart2sphere(bvecs)
    B = real_sh_matrix(theta, phi, l_max)  # (n_dirs, n_coef)
    L = laplace_beltrami_diag(l_max)

    BtB = B.T @ B
    reg = lambda_reg * np.diag(L)
    inv = np.linalg.pinv(BtB + reg)
    fit_matrix = inv @ B.T  # (n_coef, n_dirs)

    orig_shape = signal.shape[:-1]
    flat = signal.reshape(-1, n_dirs)
    coef = (fit_matrix @ flat.T).T  # (n_voxels, n_coef)
    coef = coef.reshape(*orig_shape, -1)
    return coef, B, l_max


def predict_sh(coef: np.ndarray, bvecs_target: np.ndarray, l_max: int) -> np.ndarray:
    """Prediz sinal em novas direcoes a partir dos coeficientes SH ajustados."""
    theta, phi = cart2sphere(bvecs_target)
    B_target = real_sh_matrix(theta, phi, l_max)  # (n_target, n_coef)
    orig_shape = coef.shape[:-1]
    flat = coef.reshape(-1, coef.shape[-1])
    pred = flat @ B_target.T  # (n_voxels, n_target)
    return pred.reshape(*orig_shape, -1)
