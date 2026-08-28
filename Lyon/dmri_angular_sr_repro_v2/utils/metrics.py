"""
Metricas de avaliacao de reconstrucao (nivel de sinal).

PSNR, SSIM, NMSE e ACC (angular correlation coefficient) entre volumes/
coeficientes reconstruidos e o ground truth. Implementadas com numpy/scipy
puros onde possivel; SSIM usa skimage se disponivel, com fallback proprio.
"""
from __future__ import annotations

import numpy as np


def nmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Erro quadratico medio normalizado pela energia do target."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is not None:
        pred = pred[mask.astype(bool)]
        target = target[mask.astype(bool)]
    denom = np.sum(target ** 2)
    if denom <= 0:
        return float("nan")
    return float(np.sum((pred - target) ** 2) / denom)


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is not None:
        pred = pred[mask.astype(bool)]
        target = target[mask.astype(bool)]
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None,
          data_range: float | None = None) -> float:
    """PSNR em dB. data_range default = max(target) dentro da mascara."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is not None:
        m = mask.astype(bool)
        pred_v, target_v = pred[m], target[m]
    else:
        pred_v, target_v = pred, target
    mse = np.mean((pred_v - target_v) ** 2)
    if mse <= 0:
        return float("inf")
    if data_range is None:
        data_range = float(np.max(target_v)) if target_v.size else 1.0
        if data_range <= 0:
            data_range = 1.0
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim3d(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None,
           data_range: float | None = None) -> float:
    """SSIM para volume 3D. Usa skimage.metrics.structural_similarity se
    disponivel (recomendado); caso contrario cai num fallback global simples
    (nao equivalente, apenas para nao travar o pipeline caso skimage falte).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if data_range is None:
        ref = target[mask.astype(bool)] if mask is not None else target
        data_range = float(np.max(ref) - np.min(ref)) if ref.size else 1.0
        if data_range <= 0:
            data_range = 1.0
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        # aplica mascara zerando fora da regiao de interesse (aproximacao;
        # para SSIM por regiao mais correto, seria preciso crop na bbox)
        p = pred.copy()
        t = target.copy()
        if mask is not None:
            m = ~mask.astype(bool)
            p[m] = 0
            t[m] = 0
        val = sk_ssim(t, p, data_range=data_range)
        return float(val)
    except ImportError:
        # fallback: SSIM global (formula classica, sem janela deslizante)
        m = mask.astype(bool) if mask is not None else np.ones_like(pred, dtype=bool)
        p, t = pred[m], target[m]
        mu_p, mu_t = p.mean(), t.mean()
        var_p, var_t = p.var(), t.var()
        cov = np.mean((p - mu_p) * (t - mu_t))
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        num = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
        den = (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2)
        return float(num / den)


def angular_correlation_coefficient(sh_pred: np.ndarray, sh_target: np.ndarray) -> np.ndarray:
    """ACC entre coeficientes SH preditos e ground truth, por voxel.

    sh_pred, sh_target: (..., n_coef) coeficientes de harmonicos esfericos reais
    (mesma base/ordem). Retorna array (...) com o ACC por voxel, formula de
    Anderson (2005): ACC = sum(f_l * f_l') / sqrt(sum(f_l^2) * sum(f_l'^2)),
    aqui aplicada diretamente sobre o vetor de coeficientes (equivalente
    quando ambos usam a mesma base ortonormal).
    """
    sh_pred = np.asarray(sh_pred, dtype=np.float64)
    sh_target = np.asarray(sh_target, dtype=np.float64)
    num = np.sum(sh_pred * sh_target, axis=-1)
    den = np.sqrt(np.sum(sh_pred ** 2, axis=-1) * np.sum(sh_target ** 2, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(den > 0, num / den, 0.0)
    return acc


def signal_bias(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Erro medio COM SINAL (pred-target), diferente de rmse (que usa erro
    ao quadrado e por isso mistura vies sistematico com ruido aleatorio de
    verdade). Vies proximo de 0 com rmse/resid_std alto = ruido genuino em
    torno do valor certo; vies grande em modulo (nao cancela ao somar) =
    erro sistematico (ex.: reconstrucao sempre "puxando" pra media,
    assinatura classica de fluxo/blend quando a correspondencia real nao
    existe). Ver tambem residual_std."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is not None:
        pred = pred[mask.astype(bool)]
        target = target[mask.astype(bool)]
    return float(np.mean(pred - target))


def residual_std(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Desvio-padrao do residuo (pred-target) -- a parte do erro que SOBRA
    depois de descontar o vies medio (signal_bias). Usar os dois juntos:
    rmse^2 == bias^2 + resid_std^2 (identidade exata quando calculados
    sobre a MESMA populacao de valores) -- decompoe o rmse total em quanto
    e vies sistematico vs. quanto e variancia/ruido de verdade."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is not None:
        pred = pred[mask.astype(bool)]
        target = target[mask.astype(bool)]
    return float(np.std(pred - target))


def r2_score_per_voxel(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """R^2 (fracao da variancia explicada) por voxel, ao longo do ULTIMO
    EIXO do array (as direcoes held-out desse voxel). pred/target devem ter
    shape (n_voxels, n_targets) -- por exemplo `recon_sel[mask]`/
    `gt_sel[mask]` em scripts/06_evaluate_reconstruction.py.

    Mede algo DIFERENTE de nmse: nmse normaliza o erro pela ENERGIA total
    do sinal alvo (sum(target^2)), entao um voxel de sinal quase constante
    entre direcoes ja tem nmse baixo so por ter pouca variacao pra errar --
    nao prova que o metodo capturou nada. R^2 normaliza pela VARIANCIA do
    alvo entre as direcoes held-out desse voxel especifico: R^2=1 e
    reconstrucao perfeita; R^2=0 empata com o "modelo nulo" mais simples
    possivel (prever, pra toda direcao held-out do voxel, a MEDIA do
    proprio alvo real nessas direcoes); R^2<0 e PIOR que esse modelo nulo
    -- ou seja, o metodo ativamente atrapalha em vez de ajudar. E o teste
    mais direto de "recuperou a variacao angular real do sinal" vs. "so
    ficou perto de um valor plausivel/medio" (o sintoma classico de
    hallucination/regressao-a-media quando a correspondencia entre
    direcoes de gradiente e' fraca -- ver discussao de fluxo optico no
    addendum do projeto).

    Voxels com variancia real ~0 entre as direcoes held-out (SS_tot<=0,
    sinal praticamente constante ali) tem R^2 mal-definido -- retornam NaN
    em vez de um numero arbitrario; use np.nanmean/np.nanmedian ao agregar."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mean_target = np.mean(target, axis=-1, keepdims=True)
    ss_res = np.sum((pred - target) ** 2, axis=-1)
    ss_tot = np.sum((target - mean_target) ** 2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return r2


def paired_wilcoxon(a: np.ndarray, b: np.ndarray):
    """Teste de Wilcoxon pareado (a vs b), retorna (estatistica, p-valor).
    Requer scipy. Usar para comparar baseline vs RCAE por sujeito/ROI.
    """
    from scipy.stats import wilcoxon
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    stat, p = wilcoxon(a, b)
    return float(stat), float(p)