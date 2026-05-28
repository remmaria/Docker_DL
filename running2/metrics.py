import torch
import numpy as np
from scipy.stats import pearsonr


def calculate_res_sign_consistency(pred_res, target, media_vizinhos, mask):
    """
    Calcula a porcentagem de voxels onde o sinal do resíduo predito
    coincide com o sinal do resíduo real.
    """
    # Resíduo real: o que a física (média) não explicou
    real_res = target - media_vizinhos
    
    # Máscara para focar apenas no cérebro
    mask_bool = mask.bool()
    
    if not mask_bool.any():
        return 0.0

    p_res = pred_res[mask_bool]
    r_res = real_res[mask_bool]

    # Verifica se os sinais são iguais (ambos pos ou ambos neg)
    # (p * r > 0) é True se tiverem o mesmo sinal
    same_sign = (p_res * r_res > 0).float()
    
    return float(same_sign.mean())

def calculate_rmse_corr(pred, target, mask):
    """
    Calcula RMSE e Correlação de Pearson apenas dentro da máscara cerebral.

    Parâmetros
    ----------
    pred   : tensor [B, 1, H, W, D] — predição do modelo.
    target : tensor [B, 1, H, W, D] — ground truth.
    mask   : tensor [B, 1, H, W, D] — máscara binária (1=dentro do cérebro).

    Retorna
    -------
    rmse : float
    corr : float  (0.0 se não houver variância suficiente)
    """
    mask_bool = mask.bool()
    p = pred[mask_bool].detach().cpu().float().numpy()
    t = target[mask_bool].detach().cpu().float().numpy()

    if len(p) == 0:
        return 0.0, 0.0

    rmse = float(np.sqrt(np.mean((p - t) ** 2)))

    if len(p) > 1 and np.std(p) > 1e-6 and np.std(t) > 1e-6:
        corr, _ = pearsonr(p, t)
        corr = float(corr)
    else:
        corr = 0.0

    return rmse, corr


def calculate_region_metrics(pred, target, residual, mask):
    """
    Calcula métricas separadas para dentro e fora da máscara cerebral,
    além de estatísticas do resíduo por região.

    Isso permite detectar se a rede está aprendendo o sinal DWI real
    (rmse_brain baixo, pearson_brain alto) ou apenas memorizando zeros
    fora do cérebro (rmse_background artificialmente baixo, resíduo
    maior fora do que dentro).

    Parâmetros
    ----------
    pred     : tensor [B, 1, H, W, D] — predição final do modelo
    target   : tensor [B, 1, H, W, D] — ground truth
    residual : tensor [B, 1, H, W, D] — resíduo predito pelo decoder
    mask     : tensor [B, 1, H, W, D] — máscara binária (1=cérebro)

    Retorna
    -------
    dict com as seguintes chaves:
        rmse_brain          : RMSE dentro do cérebro
        rmse_background     : RMSE fora do cérebro
        pearson_brain       : Pearson dentro do cérebro
        pearson_background  : Pearson fora do cérebro
        res_mean_brain      : Magnitude média do resíduo dentro do cérebro
        res_mean_background : Magnitude média do resíduo fora do cérebro
        res_ratio           : res_brain / res_background
                              > 1 → rede focando onde importa        ✅
                              < 1 → rede corrigindo o fundo mais     ❌
    """
    pred     = pred.detach().cpu().float()
    target   = target.detach().cpu().float()
    residual = residual.detach().cpu().float()
    mask     = mask.cpu()

    brain_mask = mask.bool()
    bg_mask    = ~brain_mask

    def _rmse_corr(p_arr, t_arr):
        if len(p_arr) == 0:
            return 0.0, 0.0
        rmse = float(np.sqrt(np.mean((p_arr - t_arr) ** 2)))
        if len(p_arr) > 1 and np.std(p_arr) > 1e-6 and np.std(t_arr) > 1e-6:
            corr, _ = pearsonr(p_arr, t_arr)
        else:
            corr = 0.0
        return rmse, float(corr)

    p_brain = pred[brain_mask].numpy()
    t_brain = target[brain_mask].numpy()
    rmse_brain, pearson_brain = _rmse_corr(p_brain, t_brain)

    p_bg = pred[bg_mask].numpy()
    t_bg = target[bg_mask].numpy()
    rmse_bg, pearson_bg = _rmse_corr(p_bg, t_bg)

    res_abs   = residual.abs()
    res_brain = float(res_abs[brain_mask].mean()) if brain_mask.any() else 0.0
    res_bg    = float(res_abs[bg_mask].mean())    if bg_mask.any()    else 0.0

    # > 1 → resíduo maior dentro do cérebro (correto)
    # < 1 → rede trabalhando mais fora do cérebro (sinal de problema)
    res_ratio = res_brain / (res_bg + 1e-8)

    return {
        "rmse_brain":          rmse_brain,
        "rmse_background":     rmse_bg,
        "pearson_brain":       pearson_brain,
        "pearson_background":  pearson_bg,
        "res_mean_brain":      res_brain,
        "res_mean_background": res_bg,
        "res_ratio":           res_ratio,
    }