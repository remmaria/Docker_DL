import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from monai.losses import SSIMLoss


class PhysicalCompoundLoss(nn.Module):
    """
    Loss composta para predição de DWI no espaço-q.

    Componentes
    -----------
    L1 ponderado    : erro de intensidade por voxel, com peso extra na WM.
    SSIM            : perda estrutural (MONAI, 3D).
    Gradiente       : penaliza erros de borda/textura.
    Residual        : penaliza o resíduo predito longe do resíduo real.

    CORRIGIDO: os pesos de grad e res agora podem ser escalonados
    externamente (via set_phase_weights) para um warmup gradual.
    Sem isso, res_weight=20 no início do treino fazia a loss de resíduo
    dominar enquanto o modelo ainda produzia ruído, impedindo o
    aprendizado da média ponderada.
    """

    def __init__(
        self,
        patch_size,
        l1_weight=1.0,
        ssim_weight=1.0,
        grad_weight=10.0,
        res_weight=20.0,
        wm_multiplier=2.0,
    ):
        super(PhysicalCompoundLoss, self).__init__()

        win_size = min(5, patch_size)

        if win_size % 2 == 0:
            win_size -= 1

        win_size = max(win_size, 3)

        self.ssim = SSIMLoss(
            spatial_dims=3,
            data_range=5.0,
            win_size=win_size,
        )

        # Pesos base (fixos após inicialização)
        self.l1_w_base   = float(l1_weight)
        self.ssim_w_base = float(ssim_weight)
        self.grad_w_base = float(grad_weight)
        self.res_w_base  = float(res_weight)
        self.wm_multiplier = float(wm_multiplier)

        # Pesos ativos (podem ser escalonados via set_phase_weights)
        self.l1_w   = self.l1_w_base
        self.ssim_w = self.ssim_w_base
        self.grad_w = self.grad_w_base
        self.res_w  = self.res_w_base

    def _compute_ssim(self, pred, target, mask):
        mask_bin = (mask > 0.5).float()

        fg_count = mask_bin.sum().clamp(min=1.0)
        fg_mean = (pred * mask_bin).sum() / fg_count

        pred_fg = pred * mask_bin + fg_mean.detach() * (1.0 - mask_bin)
        target_fg = target * mask_bin + fg_mean.detach() * (1.0 - mask_bin)

        # Clamp para evitar valores extremos que quebram a janela SSIM
        pred_fg = pred_fg.clamp(0.0, 5.0)
        target_fg = target_fg.clamp(0.0, 5.0)

        ssim_val = self.ssim(pred_fg, target_fg)

        # Protege NaN que vem de janelas sem variância
        ssim_val = torch.nan_to_num(ssim_val, nan=1.0)  # SSIM loss=1 → pior caso

        return ssim_val

    def set_phase_weights(self, fase: int):
        """
        Ajusta os pesos de cada componente de acordo com a fase de treino.

        Fase 1 (geometria): res e grad com peso baixo para o modelo
                            primeiro aprender a média ponderada.
        Fase 2 (multi-shell): escala intermediária.
        Fase 3 (refinamento): pesos completos.

        Chame este método no loop de treino após set_fase() no dataset.
        """
        if fase == 1:
            self.grad_w = self.grad_w_base * 0.1
            self.res_w  = self.res_w_base  * 0.1
        elif fase == 2:
            self.grad_w = self.grad_w_base * 0.5
            self.res_w  = self.res_w_base  * 0.5
        else:
            self.grad_w = self.grad_w_base
            self.res_w  = self.res_w_base

    def forward(self, pred_final, target, mask, maskWM, pred=None, media_vizinhos=None):
        """
        Parâmetros
        ----------
        pred_final    : [B, 1, H, W, D] — predição final (média + resíduo).
        target        : [B, 1, H, W, D] — ground truth.
        mask          : [B, 1, H, W, D] — máscara do cérebro.
        maskWM        : [B, 1, H, W, D] — máscara de white matter.
        pred          : [B, 1, H, W, D] — resíduo predito pelo decoder.
        media_vizinhos: [B, 1, H, W, D] — média ponderada dos vizinhos.
        """
        
        
        pred_final = torch.nan_to_num(
            pred_final,
            nan=0.0,
            posinf=5.0,
            neginf=0.0,
        )

        target = torch.nan_to_num(
            target,
            nan=0.0,
            posinf=5.0,
            neginf=0.0,
        )

        # Garante float32 para estabilidade (especialmente com SSIM)
        pred_final = pred_final.float()
        target     = target.float()
        mask       = mask.detach().float()
        maskWM     = maskWM.detach().float()

        # --- MAPA DE PESOS (cérebro=1.0, WM=wm_multiplier) ---
        weights = mask.clone()
        weights[maskWM > 0.5] = self.wm_multiplier
        weights = weights * mask   # fora do cérebro = 0

        # 1. PERDA DE INTENSIDADE PONDERADA (L1)
        diff_l1  = torch.abs(pred_final - target) * weights
        loss_l1  = diff_l1.sum() / (weights.sum() + 1e-8)

        # 2. PERDA ESTRUTURAL (SSIM)
        # Usa máscara binária simples para não distorcer estatísticas locais
        loss_ssim = self._compute_ssim(
            pred_final,
            target,
            mask
        )

        # 3. PERDA DE GRADIENTE PONDERADA
        loss_grad = self._weighted_gradient_loss(pred_final, target, weights)

        # 4. PERDA RESIDUAL PONDERADA
        loss_res = torch.tensor(0.0, device=pred_final.device)
        if media_vizinhos is not None and pred is not None:
            media_vizinhos = media_vizinhos.float()
            pred           = pred.float()
            res_real = (target - media_vizinhos) * mask
            diff_res = torch.abs(pred - res_real) * weights
            loss_res = diff_res.sum() / (weights.sum() + 1e-8)

        total_loss = (
            self.l1_w   * loss_l1
            + self.ssim_w * loss_ssim
            + self.grad_w * loss_grad
            + self.res_w  * loss_res
        )

        assert not torch.isnan(pred_final).any(), "pred_final tem NaN antes do SSIM"
        assert not torch.isnan(target).any(), "target tem NaN antes do SSIM"

        return total_loss, {
            "l1":   loss_l1,
            "ssim": loss_ssim,
            "grad": loss_grad,
            "res":  loss_res,
        }

    def _weighted_gradient_loss(self, pred, target, weights):
        def gradient(x):
            dx = x[:, :, 1:, :, :]  - x[:, :, :-1, :, :]
            dy = x[:, :, :, 1:, :]  - x[:, :, :, :-1, :]
            dz = x[:, :, :, :, 1:]  - x[:, :, :, :, :-1]
            return dx, dy, dz

        p_dx, p_dy, p_dz = gradient(pred)
        t_dx, t_dy, t_dz = gradient(target)

        # Crop dos pesos para bater com o shape dos gradientes
        w_dx = weights[:, :, 1:, :, :]
        w_dy = weights[:, :, :, 1:, :]
        w_dz = weights[:, :, :, :, 1:]

        loss_dx = (torch.abs(p_dx - t_dx) * w_dx).sum() / (w_dx.sum() + 1e-8)
        loss_dy = (torch.abs(p_dy - t_dy) * w_dy).sum() / (w_dy.sum() + 1e-8)
        loss_dz = (torch.abs(p_dz - t_dz) * w_dz).sum() / (w_dz.sum() + 1e-8)

        return loss_dx + loss_dy + loss_dz
