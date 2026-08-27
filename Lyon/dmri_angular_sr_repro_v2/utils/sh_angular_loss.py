"""
Termo de loss opcional no dominio angular/SH (ver protocolo, secao 9,
"Prioridade 1" -- RCAE -- e secao 14.5 item 2 -- porte para o RRIN).

Extraido de scripts/04_train_rcae.py (onde foi implementado e validado
primeiro) para um modulo compartilhado, para que scripts/04b_train_rrin.py
possa reusar exatamente a MESMA matematica sem duplicar codigo -- os dois
scripts convergem pro mesmo formato de tensores (B, N, ...) antes de
chamar `compute_sh_angular_loss` (RCAE: N = q_out direcoes-alvo por item;
RRIN: N = sh_q_out trincas amostradas do mesmo sujeito/patch, ver
utils/rrin_dataset.py:RRINTripletDataset e a explicacao no docstring de
`compute_sh_angular_loss` abaixo).

Depende de torch (ao contrario de utils/sh_basis.py, que e puro
numpy/scipy e usado tambem pelo baseline classico sem rede neural) --
por isso fica em modulo separado, para nao forcar uma dependencia de
torch em quem so precisa do ajuste SH classico.
"""
from __future__ import annotations

import numpy as np
import torch

from .sh_basis import real_sh_matrix, max_order_for_n_directions, cart2sphere


def n_coeffs_even(l_max: int) -> int:
    """Numero de coeficientes de uma base SH real so com ordens PARES ate
    `l_max` (inclusive): R = sum_{l=0,2,...,l_max} (2l+1) = (l_max+1)(l_max+2)/2.
    Mesma formula usada implicitamente em utils/sh_basis.py
    (max_order_for_n_directions/real_sh_matrix), so exposta aqui separada
    pra usar em avisos de startup sem precisar montar a matriz inteira."""
    return (l_max + 1) * (l_max + 2) // 2


def sh_column_degrees(l_max: int) -> np.ndarray:
    """Grau `l` de cada coluna da matriz devolvida por
    utils.sh_basis.real_sh_matrix -- essa funcao NAO devolve os graus
    junto (ao contrario da versao alternativa em spherical_harmonics.py),
    entao replicamos aqui o MESMO laco de enumeracao de colunas usado
    dentro dela (`for l in range(0, l_max+1, 2): for m in range(-l, l+1)`)
    pra saber quais colunas sao "ordem alta" sem duplicar a matematica da
    base em si."""
    ls = []
    for l in range(0, l_max + 1, 2):
        for _m in range(-l, l + 1):
            ls.append(l)
    return np.array(ls, dtype=np.float64)


def compute_sh_angular_loss(pred, target_vols, target_bvecs, target_mask,
                             l_max_cap=8, high_order_min=4):
    """Termo de loss opcional no dominio angular/SH: alem da MAE sobre o
    sinal bruto (que pesa TODO voxel igual, dominado pelos voxels de fibra
    unica que sao maioria do volume), penaliza tambem o erro nos
    coeficientes SH de ordem >= `high_order_min` (default l>=4) da FOD
    reconstruida -- exatamente os coeficientes que carregam a informacao
    de fibra cruzando que a POC de CSD (poc_csd_direction_count.py /
    sh_energy_by_order) mostrou serem onde o RCAE ja leva vantagem sobre o
    baseline SH.

    Formato de entrada esperado -- GENERICO o suficiente para servir tanto
    ao RCAE (N = q_out direcoes-alvo de um mesmo item de sequencia N-para-M)
    quanto ao RRIN (N = sh_q_out trincas amostradas do mesmo sujeito/patch,
    cada uma prevendo UMA direcao por chamada do modelo -- ver
    scripts/04b_train_rrin.py, que empilha essas N previsoes num eixo extra
    antes de chamar esta funcao, exatamente para poder reaproveita-la sem
    modificacao):
        pred, target_vols: (B, N, ..., D, H, W) -- mesma forma, C pode ser
            1 ou omitido, o que importa e que o eixo 1 seja "direcao/trinca".
        target_bvecs: (B, N, 3).
        target_mask: (B, N) bool/int -- False/0 marca posicoes de PADDING
            (RCAE: sujeitos do batch com menos de q_out direcoes reais,
            via collate_variable_targets; RRIN: sujeitos com menos de
            sh_q_out trincas validas disponiveis, ver RRINTripletDataset) a
            excluir do ajuste SH daquele item.

    Projeta pred/target nas MESMAS direcoes-alvo (target_bvecs) usando a
    mesma base SH real do baseline classico (utils/sh_basis.py), por item
    do batch. A projecao (pseudo-inversa da matriz de base, calculada a
    partir dos bvecs, sem gradiente) e um mapeamento LINEAR nas direcoes --
    o matmul com pred/target continua totalmente diferenciavel.

    Limitacao importante (RCAE, ver protocolo secao 9): com q_out=10
    (default do treino), so ha direcoes-alvo suficientes pra sustentar ate
    ordem l=2 (R=6 coeficientes; l=4 precisaria de 15). Para o RRIN, o
    "q_out efetivo" e `sh_q_out` (novo hiperparametro, independente de
    quantas direcoes de entrada o RRIN usa por triplet -- ver
    scripts/04b_train_rrin.py --sh-loss-q-out), sujeito ao mesmo piso:
    l=4 precisa sh_q_out>=15, l=6 precisa >=28, l=8 precisa >=45 -- e
    limitado tambem pelo numero de trincas VALIDAS que o sujeito realmente
    tem para aquele (shell, n_level) (ver protocolo secao 10.3 sobre a
    fracao `valid` cair bastante em n_level baixo). Itens do batch sem
    direcoes-alvo validas suficientes pra sustentar `high_order_min` sao
    pulados (nao contribuem pra este termo; a loss de sinal continua
    normal pra eles)."""
    device = pred.device
    losses = []
    target_mask_cpu = target_mask.detach().cpu().numpy()
    for b in range(pred.shape[0]):
        valid = target_mask_cpu[b].astype(bool)
        n_valid = int(valid.sum())
        if n_valid == 0:  # item totalmente padding -- sem direcao valida
            continue
        bvecs_b = target_bvecs[b, valid].detach().cpu().numpy()
        l_max = min(max_order_for_n_directions(n_valid), l_max_cap)
        if l_max < high_order_min:
            # direcoes-alvo validas neste item nao sustentam a ordem pedida
            # -- pula (nao inventa coeficiente de ordem alta sem dado pra
            # sustentar), so a loss de sinal cobre este item.
            continue
        theta, phi = cart2sphere(bvecs_b)
        Bmat = real_sh_matrix(theta, phi, l_max)  # (n_valid, R)
        ls = sh_column_degrees(l_max)             # (R,) -- mesma ordem de colunas
        high_idx = np.nonzero(ls >= high_order_min)[0]
        if high_idx.size == 0:
            continue
        Bmat_t = torch.as_tensor(Bmat, dtype=pred.dtype, device=device)
        pinv = torch.linalg.pinv(Bmat_t)  # (R, n_valid)
        valid_idx = torch.as_tensor(np.nonzero(valid)[0], device=device, dtype=torch.long)
        pred_b = pred[b, valid_idx].reshape(n_valid, -1)          # (n_valid, voxels)
        target_b = target_vols[b, valid_idx].reshape(n_valid, -1)
        coeffs_pred = pinv @ pred_b        # (R, voxels)
        coeffs_target = pinv @ target_b
        high_idx_t = torch.as_tensor(high_idx, device=device, dtype=torch.long)
        err = (coeffs_pred[high_idx_t] - coeffs_target[high_idx_t]).abs()
        losses.append(err.mean())
    if not losses:
        return torch.zeros((), device=device, dtype=pred.dtype)
    return torch.stack(losses).mean()