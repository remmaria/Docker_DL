"""Mascara de cerebro simples por threshold, usada como fallback quando nao
ha uma mascara real (ex.: gerada por `dwi2mask` do MRtrix3 ou `bet` do FSL).
Mantida separada para ser reutilizada pelos scripts de baseline, treino e
avaliacao sem import circular.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


def simple_brain_mask(b0_mean: np.ndarray, percentile: float = 40.0) -> np.ndarray:
    nonzero = b0_mean[b0_mean > 0]
    if nonzero.size == 0:
        return np.zeros_like(b0_mean, dtype=bool)
    thresh = np.percentile(nonzero, percentile)
    return b0_mean > thresh


def strip_nii_ext(path_str: str) -> str:
    """Remove .nii.gz ou .nii do fim do caminho, sem presumir nenhum
    sufixo de nome especifico (ex.: nao assume convencao BIDS "_dwi").
    """
    if path_str.endswith(".nii.gz"):
        return path_str[: -len(".nii.gz")]
    if path_str.endswith(".nii"):
        return path_str[: -len(".nii")]
    return path_str


def find_mask_path(dwi_path: str, mask_suffix: str) -> Path | None:
    """Procura uma mascara chamada "<dwi_sem_extensao><mask_suffix>" na
    mesma pasta do dwi (ex.: dwi_path=".../bgpdwis_PA_geomcorr.nii" +
    mask_suffix="_mask3d.nii.gz" -> ".../bgpdwis_PA_geomcorr_mask3d.nii.gz").
    Ajuste --mask-suffix na linha de comando dos scripts para bater com a
    sua convencao real de nome de mascara.
    """
    stem = strip_nii_ext(str(dwi_path))
    candidate = Path(stem + mask_suffix)
    return candidate if candidate.exists() else None


# ATENCAO: o default aqui e "_mask3d.nii.gz" -- CONFIRMADO como o sufixo
# real dos seus arquivos de mascara. Ate essa correcao, utils/dataset.py e
# scripts/04_train_rcae.py (unicos consumidores sem --mask-suffix exposto
# na CLI) usavam o default ANTIGO "_brainmask.nii.gz", que nao bate com
# nenhum arquivo real -- ou seja, TODO treino/validacao do RCAE ate agora
# caiu silenciosamente no fallback `simple_brain_mask` (threshold simples
# por percentil do b0) em vez de usar a mascara real. Os outros scripts
# (03_baseline_sh_interpolation.py, 05_reconstruct_rcae.py,
# 07_downstream_dti_noddi.py, 08_downstream_tractography.py) sempre
# tiveram o default certo, porque foram escritos depois e ja tinham
# --mask-suffix com "_mask3d.nii.gz". Ver aviso abaixo (print no fallback)
# pra essa classe de bug nao passar batido de novo.
def load_or_build_mask(dwi_path: str, b0_mean: np.ndarray, mask_suffix: str = "_mask3d.nii.gz"):
    mask_path = find_mask_path(dwi_path, mask_suffix)
    if mask_path is not None:
        import nibabel as nib
        return nib.load(str(mask_path)).get_fdata() > 0.5
    # AVISO (nao silencioso mais) -- sem isso, um --mask-suffix errado (ou
    # arquivo de mascara faltando pra 1 sujeito especifico) passava batido,
    # trocando a mascara real por uma heuristica bem mais grosseira sem
    # nenhum sinal no log. Print (nao warnings.warn) porque precisa
    # aparecer no .out do SLURM mesmo sem nenhuma configuracao extra de
    # logging.
    print(f"[aviso] mascara real nao encontrada para {dwi_path!r} com sufixo "
          f"{mask_suffix!r} -- usando fallback simple_brain_mask (threshold "
          f"por percentil do b0). Confira --mask-suffix se isso for "
          f"inesperado.", flush=True)
    return simple_brain_mask(b0_mean)