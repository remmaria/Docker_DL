"""Mascara de cerebro simples por threshold, usada como fallback quando nao
ha uma mascara real (ex.: gerada por `dwi2mask` do MRtrix3 ou `bet` do FSL).
Mantida separada para ser reutilizada pelos scripts de baseline, treino e
avaliacao sem import circular.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

# Rotulos legiveis dos tratos do atlas JHU-ICBM usados na analise focada em
# Alzheimer (fornix, cingulo, uncinado -- ver discussao no protocolo do
# projeto). Nomes seguem exatamente o que aparece nos arquivos do usuario
# (ex.: "JHU-ICBM-labels-1mm_warped_s_CGC_R.nii.gz").
JHU_TRACT_LABELS = {
    "CGC": "Cingulo (giro do cingulo)",
    "CGH": "Cingulo (porcao hipocampal)",
    "FX": "Fornix",
    "FX_ST": "Fornix / stria terminalis",
    "UF": "Fasciculo uncinado",
}


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


def find_jhu_roi_mask(dwi_path: str, tract: str) -> tuple[np.ndarray | None, list[str]]:
    """Procura mascaras do atlas JHU-ICBM (ja registradas/warped por sujeito)
    no MESMO diretorio do dwi_path, seguindo o padrao real observado:
    "JHU-ICBM-labels-1mm_warped_s_<TRACT>_<SIDE>.nii.gz" (ex.: "..._CGC_R.nii.gz"),
    com <SIDE> em {"R", "L"}. Isso e uma convencao DIFERENTE da mascara de
    cerebro (find_mask_path, que concatena sufixo no stem do proprio dwi) --
    aqui o nome do arquivo do atlas nao deriva do nome do dwi, so mora na
    mesma pasta.

    Combina R+L com OR logico quando os dois lados existem. Para tratos nao
    lateralizados no atlas (ex.: FX -- fornix e uma estrutura de linha
    media), tenta tambem o nome sem sufixo de lado.

    Devolve (mascara_bool_ou_None, lista_de_lados_encontrados) -- a lista de
    lados serve so para log/diagnostico (ex.: alertar se so achou "R" e nao
    "L", o que pode indicar um arquivo faltando por engano).
    """
    import nibabel as nib
    parent = Path(dwi_path).parent
    found: dict[str, np.ndarray] = {}
    for side in ("R", "L"):
        candidate = parent / f"JHU-ICBM-labels-1mm_warped_s_{tract}_{side}.nii.gz"
        if candidate.exists():
            found[side] = nib.load(str(candidate)).get_fdata() > 0.5
    if not found:
        # trato nao lateralizado no atlas (ex.: FX) -- tenta sem sufixo de lado
        candidate = parent / f"JHU-ICBM-labels-1mm_warped_s_{tract}.nii.gz"
        if candidate.exists():
            found["_"] = nib.load(str(candidate)).get_fdata() > 0.5
    if not found:
        return None, []
    sides = sorted(found.keys())
    mask = np.zeros_like(next(iter(found.values())), dtype=bool)
    for m in found.values():
        mask |= m
    return mask, sides


def load_roi_masks(dwi_path: str, tracts: list[str], base_mask: np.ndarray | None = None):
    """Carrega e combina (R+L) as mascaras JHU pedidas em `tracts` para um
    sujeito, intersectando cada uma com `base_mask` (tipicamente a mascara
    de cerebro/substancia branca) quando fornecida -- isso evita que erros
    de registro do atlas (bordas fora do cerebro) contaminem as metricas.

    Devolve um dict {tract: mascara_bool} SOMENTE para os tratos cujo
    arquivo foi encontrado; tratos ausentes sao pulados com um aviso
    explicito no stdout (nao um erro -- o script continua sem esse trato
    para esse sujeito).
    """
    rois: dict[str, np.ndarray] = {}
    for tract in tracts:
        mask, sides = find_jhu_roi_mask(dwi_path, tract)
        if mask is None:
            print(f"[aviso] ROI '{tract}' nao encontrada para {dwi_path!r} "
                  f"(procurado em JHU-ICBM-labels-1mm_warped_s_{tract}[_R/_L].nii.gz "
                  f"na mesma pasta) -- pulando esse trato para esse sujeito.",
                  flush=True)
            continue
        if sides == ["R"] or sides == ["L"]:
            print(f"[aviso] ROI '{tract}' para {dwi_path!r}: so encontrou o "
                  f"lado {sides[0]} (esperado R+L, a menos que este seja um "
                  f"trato de linha media) -- confira se nao falta arquivo.",
                  flush=True)
        if base_mask is not None:
            mask = mask & base_mask.astype(bool)
        n_vox = int(mask.sum())
        if n_vox == 0:
            print(f"[aviso] ROI '{tract}' para {dwi_path!r}: 0 voxels apos "
                  f"intersectar com a mascara base -- pulando (registro do "
                  f"atlas pode ter falhado para esse sujeito).", flush=True)
            continue
        rois[tract] = mask
    return rois