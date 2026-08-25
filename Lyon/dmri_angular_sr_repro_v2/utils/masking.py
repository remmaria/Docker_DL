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

# Rotulos legiveis das mascaras de SEGMENTACAO por sujeito (ja lateralizadas
# quando aplicavel, ao contrario dos tratos JHU acima que sao combinados
# R+L) -- nomes seguem exatamente o meio do nome de arquivo do usuario, ex.
# ".../bgpdwis_PA_geomcorr_maskseg_CbWM_L_e1.nii.gz" -> nome "CbWM_L".
# So documentacao/leitura -- find_seg_roi_mask nao depende deste dict.
SEG_ROI_LABELS = {
    "WM": "Substancia branca (mascara inteira)",
    "CbWM_L": "Substancia branca cerebelar (esquerda)",
    "CbWM_R": "Substancia branca cerebelar (direita)",
    "Ctx_L": "Cortex (esquerdo)",
    "Ctx_R": "Cortex (direito)",
    "Hipp_L": "Hipocampo (esquerdo)",
    "Hipp_R": "Hipocampo (direito)",
    "SubCtx_L": "Substancia cinzenta subcortical (esquerda)",
    "SubCtx_R": "Substancia cinzenta subcortical (direita)",
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


def find_seg_roi_mask(dwi_path: str, name: str) -> np.ndarray | None:
    """Procura uma mascara de SEGMENTACAO por sujeito (tipo de arquivo
    diferente do atlas JHU: aqui o nome do arquivo DERIVA do nome do dwi,
    igual find_mask_path, so que com um meio de nome extra) seguindo o
    padrao observado nos arquivos do usuario:
        "<dwi_sem_extensao>_maskseg_<name>_e1.nii.gz"  (a maioria)
        "<dwi_sem_extensao>_maskseg_<name>.nii.gz"     (alguns, sem "_e1")
    Ex.: dwi=".../bgpdwis_PA_geomcorr.nii" + name="CbWM_L" ->
    ".../bgpdwis_PA_geomcorr_maskseg_CbWM_L_e1.nii.gz".

    Ao contrario de find_jhu_roi_mask, cada `name` aqui ja e uma mascara
    unica e completa (as vezes ja lateralizada, ex. "CbWM_L", "Ctx_R" --
    ver SEG_ROI_LABELS) -- nao ha combinacao R+L automatica; se quiser as
    duas mascaras separadas, peca os dois nomes em ROI_TRACTS (ex.
    "CbWM_L,CbWM_R"); se quiser R+L juntos, junte-as voce mesmo (ex. com
    fslmaths antes) e trate como um `name` novo.

    Devolve a mascara bool, ou None se nenhum dos dois padroes de nome
    existir.
    """
    import nibabel as nib
    stem = strip_nii_ext(str(dwi_path))
    for suffix in (f"_maskseg_{name}_e1.nii.gz", f"_maskseg_{name}.nii.gz"):
        candidate = Path(stem + suffix)
        if candidate.exists():
            return nib.load(str(candidate)).get_fdata() > 0.5
    return None


def load_roi_masks(dwi_path: str, tracts: list[str], base_mask: np.ndarray | None = None):
    """Carrega as mascaras de ROI pedidas em `tracts` para um sujeito,
    intersectando cada uma com `base_mask` (tipicamente a mascara de
    cerebro/substancia branca) quando fornecida -- isso evita que erros de
    registro/segmentacao (bordas fora do cerebro) contaminem as metricas.

    Cada nome em `tracts` pode ser (tentados nesta ordem):
    1. um trato do atlas JHU-ICBM (find_jhu_roi_mask -- combina R+L
       automaticamente, ex. "FX", "CGC", "CGH", "UF"); ou
    2. uma mascara de segmentacao por sujeito (find_seg_roi_mask -- ja
       lateralizada quando aplicavel, ver SEG_ROI_LABELS, ex. "WM",
       "CbWM_L", "Ctx_R", "Hipp_R", "SubCtx_L").
    Os dois tipos podem ser misturados na mesma lista (ex.
    ROI_TRACTS="FX,CGC,CGH,UF,WM,CbWM_L,Ctx_R,Hipp_R,SubCtx_L").

    Devolve um dict {nome: mascara_bool} SOMENTE para as ROIs cujo arquivo
    foi encontrado (em qualquer uma das duas convencoes); ROIs ausentes sao
    puladas com um aviso explicito no stdout (nao um erro -- o script
    continua sem essa ROI para esse sujeito).
    """
    rois: dict[str, np.ndarray] = {}
    for tract in tracts:
        mask, sides = find_jhu_roi_mask(dwi_path, tract)
        if mask is not None and (sides == ["R"] or sides == ["L"]):
            print(f"[aviso] ROI '{tract}' para {dwi_path!r}: so encontrou o "
                  f"lado {sides[0]} (esperado R+L, a menos que este seja um "
                  f"trato de linha media) -- confira se nao falta arquivo.",
                  flush=True)
        if mask is None:
            mask = find_seg_roi_mask(dwi_path, tract)
        if mask is None:
            print(f"[aviso] ROI '{tract}' nao encontrada para {dwi_path!r} "
                  f"(procurado como trato JHU "
                  f"'JHU-ICBM-labels-1mm_warped_s_{tract}[_R/_L].nii.gz' e como "
                  f"mascara de segmentacao '<dwi>_maskseg_{tract}[_e1].nii.gz', "
                  f"ambos na mesma pasta) -- pulando essa ROI para esse sujeito.",
                  flush=True)
            continue
        if base_mask is not None:
            mask = mask & base_mask.astype(bool)
        n_vox = int(mask.sum())
        if n_vox == 0:
            print(f"[aviso] ROI '{tract}' para {dwi_path!r}: 0 voxels apos "
                  f"intersectar com a mascara base -- pulando (registro/"
                  f"segmentacao pode ter falhado para esse sujeito).", flush=True)
            continue
        rois[tract] = mask
    return rois