#!/usr/bin/env python3
"""
Etapa 8 (opcional, a mais forte para a tese, tambem a mais fragil):
compara tratografia (CSD via MRtrix3) entre ground truth, o "piso nativo"
(sem reconstrucao nenhuma -- so as n_level direcoes reais de entrada, ver
scripts/poc_native_order_curve.py/secao 22 do addendum) e reconstrucoes
(baseline SH, RCAE, RRIN-star, ou qualquer outra via --extra-method), tanto
em densidade de streamlines whole-brain (Dice entre mapas binarizados)
quanto -- ADITIVO 2026-09-02, ver addendum secao 24 -- por TRATO especifico
(--roi-tracts): contagem de streamlines que atravessam cada trato,
comprimento medio, e Dice de densidade restrito aquele trato. Esse e' o
teste mais direto de "a reconstrucao muda a conclusao clinica ou nao" desta
linha de experimentos -- todas as metricas anteriores (peak matching,
FA_r2, energy_frac_high_order) sao proxies; contagem de streamline
atravessando um trato relevante para Alzheimer's (FX/CGC/CGH/UF, ver
07_downstream_dti_noddi.py) e' o numero que um resultado clinico de fato
usaria.

Para tractometria fina por feixe (perfil ao longo do trajeto, bundle
recognition automatico), prefira `scilpy` ou `dipy.segment` num passo
posterior -- aqui o objetivo e' comparacao de plausibilidade/integridade
por trato ja segmentado (mesmas mascaras JHU/segmentacao que
06/07/11/12 ja usam), nao bundle recognition do zero.

Requer MRtrix3 instalado e no PATH (dwi2response, dwi2fod, tckgen, tckedit,
tckinfo, tckstats, tckmap, mrconvert). Chama tudo via subprocess; se algum
passo falhar (sujeito ou metodo), e' pulado com aviso (nao derruba o resto
do lote) -- mesma disciplina de fit_failed ja usada em
scripts/11_peak_confusion_by_roi.py.

TRATOGRAFIA DETERMINISTICA (ADITIVO 2026-09-02, decisao explicita da
usuaria via AskUserQuestion -- ver addendum secao 24): `tckgen` sem
`-algorithm` usa o default do MRtrix, iFOD2 (PROBABILISTICO) -- apesar do
docstring antigo deste arquivo dizer "determinístico", o codigo original
nunca de fato passava um algoritmo, entao rodava probabilistico na
pratica. Corrigido: `--algorithm` agora tem default explicito
`SD_STREAM` (determinístico, segue o pico dominante da FOD -- mais barato,
reprodutivel, sem variancia entre rodadas) -- `--algorithm iFOD2` disponivel
pra quem quiser a versao probabilistica depois.

SEEDING WHOLE-BRAIN + FILTRO POR ROI (ADITIVO 2026-09-02, escolha
explicita da usuaria -- alternativa mais barata seria semear so dentro da
ROI do trato, mas isso so testaria a fibra local, nao se a reconstrucao
ainda sustenta a CONEXAO atraves de uma regiao de cruzamento pelo
caminho): `tckgen` semeia em toda a mascara de cerebro (`-seed_image
mask.nii.gz`, como sempre), e o filtro por trato acontece DEPOIS, via
`tckedit -include <roi>.nii.gz` sobre o track whole-brain ja gerado (uma
unica geracao de streamlines por metodo/sujeito, reaproveitada pra
filtrar por quantos tratos forem pedidos -- nao gera um tckgen novo por
ROI).

PISO NATIVO (--subsampled-only/--triplets-dir, ADITIVO): ao contrario dos
metodos de reconstrucao (que mantem a contagem NOMINAL total de direcoes,
so os alvos held-out mudam de valor), o piso nativo e' uma aquisicao
GENUINAMENTE menor -- so a shell de interesse (`--shell-b`) e' mantida, e
dela so sobram os `n_level` indices de entrada REAIS (via
`<tag>_rrin_triplets.npz`, mesmo `target_idx` que
scripts/11_peak_confusion_by_roi.py/07_downstream_dti_noddi.py ja usam
como `exclude_idx`) -- outras shells (se existirem) sao descartadas, mesma
convencao de fit_peaks/fit_dti nesses dois scripts (CSD single-shell). Sem
reconstrucao nenhuma: e' literalmente "o que a tratografia conseguiria
sustentar so com o protocolo mais curto", a base de comparacao pra saber
se qualquer metodo de reconstrucao esta de fato acrescentando estrutura
ou so preenchendo o volume sem ganho tractografico real.

FILTRO DE DUAS ROIS (--roi-pair, ADITIVO 2026-09-02, ver addendum secao
25.2): a inspecao visual dos .tck de FX (secoes 25/25.1 do addendum)
revelou que alguns ramos laterais que saem do corpo do forice, contados
como "FX" pelo filtro de UMA ROI so, sao provavelmente a stria terminalis
(feixe distinto, anatomicamente paralelo/adjacente ao forice -- o proprio
JHU_TRACT_LABELS em utils/masking.py ja distingue "FX" de "FX_ST") --
qualquer streamline que so' TOQUE a mascara do trato conta, mesmo que
pertenca a um feixe vizinho que so' "raspa" a borda da ROI de passagem.
Isso nao e' um bug do desenho whole-brain-seed + filtro por ROI (afeta GT
e todos os metodos igualmente, entao nao invalida a comparacao relativa
entre eles), mas infla a contagem ABSOLUTA de streamlines de "FX" com
fibra que nao pertence ao circuito. `--roi-pair NOME=ROI_A+ROI_B`
(repetivel) aperta o filtro: exige que a streamline atravesse AMBAS as
ROIs (dois `tckedit -include`, que o proprio MRtrix combina em AND) em
vez de qualquer uma isolada -- ex. exigir FX E Hipp_R (mascara de
segmentacao do hipocampo direito por sujeito, ja suportada por
utils.masking.find_seg_roi_mask sem nenhum codigo novo) isola conexoes
fornix-hipocampo genuinas, excluindo a stria terminalis (que nao passa
pelo hipocampo do mesmo jeito). ROI_A/ROI_B usam os mesmos nomes de
--roi-tracts (tratos JHU ou mascaras de segmentacao por sujeito) e nao
precisam estar tambem em --roi-tracts -- sao carregados automaticamente
so' como insumo do par, sem virar uma linha "roi=" propria no CSV alem do
proprio par. Ex.: --roi-pair FX_Hipp_R=FX+Hipp_R --roi-pair
FX_Hipp_L=FX+Hipp_L.

DILATACAO DE ROI (--roi-dilate NOME=N, ADITIVO 2026-09-02, ver addendum
secao 26.1): rodando --roi-pair FX+Hipp_R pela primeira vez com dado real
(secao 26), o ground truth caiu de 567 streamlines em "FX" sozinho pra
so' 28 em "FX_Hipp_R" -- queda grande demais pra ser so' remocao de
contaminacao de stria terminalis (secao 25.2), ja que fisiologicamente
quase toda a crus do fornix deveria terminar perto do hipocampo. Causa
provavel: `Hipp_L`/`Hipp_R` (find_seg_roi_mask, mascara de SEGMENTACAO)
e' substancia CINZENTA do hipocampo -- a fimbria (parte do fornix que
corre colada na superficie do hipocampo antes de penetrar nele) e'
substancia branca fina, e um erro de registro de 1-2 voxels entre a
segmentacao e a fimbria de verdade ja derruba a streamline do filtro
`tckedit -include` inteiro (que exige intersecao literal de voxel), mesmo
quando a conexao anatomica e' genuina. `--roi-dilate NOME=N` dilata a
mascara daquele nome em N voxels (scipy.ndimage.binary_dilation, depois
re-intersectada com a mascara de cerebro do sujeito) antes de gravar o
nifti usado pelo tckedit -- absorve esse erro de borda/registro sem
convidar de volta trato totalmente errado (N=1-2 nao alcanca a stria
terminalis, que esta mais distante). Recomendado para mascaras de
segmentacao tipo Hipp_L/Hipp_R; NAO recomendado para os tratos
LATERALIZADOS do atlas JHU (CGC_L/CGC_R/CGH_L/CGH_R/UF_L/UF_R, secao
25.3) -- dilatar em direcao a linha media reintroduziria o mesmo
vazamento de fibra comissural que a separacao por lado corrigiu.

Uso (comparacao completa: GT, piso nativo, baseline_sh, rcae, e um metodo
extra qualquer, restrito aos 4 tratos ja usados no resto da pipeline):
    python scripts/08_downstream_tractography.py \
        --manifest work_dir/manifest.csv \
        --baseline-dir work_dir/baseline_recon \
        --rcae-dir work_dir/rcae_recon \
        --extra-method rrin_star=work_dir/rrin_star_recon \
        --subsampled-only --triplets-dir work_dir/subsampling \
        --roi-tracts FX,CGC,CGH,UF \
        --shell-b 1000 --n-level 16 \
        --out-dir work_dir/tractography \
        --n-streamlines 200000
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_dwi, load_bval_bvec, split_shells
from utils.masking import load_or_build_mask, load_roi_masks


def _via_bash(cmd: list[str]) -> list[str]:
    """Encapsula `cmd` numa chamada `bash -c "..."` em vez de executar via
    execvp direto.

    BUG REAL encontrado no cluster (2026-09-02): no ambiente da usuaria,
    `module load mrtrix3/<versao>` NAO poe binarios de verdade no PATH --
    define FUNCOES BASH (`tckgen () { singularity exec .../mrtrix3-*.sif
    tckgen "$@"; }`, exportadas via `export -f`, mesmo padrao comum de
    modulo que empacota um container singularity). Isso funciona
    perfeitamente quando chamado DIRETO dentro do proprio script .sh (a
    linha de diagnostico `tckgen -version` do wrapper sbatch funcionava),
    mas `subprocess.run(["mrconvert", ...])` sem shell nenhum resolve o
    comando via execvp/PATH, que nao enxerga funcoes de shell -- daí o
    erro real visto no .out: "[Errno 2] No such file or directory:
    'mrconvert'", apesar do `tckgen -version` do proprio wrapper ter
    funcionado segundos antes. Corrigido: toda chamada de binario MRtrix
    agora passa por um `bash -c` explicito (nao o default de
    `subprocess.run(shell=True)`, que usa /bin/sh -- functions exportadas
    via `export -f` (formato `BASH_FUNC_nome%%=`) só sao reconstruidas por
    uma bash de verdade, nao por dash/sh). `shlex.quote` em cada argumento
    evita qualquer problema de espaco/caractere especial nos paths.
    """
    return ["bash", "-c", " ".join(shlex.quote(str(c)) for c in cmd)]


def run(cmd: list[str], **kwargs):
    print(" $", " ".join(str(c) for c in cmd))
    subprocess.run(_via_bash(cmd), check=True, **kwargs)


def _nthreads_flag(nthreads: int | None) -> list[str]:
    """`-nthreads N` pra passar pros comandos MRtrix que suportam
    multithreading (dwi2response/dwi2fod/tckgen/tckedit/tckmap -- todos
    CPU, MRtrix3 nao tem nenhum passo acelerado por GPU, ver addendum
    secao 24.2: "usar GPU ajuda?" -- nao, a alavanca real e' essa).
    `nthreads=None` (default) omite a flag e deixa o MRtrix auto-detectar
    (comportamento de sempre, preservado); `main()` por default preenche
    isso com `$SLURM_CPUS_PER_TASK` quando o job roda via sbatch, pra
    garantir que o MRtrix de fato usa toda a alocacao pedida em vez de
    adivinhar sozinho quantos cores estao disponiveis dentro do job.
    """
    return [] if not nthreads else ["-nthreads", str(nthreads)]


def _resolve_shell_key(shells: dict, shell_b: float, tol: float) -> float:
    # Duplicado de scripts/11_peak_confusion_by_roi.py:_resolve_shell_key
    # (convencao do projeto: sem import cruzado entre scripts de etapa).
    best_key, best_diff = None, None
    for k in shells:
        if k == 0:
            continue
        diff = abs(k - shell_b)
        if best_diff is None or diff < best_diff:
            best_key, best_diff = k, diff
    if best_key is None or best_diff > tol:
        raise RuntimeError(f"shell {shell_b} nao encontrada (tol={tol})")
    return best_key


def native_floor_volume(data, bvals, bvecs, shell_b, shell_tol, target_idx):
    """Constroi a aquisicao "piso nativo" de verdade -- so a shell de
    interesse, so as direcoes que NAO estao em `target_idx` (as
    `n_level` direcoes de entrada reais), mais os b0. Ao contrario de
    `build_full_volume` (que mantem a contagem NOMINAL total de direcoes,
    reconstruindo os alvos held-out), aqui o volume de saida tem
    literalmente MENOS volumes -- e' a mesma aquisicao subamostrada que
    `--subsampled-only` de scripts/11_peak_confusion_by_roi.py ja compara
    no dominio de picos/CSD, agora levada ate tratografia de verdade.
    Devolve (data_subset, bvals_subset, bvecs_subset) ou None se a shell
    nao existir dentro da tolerancia.
    """
    shells = split_shells(bvals, tol=shell_tol)
    try:
        shell_key = _resolve_shell_key(shells, shell_b, shell_tol)
    except RuntimeError as exc:
        print(f"[aviso] piso nativo: {exc}")
        return None
    idx = np.concatenate([shells[0], shells[shell_key]])
    idx.sort()
    idx = np.setdiff1d(idx, np.asarray(target_idx), assume_unique=False)
    return data[..., idx], bvals[idx], bvecs[idx]


def build_full_volume(gt_data, recon_dir, tag, shell_b, n_level):
    sub_dir = Path(recon_dir) / tag / f"shell{int(shell_b)}" / f"n{n_level}"
    recon_path = sub_dir / "recon_target.nii.gz"
    target_idx_path = sub_dir / "target_idx.npy"
    if not recon_path.exists() or not target_idx_path.exists():
        return None
    import nibabel as nib
    recon = nib.load(str(recon_path)).get_fdata().astype(np.float32)
    target_idx = np.load(target_idx_path)
    out = gt_data.copy()
    out[..., target_idx] = recon
    return out


def tractography_pipeline(work_dir: Path, data, affine, bvals, bvecs, mask, n_streamlines: int,
                           algorithm: str = "SD_STREAM", nthreads: int | None = None):
    """Gera o track whole-brain (`tracks.tck`) e o mapa de densidade
    whole-brain (`density_whole_mask.nii.gz`) pra um volume/metodo. O
    filtro por ROI especifica (`filter_track_by_roi`, abaixo) roda POR
    CIMA do `tracks.tck` gerado aqui -- uma unica geracao de streamlines
    por metodo, nao uma por trato.
    `algorithm`: passado direto pra `tckgen -algorithm` (default
    SD_STREAM = deterministico, segue o pico dominante da FOD; "iFOD2" =
    probabilistico, o default do proprio MRtrix se `-algorithm` nao fosse
    passado -- ver docstring do modulo, secao "TRATOGRAFIA DETERMINISTICA").
    `nthreads`: ver `_nthreads_flag` -- MRtrix3 e' CPU-only (sem GPU em
    nenhum passo), essa e' a alavanca real de velocidade.
    """
    import nibabel as nib
    work_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), work_dir / "dwi.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), work_dir / "mask.nii.gz")
    np.savetxt(work_dir / "dwi.bval", bvals.reshape(1, -1), fmt="%d")
    np.savetxt(work_dir / "dwi.bvec", bvecs.T, fmt="%.6f")

    nthreads_flag = _nthreads_flag(nthreads)

    dwi_mif = work_dir / "dwi.mif"
    run(["mrconvert", str(work_dir / "dwi.nii.gz"), str(dwi_mif),
         "-fslgrad", str(work_dir / "dwi.bvec"), str(work_dir / "dwi.bval"), "-force"])

    response = work_dir / "response.txt"
    run(["dwi2response", "tournier", str(dwi_mif), str(response), *nthreads_flag, "-force"])

    fod = work_dir / "fod.mif"
    run(["dwi2fod", "csd", str(dwi_mif), str(response), str(fod),
         "-mask", str(work_dir / "mask.nii.gz"), *nthreads_flag, "-force"])

    tck = work_dir / "tracks.tck"
    run(["tckgen", str(fod), str(tck), "-algorithm", algorithm,
         "-seed_image", str(work_dir / "mask.nii.gz"),
         "-mask", str(work_dir / "mask.nii.gz"), "-select", str(n_streamlines),
         *nthreads_flag, "-force"])

    mask_path = work_dir / "mask.nii.gz"
    density = density_map_for_track(tck, mask_path, work_dir / "density_whole_mask.nii.gz",
                                     nthreads=nthreads)
    return tck, mask_path, density


def density_map_for_track(track_path: Path, template_path: Path, out_path: Path,
                           nthreads: int | None = None):
    import nibabel as nib
    run(["tckmap", str(track_path), str(out_path), "-template", str(template_path),
         *_nthreads_flag(nthreads), "-force"])
    return nib.load(str(out_path)).get_fdata()


def filter_track_by_roi(track_path: Path, roi_mask_paths: list[Path], out_track_path: Path,
                         nthreads: int | None = None):
    """Filtra, de um track ja gerado (whole-brain), so as streamlines que
    ATRAVESSAM TODAS as ROIs dadas (`tckedit -include` repetido uma vez
    por ROI -- multiplos `-include` sao combinados em AND pelo proprio
    MRtrix: so sobrevive a streamline que atravessa CADA UMA das ROIs
    listadas, nao qualquer uma delas). Com uma unica ROI (o caso de
    sempre, `--roi-tracts`), isso e' identico ao comportamento antigo --
    a lista com 2+ ROIs e' o mecanismo de `--roi-pair` (ver docstring do
    modulo, secao "FILTRO DE DUAS ROIS"): exigir que a streamline passe
    por DUAS regioes (ex. FX E hipocampo) filtra fibra de estruturas
    vizinhas que so' "raspam" a borda de uma unica ROI sem genuinamente
    pertencer aquele circuito (achado real, ver addendum secao 25.2).
    """
    cmd = ["tckedit", str(track_path), str(out_track_path)]
    for p in roi_mask_paths:
        cmd += ["-include", str(p)]
    cmd += [*_nthreads_flag(nthreads), "-force"]
    run(cmd)


def tckinfo_count(track_path: Path) -> int:
    """Numero exato de streamlines num .tck -- `tckinfo -count` forca a
    recontagem real (o header do arquivo pode ter só uma estimativa,
    especialmente apos `tckedit`, que reescreve o arquivo mas nem sempre
    atualiza o campo de contagem do header sem essa flag).
    """
    out = subprocess.run(_via_bash(["tckinfo", str(track_path), "-count"]), check=True,
                          capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().lower().startswith("count:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"nao consegui parsear a contagem de streamlines de 'tckinfo {track_path}'\n{out}")


def tckstats_mean_length(track_path: Path) -> float:
    """Comprimento medio (mm) das streamlines de um .tck, via
    `tckstats -output mean` (imprime so o valor pedido em stdout, sem
    cabecalho -- pensado pra uso em script). Streamline vazio (0
    streamlines) faz o proprio MRtrix reclamar -- tratado no chamador via
    o mesmo try/except de "tratografia falhou" ja usado no resto do script.
    """
    out = subprocess.run(_via_bash(["tckstats", str(track_path), "-output", "mean"]), check=True,
                          capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[-1])


def dice_from_density(density_a, density_b, percentile: float = 50.0):
    """Binariza os mapas de densidade de streamlines por percentil (dentro
    dos voxels com streamline > 0) e calcula o coeficiente de Dice.
    """
    def binarize(d):
        nz = d[d > 0]
        if nz.size == 0:
            return np.zeros_like(d, dtype=bool)
        thr = np.percentile(nz, percentile)
        return d >= thr

    a = binarize(density_a)
    b = binarize(density_b)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2 * inter / denom)


def roi_metrics_for_method(track_path: Path, mask_path: Path, roi_name: str,
                            roi_mask_paths: list[Path], work_dir: Path,
                            gt_density_roi: np.ndarray | None,
                            nthreads: int | None = None):
    """Filtra o track whole-brain pela(s) ROI(s) (uma so' = trato normal;
    duas = `--roi-pair`, AND das duas), e devolve
    (n_streamlines, mean_length_mm, density_map, dice_vs_gt_ou_None).
    `gt_density_roi=None` sinaliza que este E' o proprio ground truth
    (nao compara Dice contra si mesmo, so guarda a densidade pra servir de
    referencia pros outros metodos).
    """
    filtered = work_dir / f"tracks_{roi_name}.tck"
    filter_track_by_roi(track_path, roi_mask_paths, filtered, nthreads=nthreads)
    n_stream = tckinfo_count(filtered)
    # n_stream==0 (trato genuinamente perdido nessa reconstrucao) nao e'
    # erro -- e' o proprio resultado. tckmap sobre um .tck vazio ainda
    # produz um mapa de densidade zero em todo lugar (mapa "vazio", nao
    # falha) -- Dice contra o GT nesse caso da' 0.0 (nenhuma intersecao),
    # que e' a leitura correta (trato inteiramente perdido).
    mean_length = tckstats_mean_length(filtered) if n_stream > 0 else float("nan")
    density = density_map_for_track(filtered, mask_path, work_dir / f"density_{roi_name}.nii.gz",
                                     nthreads=nthreads)
    dice = None if gt_density_roi is None else dice_from_density(density, gt_density_roi)
    return n_stream, mean_length, density, dice


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--baseline-dir", default=None)
    ap.add_argument("--rcae-dir", default=None)
    ap.add_argument("--extra-method", action="append", default=[],
                     metavar="NOME=DIR",
                     help="Metodo adicional de reconstrucao (mesma convencao de "
                          "06_evaluate_reconstruction.py/11_peak_confusion_by_roi.py), "
                          "repetivel. Ex.: --extra-method rrin_star=work_dir/rrin_star_recon")
    ap.add_argument("--subsampled-only", action="store_true",
                     help="ADITIVO: tambem tratografa o 'piso nativo' -- so as n_level "
                          "direcoes reais de entrada, sem reconstrucao nenhuma (ver "
                          "docstring do modulo). Requer --triplets-dir.")
    ap.add_argument("--triplets-dir", default=None,
                     help="Pasta com <tag>_rrin_triplets.npz (mesma convencao de "
                          "scripts/11_peak_confusion_by_roi.py) -- obrigatorio se "
                          "--subsampled-only.")
    ap.add_argument("--roi-tracts", default=None,
                     help="ADITIVO, lista separada por virgula (ex. FX,CGC,CGH,UF, mesma "
                          "convencao de 06/07/11/12) -- filtra o track whole-brain por "
                          "cada trato (tckedit -include) e reporta contagem/comprimento/"
                          "Dice POR TRATO, alem da linha whole_mask de sempre. Sem esta "
                          "flag, comportamento identico ao antigo (so whole_mask).")
    ap.add_argument("--roi-pair", action="append", default=[],
                     metavar="NOME=ROI_A+ROI_B",
                     help="ADITIVO, repetivel (ver addendum secao 25.2): exige que a "
                          "streamline atravesse AMBAS ROI_A E ROI_B (AND, via dois "
                          "'tckedit -include'), nao qualquer uma isolada -- filtra fibra "
                          "de estruturas vizinhas que so' 'raspam' a borda de uma unica "
                          "ROI sem pertencer genuinamente aquele circuito (ex.: a stria "
                          "terminalis passa perto do FX e pode contaminar a contagem de "
                          "'FX' sozinho). ROI_A/ROI_B usam os MESMOS nomes de "
                          "--roi-tracts (tratos JHU ou mascaras de segmentacao por "
                          "sujeito, ex. Hipp_L/Hipp_R -- ver utils/masking.py); nao "
                          "precisam estar tambem em --roi-tracts (sao carregados "
                          "automaticamente se so' aparecerem aqui). Ex.: "
                          "--roi-pair FX_Hipp_R=FX+Hipp_R --roi-pair FX_Hipp_L=FX+Hipp_L")
    ap.add_argument("--roi-dilate", action="append", default=[],
                     metavar="NOME=N",
                     help="ADITIVO, repetivel (ver addendum secao 26.1): dilata a mascara "
                          "de ROI 'NOME' em N voxels (scipy.ndimage.binary_dilation, "
                          "iterations=N, depois re-intersectada com a mascara de cerebro do "
                          "sujeito) ANTES de gravar o nifti usado pelo tckedit -include -- "
                          "so' afeta o(s) nome(s) listados, nenhum outro. Motivado por "
                          "mascaras de SEGMENTACAO (find_seg_roi_mask, ex. Hipp_L/Hipp_R): "
                          "sao substancia CINZENTA, e a fimbria do fornix (substancia "
                          "branca fina que corre colada na superficie do hipocampo antes de "
                          "penetrar nele) pode nao cair dentro dos voxels rotulados como "
                          "'hipocampo' por um erro de registro de 1-2 voxels -- sem dilatar, "
                          "--roi-pair FX+Hipp_R descartava conexao fornix-hipocampo GENUINA "
                          "junto com a contaminacao (ground truth caiu de 567 streamlines "
                          "em FX sozinho pra so' 28 em FX_Hipp_R, queda grande demais pra "
                          "ser so' remocao de stria terminalis). NAO recomendado para os "
                          "tratos LATERALIZADOS do atlas JHU (CGC_L/CGC_R/CGH_L/CGH_R/"
                          "UF_L/UF_R, ver secao 25.3): dilatar em direcao a linha media "
                          "reintroduz o mesmo vazamento de fibra comissural que a separacao "
                          "por lado corrigiu. Ex.: --roi-dilate Hipp_R=2 --roi-dilate Hipp_L=2")
    ap.add_argument("--algorithm", default="SD_STREAM",
                     help="Algoritmo do tckgen (default SD_STREAM = deterministico; "
                          "'iFOD2' = probabilistico, o default do proprio MRtrix se esta "
                          "flag nao existisse -- ver docstring do modulo).")
    ap.add_argument("--nthreads", type=int, default=None,
                     help="'-nthreads N' pros comandos MRtrix (dwi2response/dwi2fod/tckgen/"
                          "tckedit/tckmap -- todos CPU, MRtrix3 nao tem nenhum passo "
                          "acelerado por GPU). Default: usa $SLURM_CPUS_PER_TASK se o job "
                          "estiver rodando via sbatch (ver slurm/08_downstream_tractography.sh), "
                          "senao deixa o MRtrix auto-detectar (comportamento de sempre).")
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subjects", default=None,
                     help="ADITIVO: lista separada por virgula de 'tag' de sujeito (subject, "
                          "ou subject_session se houver sessao) para rodar em sujeitos "
                          "ESPECIFICOS em vez de todo o --split -- mesma convencao de "
                          "--subjects em scripts/11_peak_confusion_by_roi.py/"
                          "poc_csd_direction_count.py. Util pra testar rapido num sujeito so "
                          "antes de rodar o dataset inteiro (tratografia e' cara). Ex.: "
                          "--subjects 20170417094841_802780_20170417094841_802780")
    ap.add_argument("--n-streamlines", type=int, default=200_000)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    args = ap.parse_args()

    if args.subsampled_only and args.triplets_dir is None:
        sys.exit("--subsampled-only precisa de --triplets-dir")

    nthreads = args.nthreads
    if nthreads is None:
        env_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        nthreads = int(env_cpus) if env_cpus else None
    if nthreads:
        print(f"nthreads={nthreads} (via --nthreads ou $SLURM_CPUS_PER_TASK) -- "
              f"passado como -nthreads pros comandos MRtrix que suportam")
    else:
        print("nthreads nao definido -- deixando o MRtrix auto-detectar (sem $SLURM_CPUS_PER_TASK "
              "no ambiente e sem --nthreads explicito)")

    extra_methods = []
    for spec in args.extra_method:
        name, _, path = spec.partition("=")
        if not name or not path:
            sys.exit(f"--extra-method mal formado: {spec!r} (esperado NOME=DIR)")
        extra_methods.append((name, path))

    roi_tracts = [t.strip() for t in args.roi_tracts.split(",")] if args.roi_tracts else []

    roi_pairs = []  # [(pair_name, roi_a, roi_b), ...]
    for spec in args.roi_pair:
        pair_name, _, rest = spec.partition("=")
        roi_a, _, roi_b = rest.partition("+")
        if not pair_name or not roi_a or not roi_b:
            sys.exit(f"--roi-pair mal formado: {spec!r} (esperado NOME=ROI_A+ROI_B)")
        roi_pairs.append((pair_name, roi_a, roi_b))

    # uniao de todos os nomes de ROI genuinamente precisados: os de
    # --roi-tracts mais qualquer ROI_A/ROI_B referenciada em --roi-pair que
    # nao esteja ja' em --roi-tracts (carregada automaticamente so' pra
    # servir de insumo do par, sem virar uma linha "roi=" propria no CSV).
    roi_names_needed = list(roi_tracts)
    for _pair_name, roi_a, roi_b in roi_pairs:
        for name in (roi_a, roi_b):
            if name not in roi_names_needed:
                roi_names_needed.append(name)

    roi_dilate = {}  # {nome: n_iteracoes}
    for spec in args.roi_dilate:
        name, _, n_str = spec.partition("=")
        if not name or not n_str.strip():
            sys.exit(f"--roi-dilate mal formado: {spec!r} (esperado NOME=N)")
        try:
            n = int(n_str)
        except ValueError:
            sys.exit(f"--roi-dilate mal formado: {spec!r} (N precisa ser inteiro)")
        if n <= 0:
            sys.exit(f"--roi-dilate mal formado: {spec!r} (N precisa ser > 0)")
        roi_dilate[name] = n
        if name not in roi_names_needed:
            # dilatar uma ROI so' faz sentido se ela for de fato usada em
            # algum lugar (--roi-tracts ou --roi-pair) -- senao e' um nome
            # solto que nunca vira filtro nenhum.
            print(f"[aviso] --roi-dilate {spec!r}: '{name}' nao aparece em "
                  f"--roi-tracts nem em --roi-pair -- essa dilatacao nunca vai "
                  f"ser usada.", flush=True)

    def _tag_of(e):
        return e.subject if not e.session else f"{e.subject}_{e.session}"

    entries = [e for e in load_manifest(args.manifest) if e.split == args.split]
    if args.subjects:
        wanted = {t.strip() for t in args.subjects.split(",") if t.strip()}
        found = {_tag_of(e) for e in entries}
        missing = wanted - found
        if missing:
            print(f"[aviso] --subjects pediu {sorted(missing)}, mas nao encontrei no split "
                  f"{args.split!r} do manifesto.", flush=True)
        entries = [e for e in entries if _tag_of(e) in wanted]
        if not entries:
            sys.exit(f"Nenhum dos sujeitos pedidos em --subjects foi encontrado no split "
                      f"{args.split!r} -- nada a fazer.")

    out_dir = Path(args.out_dir)
    rows = []

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
        gt_data, affine, header = load_dwi(e.dwi_path)
        b0_mean = gt_data[..., bvals < 50].mean(axis=-1)
        mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)

        # (volume, bvals, bvecs) por metodo -- ground_truth e reconstrucoes
        # de volume completo usam os MESMOS bvals/bvecs de entrada (so o
        # CONTEUDO dos alvos held-out muda); o piso nativo usa um
        # bvals/bvecs genuinamente menor (ver native_floor_volume).
        variants = {"ground_truth": (gt_data, bvals, bvecs)}
        methods_to_try = [("baseline_sh", args.baseline_dir), ("rcae", args.rcae_dir)] + extra_methods
        for method, recon_dir in methods_to_try:
            if recon_dir is None:
                continue
            full = build_full_volume(gt_data, recon_dir, tag, args.shell_b, args.n_level)
            if full is not None:
                variants[method] = (full, bvals, bvecs)
            else:
                print(f"[aviso] sem reconstrucao '{method}' para {tag}, pulando esse metodo")

        if args.subsampled_only:
            trip_path = Path(args.triplets_dir) / f"{tag}_rrin_triplets.npz"
            trip_key = f"{args.shell_b}__{args.n_level}__target"
            if not trip_path.exists() or trip_key not in np.load(trip_path).files:
                print(f"[aviso] {tag}: sem trincas para --subsampled-only, pulando esse metodo")
            else:
                target_idx = np.load(trip_path)[trip_key]
                floor = native_floor_volume(gt_data, bvals, bvecs, args.shell_b,
                                             args.shell_tol, target_idx)
                if floor is not None:
                    variants["subsampled_only"] = floor

        # ROIs (whole_mask sempre; tratos pedidos, se houver, adicionais)
        rois = {"whole_mask": mask}
        if roi_names_needed:
            rois.update(load_roi_masks(e.dwi_path, roi_names_needed, base_mask=mask))
        if roi_dilate:
            from scipy import ndimage
            for name, n in roi_dilate.items():
                if name not in rois:
                    continue  # ja avisado em load_roi_masks se a mascara nao existir
                n_before = int(rois[name].sum())
                dilated = ndimage.binary_dilation(rois[name], iterations=n)
                # re-intersecta com a mascara de cerebro do sujeito -- dilatar
                # nao deve "crescer" a ROI pra fora do cerebro.
                rois[name] = dilated & mask.astype(bool)
                n_after = int(rois[name].sum())
                print(f"[info] {tag}: ROI '{name}' dilatada em {n} voxel(s) "
                      f"({n_before} -> {n_after} voxels, apos re-intersectar com "
                      f"a mascara de cerebro)", flush=True)
        roi_mask_paths: dict[str, Path] = {}

        tracks = {}
        mask_paths = {}
        densities_whole = {}
        for method, (vol, m_bvals, m_bvecs) in variants.items():
            try:
                wd = out_dir / "mrtrix_tmp" / tag / method
                tck, mask_path, density = tractography_pipeline(
                    wd, vol, affine, m_bvals, m_bvecs, mask, args.n_streamlines,
                    algorithm=args.algorithm, nthreads=nthreads)
                tracks[method] = tck
                mask_paths[method] = mask_path
                densities_whole[method] = density
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"[aviso] tratografia falhou para {tag}/{method}: {exc}. "
                      f"Confira se MRtrix3 esta instalado e no PATH.")

        if "ground_truth" not in tracks:
            print(f"[aviso] {tag}: ground_truth falhou, pulando sujeito inteiro")
            continue

        # grava as mascaras de ROI como nifti (uma vez por sujeito, reaproveitada
        # em todos os metodos -- mesmo affine/grid que o resto do sujeito).
        if roi_names_needed:
            import nibabel as nib
            roi_dir = out_dir / "mrtrix_tmp" / tag / "_rois"
            roi_dir.mkdir(parents=True, exist_ok=True)
            for roi_name, roi_mask in rois.items():
                if roi_name == "whole_mask":
                    continue
                p = roi_dir / f"{roi_name}.nii.gz"
                nib.save(nib.Nifti1Image(roi_mask.astype(np.uint8), affine), p)
                roi_mask_paths[roi_name] = p

        for roi_name in ["whole_mask"] + [t for t in roi_tracts if t in rois]:
            if roi_name == "whole_mask":
                # comportamento de sempre: Dice de densidade whole-brain,
                # sem filtrar streamline nenhuma.
                gt_density = densities_whole["ground_truth"]
                for method, tck in tracks.items():
                    if method == "ground_truth":
                        continue
                    dice = dice_from_density(densities_whole[method], gt_density)
                    n_stream = tckinfo_count(tck)
                    try:
                        mean_len = tckstats_mean_length(tck)
                    except subprocess.CalledProcessError:
                        mean_len = float("nan")
                    rows.append({"subject": e.subject, "tag": tag, "method": method,
                                 "shell": args.shell_b, "n_level": args.n_level,
                                 "roi": roi_name, "n_streamlines": n_stream,
                                 "mean_length_mm": mean_len,
                                 "dice_streamline_density": dice})
                continue

            # tratos especificos: filtra por ROI (tckedit -include) POR CIMA
            # do track whole-brain ja gerado, um tckedit por metodo/trato.
            try:
                roi_mask_path = roi_mask_paths[roi_name]
            except KeyError:
                continue
            gt_wd = out_dir / "mrtrix_tmp" / tag / "ground_truth"
            try:
                n_gt, len_gt, density_gt_roi, _ = roi_metrics_for_method(
                    tracks["ground_truth"], mask_paths["ground_truth"], roi_name,
                    [roi_mask_path], gt_wd, gt_density_roi=None, nthreads=nthreads)
            except subprocess.CalledProcessError as exc:
                print(f"[aviso] {tag}/ground_truth/{roi_name}: filtro de trato falhou "
                      f"({exc}) -- pulando trato inteiro para este sujeito")
                continue
            rows.append({"subject": e.subject, "tag": tag, "method": "ground_truth",
                         "shell": args.shell_b, "n_level": args.n_level, "roi": roi_name,
                         "n_streamlines": n_gt, "mean_length_mm": len_gt,
                         "dice_streamline_density": np.nan})

            for method, tck in tracks.items():
                if method == "ground_truth":
                    continue
                wd = out_dir / "mrtrix_tmp" / tag / method
                try:
                    n_stream, mean_len, _density, dice = roi_metrics_for_method(
                        tck, mask_paths[method], roi_name, [roi_mask_path], wd,
                        gt_density_roi=density_gt_roi, nthreads=nthreads)
                except subprocess.CalledProcessError as exc:
                    print(f"[aviso] {tag}/{method}/{roi_name}: filtro de trato falhou "
                          f"({exc}) -- gravando fit_failed para este metodo/trato")
                    rows.append({"subject": e.subject, "tag": tag, "method": method,
                                 "shell": args.shell_b, "n_level": args.n_level,
                                 "roi": roi_name, "n_streamlines": np.nan,
                                 "mean_length_mm": np.nan, "dice_streamline_density": np.nan})
                    continue
                rows.append({"subject": e.subject, "tag": tag, "method": method,
                             "shell": args.shell_b, "n_level": args.n_level, "roi": roi_name,
                             "n_streamlines": n_stream, "mean_length_mm": mean_len,
                             "dice_streamline_density": dice})

        # pares de ROI (--roi-pair, AND das duas -- ver docstring do modulo,
        # secao "FILTRO DE DUAS ROIS", e addendum secao 25.2): mesmo padrao
        # do loop de trato unico acima (ground_truth primeiro pra pegar a
        # densidade de referencia, depois cada metodo), so' que passando as
        # DUAS mascaras da ROI pro filtro em vez de uma so'.
        for pair_name, roi_a, roi_b in roi_pairs:
            try:
                mask_a = roi_mask_paths[roi_a]
                mask_b = roi_mask_paths[roi_b]
            except KeyError as exc:
                print(f"[aviso] {tag}/--roi-pair {pair_name}: ROI {exc} nao disponivel "
                      f"(mascara ausente/vazia para este sujeito) -- pulando este par")
                continue
            gt_wd = out_dir / "mrtrix_tmp" / tag / "ground_truth"
            try:
                n_gt, len_gt, density_gt_pair, _ = roi_metrics_for_method(
                    tracks["ground_truth"], mask_paths["ground_truth"], pair_name,
                    [mask_a, mask_b], gt_wd, gt_density_roi=None, nthreads=nthreads)
            except subprocess.CalledProcessError as exc:
                print(f"[aviso] {tag}/ground_truth/--roi-pair {pair_name}: filtro falhou "
                      f"({exc}) -- pulando este par para este sujeito")
                continue
            rows.append({"subject": e.subject, "tag": tag, "method": "ground_truth",
                         "shell": args.shell_b, "n_level": args.n_level, "roi": pair_name,
                         "n_streamlines": n_gt, "mean_length_mm": len_gt,
                         "dice_streamline_density": np.nan})

            for method, tck in tracks.items():
                if method == "ground_truth":
                    continue
                wd = out_dir / "mrtrix_tmp" / tag / method
                try:
                    n_stream, mean_len, _density, dice = roi_metrics_for_method(
                        tck, mask_paths[method], pair_name, [mask_a, mask_b], wd,
                        gt_density_roi=density_gt_pair, nthreads=nthreads)
                except subprocess.CalledProcessError as exc:
                    print(f"[aviso] {tag}/{method}/--roi-pair {pair_name}: filtro falhou "
                          f"({exc}) -- gravando fit_failed para este metodo/par")
                    rows.append({"subject": e.subject, "tag": tag, "method": method,
                                 "shell": args.shell_b, "n_level": args.n_level,
                                 "roi": pair_name, "n_streamlines": np.nan,
                                 "mean_length_mm": np.nan, "dice_streamline_density": np.nan})
                    continue
                rows.append({"subject": e.subject, "tag": tag, "method": method,
                             "shell": args.shell_b, "n_level": args.n_level, "roi": pair_name,
                             "n_streamlines": n_stream, "mean_length_mm": mean_len,
                             "dice_streamline_density": dice})

        print(f"{tag}: tratografia comparada para {list(tracks)} (rois: {list(rois)}, "
              f"roi_pairs: {[p[0] for p in roi_pairs]})")

    if not rows:
        sys.exit("Nenhum resultado de tratografia (confira instalacao do MRtrix3 e os diretorios)")

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"tractography_metrics_shell{int(args.shell_b)}_n{args.n_level}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Metricas de tratografia salvas em", out_csv)
    print(df.groupby(["roi", "method"])[["n_streamlines", "mean_length_mm",
                                          "dice_streamline_density"]].mean())


if __name__ == "__main__":
    main()