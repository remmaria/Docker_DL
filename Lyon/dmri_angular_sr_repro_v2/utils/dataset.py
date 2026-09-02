"""
Dataset PyTorch para treino do RCAE.

Reescrito para reproduzir mais fielmente o pipeline de treino do paper
(Lyon et al. 2022 / github.com/m-lyon/dMRI-RCNN), com base na leitura do
codigo oficial (dmri_rcnn/core/processing/training/{patcher,scaler,
shell_reorder}.py):

  1) split entrada/alvo do q-space RE-AMOSTRADO a cada exemplo de TREINO
     (nao fixo por sujeito) -- ver `_dynamic_split` / ShellReorder no
     paper. Validacao continua com o split fixo pre-computado (o paper
     tambem desativa o reorder pra validacao: `if not validation`).
  2) normalizacao por percentil DENTRO da shell (dmri / xmax, xmax =
     percentil 99 dos valores dentro da mascara), substituindo a divisao
     por b0 usada antes.
  3) patches em grade DETERMINISTICA, nao-sobreposta, com zero-padding nas
     bordas (cobre o volume inteiro, mantem qualquer tile com >=1 voxel de
     mascara) -- substitui o crop aleatorio + rejeicao por min_mask_frac.
  4) N_out (q_out) fixo e configuravel (default 10), em vez de "todas as
     direcoes restantes da shell".

Mantem tudo em memoria por sujeito de forma preguicosa (cache LRU), o que
assume que os volumes cabem em RAM -- razoavel para dMRI tipico (algumas
centenas de MB por sujeito).

Requer torch (nao disponivel neste ambiente de desenvolvimento -- revisado
manualmente, testar no cluster).

O que NAO foi replicado do paper (simplificacoes deliberadas, fora do
escopo desta rodada de ajustes):
  - a arquitetura em si (blocos Inception multi-kernel-size, canais bem
    maiores) -- ver model/rcae.py para o que FOI ajustado la (reinjecao do
    embedding em cada bloco do decoder, swish, instance/batch norm).
  - o canal de b0 medio como entrada extra (o paper usa `b0` como uma das
    shells de entrada quando aplicavel); aqui continuamos treinando
    shell-a-shell sem reinjetar b0 explicitamente.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .gradients import load_bval_bvec, load_dwi, split_shells, farthest_point_sampling
from .masking import load_or_build_mask, find_mask_path, simple_brain_mask


def _resolve_shell_key(shells: dict, shell_b: float, tol: float) -> float:
    """Acha a chave de `shells` (dict de split_shells) mais proxima de
    `shell_b`, ignorando b0 (chave 0). Levanta erro se nao achar nada
    dentro de `tol`."""
    best_key, best_diff = None, None
    for k in shells:
        if k == 0:
            continue
        diff = abs(k - shell_b)
        if best_diff is None or diff < best_diff:
            best_key, best_diff = k, diff
    if best_key is None or best_diff > tol:
        raise RuntimeError(f"shell {shell_b} nao encontrada (tol={tol}), shells disponiveis: {list(shells)}")
    return best_key


def _lightweight_subject_mask(entry, mask_suffix: str, shell_tol: float) -> np.ndarray:
    """Mascara de cerebro pra um sujeito SEM carregar o volume 4D inteiro --
    usada so no __init__ pra montar a grade de tiles determinicos (precisa
    do shape/mascara de cada sujeito antes de comecar o treino de verdade,
    mas carregar todos os 4D completos de uma vez so pra isso derrotaria o
    proposito do cache LRU preguicoso).

    1) tenta achar uma mascara ja pronta em disco (find_mask_path) -- caso
       comum e barato (so le o arquivo de mascara, nao o dwi).
    2) senao, carrega so os volumes b0 via `dataobj` (memmap, sem trazer o
       4D inteiro pra RAM) pra montar uma mascara simples por threshold.

    Nota: se a mascara vier do caminho (2), `_load_subject` (mais abaixo)
    vai recalcula-la de novo (dessa vez carregando o b0 "de verdade" via
    get_fdata, nao dataobj) -- redundante mas barato (poucos volumes) e
    deterministico, o resultado e o mesmo.
    """
    mask_path = find_mask_path(entry.dwi_path, mask_suffix)
    if mask_path is not None:
        import nibabel as nib
        return nib.load(str(mask_path)).get_fdata() > 0.5

    import nibabel as nib
    bvals, _ = load_bval_bvec(entry.bval_path, entry.bvec_path)
    shells = split_shells(bvals, tol=shell_tol)
    b0_idx = shells[0]
    img = nib.load(entry.dwi_path)
    proxy = img.dataobj
    b0_vols = np.stack([np.asarray(proxy[..., int(i)], dtype=np.float32) for i in b0_idx], axis=-1)
    b0_mean = b0_vols.mean(axis=-1)
    return simple_brain_mask(b0_mean)


def _tile_origins(mask: np.ndarray, ps: int) -> list:
    """Grade de patches NAO-SOBREPOSTA cobrindo o volume inteiro (origem em
    multiplos de `ps`, ultimo tile de cada eixo pode ficar parcialmente
    fora do volume -- zero-padding e aplicado na hora de extrair o patch,
    ver `DWIPatchDataset._extract`). Mantem so tiles com pelo menos 1 voxel
    de mascara (senao a maior parte da grade seria fundo puro, sem
    sinal nenhum pra aprender).

    Devolve lista de (origin, coverage) -- coverage = fracao de voxels
    DENTRO da mascara, calculada sobre o PATCH INTEIRO ja com zero-padding
    (ps**3 no denominador), NAO so sobre a regiao recortada real (ex-ox)*
    (ey-oy)*(ez-oz). Isso importa muito pra tiles de BORDA: um tile que so
    tem, digamos, 2 fatias reais antes de bater no limite do volume (o
    resto vira padding de zero em `_extract`) podia ter essas 2 fatias
    100% dentro da mascara e ainda assim reportar coverage=1.0 com
    `tile_mask.mean()` (que divide pelo tamanho da regiao RECORTADA, so
    2*ps*ps voxels) -- passando folgado em qualquer filtro de
    --min-tile-coverage, mesmo que o patch de verdade entregue a rede
    (ps,ps,ps, com 8 das 10 fatias zeradas por padding) seja quase todo
    fundo. Dividindo por ps**3 (o tamanho real do patch pos-padding), esse
    mesmo tile cai pra coverage=0.2*(2/10)=0.02 -- corretamente abaixo do
    filtro. Bug real encontrado via inspecao de um snapshot de debug com
    input/target/pred TODOS zerados (step 4800, epoca 1) apesar do
    --min-tile-coverage 0.15 estar ativo -- confirma que esse era
    exatamente o mecanismo (tile de borda com coverage superestimado pela
    formula antiga). Tiles interiores (ex-ox==ey-oy==ez-oz==ps, sem
    padding) tem o mesmo valor de coverage nas duas formulas -- so tiles de
    borda mudam.

    O tile (0,0,0) (o PRIMEIRO da grade, em ordem crescente de eixo) costuma
    cair bem no canto da imagem, quase sempre fundo puro (o cerebro
    raramente comeca exatamente na origem do volume) -- so passa no filtro
    ">=1 voxel de mascara" por pouco, com cobertura pertinho de zero. Isso
    importa porque o antigo codigo em scripts/04_train_rcae.py pegava
    sempre `val_ds[0]` (== o tile 0 do primeiro sujeito) como "o patch
    fixo" de debug -- ou seja, seguidamente escolhia um patch quase sem
    sinal nenhum pra acompanhar a evolucao do treino, o que inutilizava
    essa serie de snapshots. `coverage` aqui e o que permite ao script de
    treino escolher um tile de verdade representativo (maior cobertura de
    cerebro) em vez do indice 0 cego."""
    shape = mask.shape
    patch_volume = ps ** 3
    origins = []
    for ox in range(0, shape[0], ps):
        ex = min(ox + ps, shape[0])
        for oy in range(0, shape[1], ps):
            ey = min(oy + ps, shape[1])
            for oz in range(0, shape[2], ps):
                ez = min(oz + ps, shape[2])
                tile_mask = mask[ox:ex, oy:ey, oz:ez]
                if tile_mask.any():
                    coverage = float(tile_mask.sum()) / patch_volume
                    origins.append(((ox, oy, oz), coverage))
    return origins


class DWIPatchDataset(Dataset):
    def __init__(self, entries, scheme_dir: str, shell_b: float, n_level: int,
                 patch_size: int = 10, q_out: int = 10, training: bool = False,
                 mask_suffix: str = "_mask3d.nii.gz", shell_tol: float = 100.0,
                 seed: int = 0, max_cached_subjects: int = 2,
                 min_tile_coverage: float = 0.0):
        """
        patch_size: default 10 (era 24) -- patch_shape=(10,10,10) no paper.
        q_out: quantas direcoes-alvo por exemplo (default 10, fixo -- no
            paper e sempre 10, nao "todas as direcoes restantes da shell").
        training: quando True, o split entrada/alvo do q-space e
            RE-AMOSTRADO a cada __getitem__ (ver `_dynamic_split`),
            reproduzindo o ShellReorder do paper (so ativo em treino). Em
            validacao (training=False) usa sempre o split fixo pre-computado
            em `<subject>_scheme.npz` (truncado a `q_out` direcoes-alvo),
            pra manter os snapshots/losses de validacao comparaveis entre
            epocas.
        min_tile_coverage: descarta da grade os tiles com fracao de voxels
            de mascara MENOR que isso (0 a 1). O filtro antigo de
            `_tile_origins` so exige `tile_mask.any()` (>=1 voxel de
            mascara) -- ou seja, tiles de borda do cerebro com cobertura de
            1-2% (999 dos 1000 voxels sao fundo puro, zerado explicitamente
            em `__getitem__` via `* mask_patch`) SEMPRE entravam no pool de
            treino/validacao com o mesmo peso de amostragem que um tile
            cheio de sinal. Isso e esperado que produza bastante patch
            "quase todo zero" no debug (nao e bug de mascara -- e o
            resultado natural de uma grade nao-sobreposta cobrindo o volume
            inteiro sobre um cerebro com formato irregular). Suba este
            valor (ex.: 0.1 = pelo menos 10% do tile dentro da mascara) pra
            tirar os tiles mais extremos do pool. Default 0.0 mantem o
            comportamento antigo (nenhum tile descartado por cobertura).
        """
        self.entries = entries
        self.scheme_dir = Path(scheme_dir)
        self.shell_b = shell_b
        self.n_level = n_level
        self.patch_size = patch_size
        self.q_out = q_out
        self.training = training
        self.mask_suffix = mask_suffix
        self.shell_tol = shell_tol
        self.min_tile_coverage = min_tile_coverage
        # LRU limitado -- ver comentario original: cada worker do DataLoader
        # e um processo separado com seu proprio cache; sem limite, ao
        # longo de uma epoca cada worker acaba guardando o volume 4D
        # inteiro de varios/todos os sujeitos, multiplicando RAM por
        # --num-workers e estourando --mem (OOM kill).
        self.max_cached_subjects = max_cached_subjects
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        # ATENCAO (mesmo motivo de antes): com --num-workers > 0, cada
        # worker herda uma copia identica deste RNG no fork() -- reseedado
        # por worker via worker_init_fn (ver mais abaixo), senao o split
        # dinamico de treino (e a escolha de tiles, se algum dia virar
        # amostragem) ficaria em lockstep entre workers.
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # filtra sujeitos que realmente tem esse (shell, nivel) no esquema
        # pre-computado -- necessario mesmo em treino: usado como split
        # FIXO de validacao, e como fallback do split dinamico quando a
        # shell tem poucas direcoes disponiveis (ver _dynamic_split).
        self.usable = []
        for e in entries:
            tag = e.subject if not e.session else f"{e.subject}_{e.session}"
            scheme_path = self.scheme_dir / f"{tag}_scheme.npz"
            if not scheme_path.exists():
                continue
            scheme = np.load(scheme_path)
            key = f"{shell_b}__{n_level}"
            if f"{key}__input" in scheme.files:
                self.usable.append((e, tag))
        if not self.usable:
            raise RuntimeError(
                f"Nenhum sujeito tem shell={shell_b} nivel={n_level} no esquema em {scheme_dir}"
            )

        # grade de tiles deterministica por sujeito, pre-computada aqui
        # (mascara leve, sem carregar o 4D inteiro -- ver
        # _lightweight_subject_mask) -- substitui o antigo
        # patches_per_subject fixo por um numero de tiles que varia por
        # sujeito (proporcional ao volume/mascara real dele).
        self.tile_index = []  # list[(subj_idx_em_usable, (ox,oy,oz))]
        self.tile_coverage = []  # list[float] -- paralela a tile_index, fracao de mascara por tile
        n_seen = 0
        for si, (e, _tag) in enumerate(self.usable):
            mask = _lightweight_subject_mask(e, self.mask_suffix, self.shell_tol)
            origins = _tile_origins(mask, self.patch_size)
            n_seen += len(origins)
            for o, coverage in origins:
                if coverage < self.min_tile_coverage:
                    continue
                self.tile_index.append((si, o))
                self.tile_coverage.append(coverage)
        if not self.tile_index:
            raise RuntimeError("Nenhum tile com voxel de mascara encontrado em nenhum sujeito usavel "
                                "(ou --min-tile-coverage esta alto demais pro seu dado)")
        if n_seen:
            cov_arr = np.asarray(self.tile_coverage, dtype=np.float32)
            tag = "treino" if self.training else "val"
            print(f"[dataset:{tag}] tiles: {len(self.tile_index)}/{n_seen} mantidos "
                  f"(min_tile_coverage={self.min_tile_coverage}); cobertura mantida "
                  f"p10={np.percentile(cov_arr, 10):.3f} mediana={np.median(cov_arr):.3f} "
                  f"p90={np.percentile(cov_arr, 90):.3f}", flush=True)

    def __len__(self):
        return len(self.tile_index)

    def _load_subject(self, tag: str, entry):
        if tag in self._cache:
            self._cache.move_to_end(tag)  # marca como usado mais recentemente
            return self._cache[tag]

        # SEM print de "carregando sujeito"/"carregado" aqui de proposito
        # (ver historico -- virava ruido constante no .out). Erro de
        # leitura real (arquivo ausente/corrompido) ainda aparece
        # normalmente via excecao.
        bvals, bvecs = load_bval_bvec(entry.bval_path, entry.bvec_path)
        data, affine, header = load_dwi(entry.dwi_path)
        shells = split_shells(bvals, tol=self.shell_tol)
        b0_idx = shells[0]
        b0_mean = data[..., b0_idx].mean(axis=-1)
        mask = load_or_build_mask(entry.dwi_path, b0_mean, mask_suffix=self.mask_suffix)

        shell_key = _resolve_shell_key(shells, self.shell_b, self.shell_tol)
        shell_idxs = np.asarray(shells[shell_key], dtype=int)

        # normalizacao por percentil DENTRO da mascara e da shell (substitui
        # a divisao por b0 -- ver NormDataScaler no paper: xmax = percentil
        # 99 dos valores validos, dmri_norm = dmri / xmax). Nao ha divisao
        # por voxel nenhuma aqui, e um unico escalar por sujeito/shell.
        mask_bool = mask.astype(bool)
        shell_vals = data[..., shell_idxs][mask_bool]
        xmax = float(np.percentile(shell_vals, 99)) if shell_vals.size else 1.0
        if not np.isfinite(xmax) or xmax <= 0:
            xmax = 1.0

        scheme_path = self.scheme_dir / f"{tag}_scheme.npz"
        scheme = np.load(scheme_path)
        key = f"{self.shell_b}__{self.n_level}"
        input_idx = scheme[f"{key}__input"]
        target_idx = scheme[f"{key}__target"]
        if len(target_idx) > self.q_out:
            # trunca ao q_out fixo -- o paper usa sempre N_out=10, nao
            # "todas as direcoes restantes da shell" (que e o que o
            # esquema pre-computado guarda).
            target_idx = target_idx[: self.q_out]

        cached = {
            "dwi": data.astype(np.float32),
            "mask": mask,
            "bvecs": bvecs.astype(np.float32),
            "bvals": bvals.astype(np.float32),
            "shell_idxs": shell_idxs,
            "xmax": xmax,
            "input_idx": input_idx,
            "target_idx": target_idx,
        }
        self._cache[tag] = cached
        while len(self._cache) > self.max_cached_subjects:
            self._cache.popitem(last=False)  # descarta o menos usado recentemente
        return cached

    def _dynamic_split(self, d: dict):
        """Re-amostra o split entrada/alvo do q-space PARA ESTE EXEMPLO,
        reproduzindo o ShellReorder do paper: escolhe um ponto de partida
        aleatorio e usa farthest-point sampling (sort=False, ou seja, na
        ORDEM DE SELECAO) pra formar um grupo de `n_level + q_out`
        direcoes maximamente dispersas; os primeiros `n_level` viram
        entrada, os `q_out` seguintes viram alvo. So chamado quando
        self.training=True -- em validacao usamos sempre o split fixo
        (ver _load_subject), pra nao misturar "mudou o split" com "o
        modelo aprendeu mais" nos snapshots/losses de validacao.
        """
        shell_idxs = d["shell_idxs"]
        n_avail = len(shell_idxs)
        n_take = min(self.n_level + self.q_out, n_avail)
        if n_take <= self.n_level:
            # shell pequena demais pra sobrar alguma direcao-alvo depois do
            # split dinamico -- cai de volta pro split fixo pre-computado
            # (que ja leva isso em conta na hora de gerar o esquema).
            return d["input_idx"], d["target_idx"]

        seed_idx = int(self._rng.integers(0, n_avail))
        local_bvecs = d["bvecs"][shell_idxs]
        order = farthest_point_sampling(local_bvecs, n_take, seed_idx=seed_idx, sort=False)
        input_local = order[: self.n_level]
        target_local = order[self.n_level:n_take]
        return shell_idxs[input_local], shell_idxs[target_local]

    def _extract(self, vol: np.ndarray, ox: int, oy: int, oz: int) -> np.ndarray:
        """Extrai um bloco (patch_size, patch_size, patch_size, ...) a
        partir da origem (ox,oy,oz), com zero-padding se o tile encostar na
        borda do volume (grade deterministica nao-sobreposta, o ultimo
        tile de cada eixo normalmente extrapola o shape -- ver
        _tile_origins)."""
        ps = self.patch_size
        shape = vol.shape
        ex, ey, ez = min(ox + ps, shape[0]), min(oy + ps, shape[1]), min(oz + ps, shape[2])
        sub = vol[ox:ex, oy:ey, oz:ez, ...]
        pad_x, pad_y, pad_z = ps - (ex - ox), ps - (ey - oy), ps - (ez - oz)
        if pad_x or pad_y or pad_z:
            pad_width = [(0, pad_x), (0, pad_y), (0, pad_z)] + [(0, 0)] * (sub.ndim - 3)
            sub = np.pad(sub, pad_width, mode="constant")
        return sub

    def __getitem__(self, idx):
        si, (ox, oy, oz) = self.tile_index[idx]
        entry, tag = self.usable[si]
        d = self._load_subject(tag, entry)

        if self.training:
            input_idx, target_idx = self._dynamic_split(d)
        else:
            input_idx, target_idx = d["input_idx"], d["target_idx"]

        mask_patch = self._extract(d["mask"].astype(np.float32), ox, oy, oz)[..., None]  # (ps,ps,ps,1)

        # fora da mascara zeramos explicitamente (ver comentario historico:
        # sem isso, voxels de fundo entravam crus no treino e causavam
        # outliers extremos no batch_log.csv).
        input_raw = self._extract(d["dwi"][..., input_idx], ox, oy, oz)
        target_raw = self._extract(d["dwi"][..., target_idx], ox, oy, oz)
        xmax = d["xmax"]
        input_patch = (input_raw / xmax) * mask_patch   # (ps,ps,ps,n_in)
        target_patch = (target_raw / xmax) * mask_patch  # (ps,ps,ps,n_out)

        # -> (n_dirs, 1, ps, ps, ps)
        input_vols = np.moveaxis(input_patch, -1, 0)[:, None].astype(np.float32)
        target_vols = np.moveaxis(target_patch, -1, 0)[:, None].astype(np.float32)

        input_bvecs = d["bvecs"][input_idx].astype(np.float32)
        input_bvals = d["bvals"][input_idx].astype(np.float32)
        target_bvecs = d["bvecs"][target_idx].astype(np.float32)
        target_bvals = d["bvals"][target_idx].astype(np.float32)

        return {
            "input_vols": torch.from_numpy(input_vols),
            "input_bvecs": torch.from_numpy(input_bvecs),
            "input_bvals": torch.from_numpy(input_bvals),
            "target_vols": torch.from_numpy(target_vols),
            "target_bvecs": torch.from_numpy(target_bvecs),
            "target_bvals": torch.from_numpy(target_bvals),
            # so pra rastreabilidade (ex.: batch_log.csv) -- identificar qual
            # sujeito gerou um batch com outliers/loss estranha.
            "subject_tag": tag,
        }


def collate_variable_targets(batch: list[dict]) -> dict:
    """Collate customizado para DWIPatchDataset.

    `input_*` tem sempre o mesmo N_in (== n_level) e pode ser empilhado
    normalmente. `target_*` tem N_out <= q_out (pode ser menor em shells
    com poucas direcoes disponiveis -- ver _dynamic_split/_load_subject) e
    VARIA por sujeito -- o collate padrao do PyTorch quebra (`stack
    expects each tensor to be equal size`) assim que dois sujeitos com
    N_out diferente caem no mesmo batch. Aqui fazemos padding do lado do
    target ate o maior N_out do batch (zeros) e devolvemos "target_mask"
    (B, N_out_max) para a loss ignorar as posicoes de padding -- ver
    run_epoch em scripts/04_train_rcae.py.
    """
    input_vols = torch.stack([b["input_vols"] for b in batch], dim=0)
    input_bvecs = torch.stack([b["input_bvecs"] for b in batch], dim=0)
    input_bvals = torch.stack([b["input_bvals"] for b in batch], dim=0)

    n_out_max = max(b["target_vols"].shape[0] for b in batch)
    bsz = len(batch)
    vol_shape = batch[0]["target_vols"].shape[1:]  # (1, ps, ps, ps)

    target_vols = torch.zeros((bsz, n_out_max, *vol_shape), dtype=torch.float32)
    target_bvecs = torch.zeros((bsz, n_out_max, 3), dtype=torch.float32)
    target_bvals = torch.zeros((bsz, n_out_max), dtype=torch.float32)
    target_mask = torch.zeros((bsz, n_out_max), dtype=torch.bool)

    for i, b in enumerate(batch):
        n_out = b["target_vols"].shape[0]
        target_vols[i, :n_out] = b["target_vols"]
        target_bvecs[i, :n_out] = b["target_bvecs"]
        target_bvals[i, :n_out] = b["target_bvals"]
        target_mask[i, :n_out] = True

    return {
        "input_vols": input_vols,
        "input_bvecs": input_bvecs,
        "input_bvals": input_bvals,
        "subject_tags": [b["subject_tag"] for b in batch],  # lista de str, nao empilha
        "target_vols": target_vols,
        "target_bvecs": target_bvecs,
        "target_bvals": target_bvals,
        "target_mask": target_mask,
    }


def worker_init_fn(worker_id: int) -> None:
    """Passar como `worker_init_fn=` no DataLoader (train E val) sempre que
    num_workers > 0. Resseeda o `self._rng` de CADA worker com uma seed
    unica (seed_base + worker_id) -- sem isso, todo worker herda uma copia
    identica do RNG do dataset (mesma seed, estado "recem-criado") no
    fork(), e o split dinamico de treino (_dynamic_split) ficaria em
    lockstep entre workers.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # copia do dataset especifica deste worker
    base_seed = getattr(dataset, "seed", 0)
    dataset._rng = np.random.default_rng(base_seed + 1000 * (worker_id + 1))


class SubjectGroupedSampler(torch.utils.data.Sampler):
    """Sampler pra DWIPatchDataset que embaralha por SUJEITO, nao por tile
    individual.

    Com `shuffle=True` normal do DataLoader, os tiles de um mesmo sujeito
    ficam espalhados aleatoriamente pela epoca inteira -- na pratica, quase
    todo batch pede um sujeito novo, o que anula o cache LRU
    (max_cached_subjects) e forca releitura de disco repetida.

    Aqui embaralhamos a ORDEM dos sujeitos a cada epoca, mas mantemos os
    tiles de um mesmo sujeito agrupados (so embaralhados entre si) -- assim
    um sujeito, uma vez carregado, e reaproveitado pelos proximos tiles
    dele antes de trocar. Generalizado pra numero de tiles VARIAVEL por
    sujeito (a grade deterministica de tiles nao garante mais o mesmo
    numero de patches por sujeito que o antigo `patches_per_subject` fixo
    garantia) -- agrupa direto a partir de `dataset.tile_index`.
    """

    def __init__(self, dataset: "DWIPatchDataset", seed: int = 0, freeze_order: bool = False):
        """
        freeze_order (default False, ADITIVO -- comportamento de todo
        chamador existente continua identico sem passar isso explicitamente,
        ver addendum 2026-09-02): quando True, `set_epoch` ignora o numero
        da epoca e a ORDEM dos sujeitos fica a MESMA em toda epoca (so a
        ordem DOS SUJEITOS -- o embaralhamento dos tiles DENTRO de cada
        sujeito, mais abaixo em `__iter__`, continua variando por epoca
        normalmente, ja que usa o mesmo `rng` sequencialmente).

        Motivacao (diagnostico de gargalo de dataloading no treino
        `pairflow_ssl`, ver scripts/04g_train_pairflow_ssl.py): com
        `num_workers>0`, o `DataLoader` do PyTorch despacha os batches pros
        workers em ROUND-ROBIN (worker 0,1,...,N-1,0,1,...), nao em blocos
        por sujeito -- entao um sujeito cujos tiles rendem mais batches que
        `num_workers` acaba tendo pelo menos 1 batch atendido por CADA
        worker, e cada worker tem seu PROPRIO cache LRU
        (`max_cached_subjects`) isolado dos demais (processos separados).
        Resultado: o MESMO sujeito e' recarregado do disco em ate
        `num_workers` workers diferentes por epoca -- e, se a ordem dos
        sujeitos muda a cada epoca (comportamento antigo, default), o
        mapeamento sujeito->worker tambem muda, entao nem entre epocas um
        worker consegue reaproveitar o que ja carregou. Congelar a ordem
        (freeze_order=True) NAO elimina a redundancia entre workers dentro
        de uma mesma epoca (isso exigiria particionar sujeitos por worker,
        mudanca maior, nao feita aqui), mas garante que o mapeamento
        sujeito->worker fique ESTAVEL de epoca pra epoca -- entao, depois
        da primeira epoca "fria", cada worker tende a ja ter em cache os
        MESMOS sujeitos que vai precisar de novo, reduzindo releituras de
        disco nas epocas seguintes (contanto que `max_cached_subjects` seja
        grande o suficiente pra cobrir os sujeitos que aquele worker
        especificamente revisita)."""
        groups = defaultdict(list)
        for flat_idx, (si, _origin) in enumerate(dataset.tile_index):
            groups[si].append(flat_idx)
        self.groups = list(groups.values())
        self.seed = seed
        self.freeze_order = freeze_order
        self._epoch = 0

    def set_epoch(self, epoch: int):
        # opcional: chame antes de cada epoca se quiser uma ordem diferente
        # (senao usa sempre a mesma seed, mesma ordem toda epoca -- ainda
        # assim ja resolve o problema de cache, so nao varia a ordem).
        # Com freeze_order=True, ignora `epoch` de proposito (ver docstring
        # do __init__) -- a ordem dos SUJEITOS fica sempre a mesma.
        self._epoch = 0 if self.freeze_order else epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        order = rng.permutation(len(self.groups))
        indices = []
        for gi in order:
            block = list(self.groups[gi])
            rng.shuffle(block)  # ainda ha variedade DENTRO do sujeito
            indices.extend(block)
        return iter(indices)

    def __len__(self):
        return sum(len(g) for g in self.groups)