"""
Dataset PyTorch para o PRE-TREINO AUTO-SUPERVISIONADO do fluxo bidirecional
entre pares de direcoes REAIS medidas (Etapa 1 da linha `pairflow_ssl`, ver
model/pairflow_ssl.py e scripts/04g_train_pairflow_ssl.py -- addendum
secao 20.15, ideia originada da pergunta da usuaria sobre fluxo optico
auto-supervisionado / EMA-VFI).

DIFERENCA CENTRAL pra utils/rrin_dataset.py:RRINTripletDataset: aqui NAO HA
TRINCA NENHUMA -- nenhum `--triplets-dir`, nenhum
scripts/02b_build_rrin_triplets.py precisa ter rodado antes. Cada item
sorteia DOIS indices de direcao quaisquer dentro da shell pedida (excluindo
b0), sem exigir que exista uma terceira direcao real "entre" eles dentro de
um teto de residuo -- e' exatamente essa curadoria que o pre-treino
auto-supervisionado quer evitar, pra poder usar TODOS os pares possiveis da
aquisicao (O(N^2) por sujeito) em vez de so os que sobrevivem ao teto de
residuo/M do ensemble em estrela (ver protocolo secao 10.1/19/20.12).

A loss (ver model/pairflow_ssl.py:pairflow_ssl_losses e
scripts/04g_train_pairflow_ssl.py) reconstroi cada um dos dois extremos a
partir do OUTRO via warp -- os dois volumes do PROPRIO par ja servem de
"ground truth" auto-supervisionado, sem precisar de nenhum terceiro ponto
medido.

Reaproveita `_resolve_shell_key`/`_lightweight_subject_mask`/`_tile_origins`
de utils/dataset.py (genericos, mesmo import ja feito por
utils/rrin_dataset.py) e duplica `_extract` de utils/rrin_dataset.py (mesma
logica -- ver comentario na funcao -- sem import cruzado entre modulos de
dataset pensados pra etapas/linhas diferentes, mesmo espirito das funcoes
duplicadas entre scripts numerados)."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .gradients import load_bval_bvec, load_dwi, split_shells
from .masking import load_or_build_mask
from .dataset import _resolve_shell_key, _lightweight_subject_mask, _tile_origins


def _angular_gap_deg(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Distancia angular entre duas direcoes, com simetria antipodal
    (v == -v, convencao usual de dMRI) -- mesma formula usada em varios
    pontos de utils/gradients.py (ex.: spherical_triplet_residual),
    reproduzida aqui em forma MINIMA (sem todo o aparato de
    trinca/residuo/t_frac, que nao se aplica a este dataset -- aqui nao ha
    nenhum terceiro ponto) pra nao criar import cruzado com um modulo
    pensado para outra etapa/linha."""
    a = np.asarray(v_a, dtype=float)
    b = np.asarray(v_b, dtype=float)
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    cos = abs(float(np.dot(a, b)))
    cos = min(1.0, max(-1.0, cos))
    return float(np.degrees(np.arccos(cos)))


class PairFlowSSLDataset(Dataset):
    def __init__(self, entries, shell_b: float, mask_suffix: str = "_mask3d.nii.gz",
                 shell_tol: float = 100.0, patch_size: int = 10, training: bool = False,
                 seed: int = 0, max_cached_subjects: int = 2, min_tile_coverage: float = 0.0,
                 min_pair_gap_deg: float = 5.0, max_pair_gap_deg: float | None = None,
                 max_sample_attempts: int = 20, log_worker_loads: bool = False):
        """
        min_pair_gap_deg (default 5.0): descarta (por rejeicao, ver
            `_sample_pair_idx`) pares MUITO proximos entre si -- sem isso, o
            par quase-degenerado (a quase igual a b) tem solucao de fluxo
            ~0 satisfazendo a loss fotometrica quase de graca, sem
            aprender fluxo nenhum de verdade, e pares assim dominariam boa
            parte do sorteio aleatorio (a maioria das direcoes tem varias
            vizinhas bem proximas numa amostragem tipo Jones30/Jones64).
            NAO e' o mesmo tipo de curadoria geometrica do
            `--max-residual-deg` do RRIN (que exige uma TERCEIRA direcao
            real "entre" a e b) -- aqui so filtra o par em si, sem nenhuma
            nocao de alvo/colinearidade.
        max_pair_gap_deg (default None = sem teto): opcional, so pra
            experimentar se limitar o gap maximo (aproximando do regime
            "sempre pares proximos" que VFI de video assume implicitamente)
            muda o comportamento do fluxo aprendido -- o ponto central da
            ideia (addendum secao 20.15) e' justamente treinar TAMBEM com
            pares distantes, entao o default e' NAO filtrar.
        max_sample_attempts: tentativas de rejeicao antes de desistir e
            aceitar o ultimo par sorteado mesmo fora da faixa (evita loop
            infinito/quase-infinito num sujeito com poucas direcoes na
            shell, onde a faixa [min,max] pode ser dificil de atingir por
            sorteio puro).
        """
        self.entries = entries
        self.shell_b = shell_b
        self.mask_suffix = mask_suffix
        self.shell_tol = shell_tol
        self.patch_size = patch_size
        self.training = training
        self.seed = seed
        self.max_cached_subjects = max_cached_subjects
        self.min_tile_coverage = min_tile_coverage
        self.min_pair_gap_deg = min_pair_gap_deg
        self.max_pair_gap_deg = max_pair_gap_deg
        self.max_sample_attempts = max_sample_attempts
        self.log_worker_loads = log_worker_loads
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._rng = np.random.default_rng(seed)

        # FILTRO POR DISPONIBILIDADE DA SHELL (bug corrigido 2026-09-01,
        # achado pela usuaria ao rodar num dataset multi-shell onde nem
        # todo sujeito tem a MESMA shell): ao contrario de
        # RRINTripletDataset (que so marca "usable" os sujeitos cujo
        # <tag>_rrin_triplets.npz ja TEM a chave "{shell_b}__{n_level}__target"
        # -- essa chave so existe se 02b_build_rrin_triplets.py achou
        # aquela shell pra aquele sujeito, entao sujeitos sem ela ja saem
        # filtrados de graca antes de chegar aqui), este dataset NAO tem
        # nenhuma trinca pre-calculada pra consultar -- sem este filtro
        # explicito, o primeiro sujeito da lista sem a shell pedida
        # derrubava o DataLoader inteiro com RuntimeError dentro de
        # _load_subject (_resolve_shell_key), so' no meio do treino. Aqui
        # so' lemos o .bval (leve, sem tocar no 4D) pra decidir se o
        # sujeito fica ou nao -- mesmo espirito do "so' verifica o que
        # precisa" de _lightweight_subject_mask em utils/dataset.py.
        self.usable = []
        n_skipped = 0
        for e in entries:
            tag = e.subject if not e.session else f"{e.subject}_{e.session}"
            try:
                bvals, _bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
                shells = split_shells(bvals, tol=self.shell_tol)
                shell_key = _resolve_shell_key(shells, self.shell_b, self.shell_tol)
                if shells[shell_key].size < 2:
                    raise RuntimeError(
                        f"shell {self.shell_b} tem menos de 2 direcoes ({shells[shell_key].size})")
            except RuntimeError as exc:
                n_skipped += 1
                print(f"[pairflow_ssl_dataset][aviso] {tag}: pulando (sem shell "
                      f"{self.shell_b} utilizavel -- {exc})", flush=True)
                continue
            self.usable.append((e, tag))
        if not self.usable:
            raise RuntimeError(
                f"Nenhum sujeito tem a shell {self.shell_b} (tol={self.shell_tol}) -- "
                f"confira --shell-b (shells disponiveis variam por sujeito neste manifesto, "
                f"ver avisos acima) ou --shell-tol.")
        if n_skipped:
            print(f"[pairflow_ssl_dataset] {n_skipped}/{len(entries)} sujeito(s) pulado(s) por "
                  f"nao terem a shell {self.shell_b} -- {len(self.usable)} restante(s).",
                  flush=True)

        self.tile_index = []
        n_seen = 0
        for si, (e, _tag) in enumerate(self.usable):
            mask = _lightweight_subject_mask(e, self.mask_suffix, self.shell_tol)
            origins = _tile_origins(mask, self.patch_size)
            n_seen += len(origins)
            for o, coverage in origins:
                if coverage < self.min_tile_coverage:
                    continue
                self.tile_index.append((si, o))
        if not self.tile_index:
            raise RuntimeError("Nenhum tile com voxel de mascara encontrado (ou "
                                "--min-tile-coverage alto demais)")
        tag = "treino" if self.training else "val"
        print(f"[pairflow_ssl_dataset:{tag}] {len(self.usable)} sujeitos, "
              f"{len(self.tile_index)}/{n_seen} tiles mantidos "
              f"(min_tile_coverage={self.min_tile_coverage}); pares sorteados sob demanda "
              f"(min_pair_gap_deg={self.min_pair_gap_deg}, "
              f"max_pair_gap_deg={self.max_pair_gap_deg})", flush=True)

    def __len__(self):
        return len(self.tile_index)

    def _load_subject(self, tag: str, entry):
        if tag in self._cache:
            self._cache.move_to_end(tag)
            return self._cache[tag]

        if self.log_worker_loads:
            # Diagnostico de gargalo de dataloading (2026-09-02, ver
            # docstring de utils.dataset.SubjectGroupedSampler.__init__):
            # imprime CADA carga real de disco (cache MISS, nao HIT) junto
            # com o worker_id que a fez -- rodando com --num-workers>1,
            # procure no log por um MESMO `subject=` aparecendo com
            # MULTIPLOS `worker=` diferentes na mesma epoca: isso confirma
            # que o DataLoader esta despachando batches desse sujeito pra
            # workers diferentes (round-robin por batch, nao por bloco de
            # sujeito), forcando releitura redundante do mesmo volume 4D
            # em cada um. `worker=-1` = sem multiprocessing (--num-workers
            # 0), sempre 1 unico "worker" (o processo principal).
            worker_info = torch.utils.data.get_worker_info()
            wid = worker_info.id if worker_info is not None else -1
            print(f"[pairflow_ssl_dataset][worker-load] worker={wid} subject={tag}", flush=True)

        bvals, bvecs = load_bval_bvec(entry.bval_path, entry.bvec_path)
        data, _affine, _header = load_dwi(entry.dwi_path)
        shells = split_shells(bvals, tol=self.shell_tol)
        b0_idx = shells[0]
        b0_mean = data[..., b0_idx].mean(axis=-1)
        mask = load_or_build_mask(entry.dwi_path, b0_mean, mask_suffix=self.mask_suffix)

        shell_key = _resolve_shell_key(shells, self.shell_b, self.shell_tol)
        shell_idxs = np.asarray(shells[shell_key], dtype=int)
        if shell_idxs.size < 2:
            raise RuntimeError(f"{tag}: shell {self.shell_b} tem menos de 2 direcoes "
                                f"({shell_idxs.size}) -- impossivel formar par.")
        mask_bool = mask.astype(bool)
        shell_vals = data[..., shell_idxs][mask_bool]
        xmax = float(np.percentile(shell_vals, 99)) if shell_vals.size else 1.0
        if not np.isfinite(xmax) or xmax <= 0:
            xmax = 1.0

        cached = {
            "dwi": data.astype(np.float32),
            "mask": mask,
            "bvecs": bvecs.astype(np.float32),
            "xmax": xmax,
            "shell_idxs": shell_idxs,
        }
        self._cache[tag] = cached
        while len(self._cache) > self.max_cached_subjects:
            self._cache.popitem(last=False)
        return cached

    def _extract(self, vol: np.ndarray, ox: int, oy: int, oz: int) -> np.ndarray:
        """Duplicado de utils/rrin_dataset.py:RRINTripletDataset._extract
        (mesma logica de extracao de patch com padding nas bordas) -- sem
        import cruzado entre modulos de dataset de linhas diferentes, mesmo
        espirito ja usado pra `_resolve_shell_key` em
        scripts/11_peak_confusion_by_roi.py."""
        ps = self.patch_size
        shape = vol.shape
        ex, ey, ez = min(ox + ps, shape[0]), min(oy + ps, shape[1]), min(oz + ps, shape[2])
        sub = vol[ox:ex, oy:ey, oz:ez, ...]
        pad_x, pad_y, pad_z = ps - (ex - ox), ps - (ey - oy), ps - (ez - oz)
        if pad_x or pad_y or pad_z:
            pad_width = [(0, pad_x), (0, pad_y), (0, pad_z)] + [(0, 0)] * (sub.ndim - 3)
            sub = np.pad(sub, pad_width, mode="constant")
        return sub

    def _sample_pair_idx(self, rng: np.random.Generator, shell_idxs: np.ndarray, bvecs: np.ndarray):
        """Sorteia (por rejeicao) dois indices distintos de `shell_idxs`
        cujo gap angular caia em `[min_pair_gap_deg, max_pair_gap_deg]`
        (max=None -> sem teto superior). Desiste apos
        `max_sample_attempts` tentativas e aceita o ultimo par sorteado
        mesmo fora da faixa (log silencioso -- ver docstring da classe)."""
        n = shell_idxs.size
        last = None
        for _attempt in range(self.max_sample_attempts):
            i, j = rng.choice(n, size=2, replace=False)
            a_idx, b_idx = int(shell_idxs[i]), int(shell_idxs[j])
            gap = _angular_gap_deg(bvecs[a_idx], bvecs[b_idx])
            last = (a_idx, b_idx, gap)
            if gap < self.min_pair_gap_deg:
                continue
            if self.max_pair_gap_deg is not None and gap > self.max_pair_gap_deg:
                continue
            return a_idx, b_idx, gap
        return last

    def __getitem__(self, idx):
        si, (ox, oy, oz) = self.tile_index[idx]
        entry, tag = self.usable[si]
        d = self._load_subject(tag, entry)

        if self.training:
            rng = self._rng  # evolui a cada chamada, mesmo espirito de RRINTripletDataset
        else:
            # deterministico e reprodutivel entre epocas (val_loss
            # comparavel) -- mesma logica de RRINTripletDataset, mas via
            # RNG local seedado por (seed, idx) em vez de `idx % n_triplets`
            # (aqui nao ha uma lista fixa de trincas pra indexar por modulo).
            rng = np.random.default_rng((self.seed, idx))

        a_idx, b_idx, gap_deg = self._sample_pair_idx(rng, d["shell_idxs"], d["bvecs"])

        mask_patch = self._extract(d["mask"].astype(np.float32), ox, oy, oz)[..., None]
        xmax = d["xmax"]
        vol_a = (self._extract(d["dwi"][..., [a_idx]], ox, oy, oz) / xmax) * mask_patch
        vol_b = (self._extract(d["dwi"][..., [b_idx]], ox, oy, oz) / xmax) * mask_patch

        vol_a = np.moveaxis(vol_a, -1, 0).astype(np.float32)  # (ps,ps,ps,1) -> (1,ps,ps,ps)
        vol_b = np.moveaxis(vol_b, -1, 0).astype(np.float32)
        mask_chw = np.moveaxis(mask_patch, -1, 0).astype(np.float32)

        return {
            "vol_a": torch.from_numpy(vol_a),
            "vol_b": torch.from_numpy(vol_b),
            "bvec_a": torch.from_numpy(d["bvecs"][a_idx]),
            "bvec_b": torch.from_numpy(d["bvecs"][b_idx]),
            "mask": torch.from_numpy(mask_chw),
            "gap_deg": torch.tensor(gap_deg, dtype=torch.float32),
            "subject_tag": tag,
        }