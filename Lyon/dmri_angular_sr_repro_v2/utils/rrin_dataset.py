"""
Dataset PyTorch para treino da linha RRIN/VFI-por-triplets (ver protocolo,
secao 10.1). Espelha bastante utils/dataset.py:DWIPatchDataset (mesma
mascara, mesma normalizacao por percentil, mesma grade de patches
deterministica), mas le o esquema de TRINCAS ja construido por
scripts/02b_build_rrin_triplets.py (`<tag>_rrin_triplets.npz`) em vez do
esquema de subamostragem `n_level`/`q_out` (`<tag>_scheme.npz`).

Diferenca central pro DWIPatchDataset: cada item aqui e UM par (a,b) +
UMA direcao-alvo (nao uma sequencia N_in -> N_out) -- RRIN3D (ver
model/rrin3d.py) preve uma direcao de cada vez a partir de duas vizinhas,
entao nao ha necessidade do collate customizado com padding de
DWIPatchDataset (todo item tem exatamente o mesmo shape), o DataLoader
default_collate do PyTorch basta.

Reaproveita `SubjectGroupedSampler` e `worker_init_fn` de utils/dataset.py
(sao genericos -- so dependem de dataset.tile_index/.seed/._rng, presentes
aqui tambem) em vez de duplicar.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .gradients import load_bval_bvec, load_dwi, split_shells
from .masking import load_or_build_mask
from .dataset import _resolve_shell_key, _lightweight_subject_mask, _tile_origins


class RRINTripletDataset(Dataset):
    def __init__(self, entries, triplets_dir: str, shell_b: float, n_level: int,
                 patch_size: int = 10, training: bool = False, only_valid: bool = True,
                 mask_suffix: str = "_mask3d.nii.gz", shell_tol: float = 100.0,
                 seed: int = 0, max_cached_subjects: int = 2,
                 min_tile_coverage: float = 0.0):
        """
        only_valid: quando True (default), so usa trincas com `valid=True`
            (residuo de colinearidade dentro do teto usado em
            02b_build_rrin_triplets.py --max-residual-deg) -- treinar em
            trincas geometricamente sem sentido so ensinaria a rede a
            "chutar" nesses casos, sem sinal real de fluxo pra aprender.
            Sujeitos sem NENHUMA trinca valida para este (shell,n_level)
            sao descartados do dataset inteiro (ver __init__).
        """
        self.entries = entries
        self.triplets_dir = Path(triplets_dir)
        self.shell_b = shell_b
        self.n_level = n_level
        self.patch_size = patch_size
        self.training = training
        self.only_valid = only_valid
        self.mask_suffix = mask_suffix
        self.shell_tol = shell_tol
        self.min_tile_coverage = min_tile_coverage
        self.max_cached_subjects = max_cached_subjects
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        key = f"{shell_b}__{n_level}"
        self.usable = []
        for e in entries:
            tag = e.subject if not e.session else f"{e.subject}_{e.session}"
            trip_path = self.triplets_dir / f"{tag}_rrin_triplets.npz"
            if not trip_path.exists():
                continue
            trip = np.load(trip_path)
            if f"{key}__target" not in trip.files:
                continue
            valid = trip[f"{key}__valid"]
            if only_valid and not valid.any():
                continue
            self.usable.append((e, tag))
        if not self.usable:
            raise RuntimeError(
                f"Nenhum sujeito tem trincas (validas={only_valid}) para shell={shell_b} "
                f"nivel={n_level} em {triplets_dir} -- rode scripts/02b_build_rrin_triplets.py "
                f"primeiro, e confira --max-residual-deg se only_valid=True nao achar nada."
            )

        self.tile_index = []
        self.tile_coverage = []
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
            raise RuntimeError("Nenhum tile com voxel de mascara encontrado (ou "
                                "--min-tile-coverage alto demais)")
        if n_seen:
            cov_arr = np.asarray(self.tile_coverage, dtype=np.float32)
            tag = "treino" if self.training else "val"
            print(f"[rrin_dataset:{tag}] tiles: {len(self.tile_index)}/{n_seen} mantidos "
                  f"(min_tile_coverage={self.min_tile_coverage}); cobertura mantida "
                  f"p10={np.percentile(cov_arr, 10):.3f} mediana={np.median(cov_arr):.3f} "
                  f"p90={np.percentile(cov_arr, 90):.3f}", flush=True)

    def __len__(self):
        return len(self.tile_index)

    def _load_subject(self, tag: str, entry):
        if tag in self._cache:
            self._cache.move_to_end(tag)
            return self._cache[tag]

        bvals, bvecs = load_bval_bvec(entry.bval_path, entry.bvec_path)
        data, affine, header = load_dwi(entry.dwi_path)
        shells = split_shells(bvals, tol=self.shell_tol)
        b0_idx = shells[0]
        b0_mean = data[..., b0_idx].mean(axis=-1)
        mask = load_or_build_mask(entry.dwi_path, b0_mean, mask_suffix=self.mask_suffix)

        shell_key = _resolve_shell_key(shells, self.shell_b, self.shell_tol)
        shell_idxs = np.asarray(shells[shell_key], dtype=int)
        mask_bool = mask.astype(bool)
        shell_vals = data[..., shell_idxs][mask_bool]
        xmax = float(np.percentile(shell_vals, 99)) if shell_vals.size else 1.0
        if not np.isfinite(xmax) or xmax <= 0:
            xmax = 1.0

        trip_path = self.triplets_dir / f"{tag}_rrin_triplets.npz"
        trip = np.load(trip_path)
        key = f"{self.shell_b}__{self.n_level}"
        target_idx = trip[f"{key}__target"]
        pair_a = trip[f"{key}__pair_a"]
        pair_b = trip[f"{key}__pair_b"]
        t_frac = trip[f"{key}__t_frac"]
        residual_deg = trip[f"{key}__residual_deg"]
        gap_deg = trip[f"{key}__gap_deg"]
        valid = trip[f"{key}__valid"]
        if self.only_valid:
            keep = valid
            target_idx = target_idx[keep]
            pair_a = pair_a[keep]
            pair_b = pair_b[keep]
            t_frac = t_frac[keep]
            residual_deg = residual_deg[keep]
            gap_deg = gap_deg[keep]
            valid = valid[keep]

        cached = {
            "dwi": data.astype(np.float32),
            "mask": mask,
            "bvecs": bvecs.astype(np.float32),
            "xmax": xmax,
            "target_idx": target_idx,
            "pair_a": pair_a,
            "pair_b": pair_b,
            "t_frac": t_frac,
            "residual_deg": residual_deg,
            "gap_deg": gap_deg,
            "valid": valid,
        }
        self._cache[tag] = cached
        while len(self._cache) > self.max_cached_subjects:
            self._cache.popitem(last=False)
        return cached

    def _extract(self, vol: np.ndarray, ox: int, oy: int, oz: int) -> np.ndarray:
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

        n_triplets = len(d["target_idx"])
        if self.training:
            # re-sorteia a cada exemplo (mesmo espirito do split dinamico de
            # DWIPatchDataset) -- qual das trincas validas deste sujeito
            # este tile usa varia a cada epoca/step, nao fixo.
            k = int(self._rng.integers(0, n_triplets))
        else:
            # deterministico -- mesmo tile sempre usa a mesma trinca entre
            # epocas (val_loss comparavel), mas espalha pelas trincas
            # disponiveis em vez de sempre pegar a primeira.
            k = idx % n_triplets

        a_idx, b_idx, t_idx = int(d["pair_a"][k]), int(d["pair_b"][k]), int(d["target_idx"][k])
        t_frac = float(d["t_frac"][k])
        # normalizados por 90 (maximo possivel com simetria antipodal, ver
        # utils/gradients.py) -- usados so quando RRIN3D(use_quality_cond=True)
        # (ver model/rrin3d.py); sempre calculados aqui (custo desprezivel),
        # quem nao usar so ignora o campo "quality" do item.
        quality = np.array([d["residual_deg"][k] / 90.0, d["gap_deg"][k] / 90.0],
                            dtype=np.float32)

        mask_patch = self._extract(d["mask"].astype(np.float32), ox, oy, oz)[..., None]
        xmax = d["xmax"]
        vol_a = (self._extract(d["dwi"][..., [a_idx]], ox, oy, oz) / xmax) * mask_patch
        vol_b = (self._extract(d["dwi"][..., [b_idx]], ox, oy, oz) / xmax) * mask_patch
        target = (self._extract(d["dwi"][..., [t_idx]], ox, oy, oz) / xmax) * mask_patch

        # (ps,ps,ps,1) -> (1,ps,ps,ps)
        vol_a = np.moveaxis(vol_a, -1, 0).astype(np.float32)
        vol_b = np.moveaxis(vol_b, -1, 0).astype(np.float32)
        target = np.moveaxis(target, -1, 0).astype(np.float32)

        return {
            "vol_a": torch.from_numpy(vol_a),
            "vol_b": torch.from_numpy(vol_b),
            "target": torch.from_numpy(target),
            "bvec_a": torch.from_numpy(d["bvecs"][a_idx]),
            "bvec_b": torch.from_numpy(d["bvecs"][b_idx]),
            "bvec_t": torch.from_numpy(d["bvecs"][t_idx]),
            "t_frac": torch.tensor(t_frac, dtype=torch.float32),
            "quality": torch.from_numpy(quality),
            "subject_tag": tag,
        }