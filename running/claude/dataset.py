"""
dataset.py
----------
Dataset para masked q-space modeling.

Filosofia:
  - Cada amostra = 1 voxel de 1 sujeito
  - As DWIs do voxel = conjunto de medições {(b_i, g_i, S_i)}
  - No treino: mascara aleatoriamente ~30% das direções
  - O modelo deve reconstruir as direções mascaradas via z_tecido

Suporte a:
  - Single-shell (ex: b=1000, 30-64 direções)
  - Multi-shell  (ex: b=0/1000/2000/3000, direções variadas)
  - Múltiplos protocolos (diferentes nº de direções, shells)

Formato esperado por sujeito (busca em ordem de prioridade):
  sub-001/
    bgpdwis_PA_geomcorr.npy             ← preferido (mais rápido)
    bgpdwis_PA_geomcorr.nii             ← ou NIfTI
    bgpdwis_PA_geomcorr.bval
    bgpdwis_PA_geomcorr.bvec
    bgpdwis_PA_geomcorr_mask3d.npy      ← preferido
    bgpdwis_PA_geomcorr_mask3d.nii.gz   ← ou NIfTI
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Nibabel para leitura de NIfTI — pip install nibabel
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("nibabel não encontrado. Use load_subject() com arrays numpy diretamente.")

# dipy para SH — pip install dipy (opcional mas recomendado)
try:
    from dipy.core.gradients import gradient_table
    from dipy.reconst.shm import sf_to_sh, sph_harm_ind_list
    HAS_DIPY = True
except ImportError:
    HAS_DIPY = False
    warnings.warn("dipy não encontrado. Features SH desabilitadas.")


# ---------------------------------------------------------------------------
# Utilitários de normalização e features
# ---------------------------------------------------------------------------

def normalize_bvals(bvals: np.ndarray, b_max: float = 5000.0) -> np.ndarray:
    """
    Normaliza b-values para [0, 1].
    b_max: valor máximo esperado no seu dataset (ajuste conforme seus dados).
    """
    return bvals / b_max


def normalize_signal(S: np.ndarray, S0: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Normaliza o sinal DWI pelo b0: S_norm = S / S0.
    Retorna valores em [0, 1] (teoricamente — clip por segurança).
    """
    S_norm = S / (S0[..., None] + eps)
    return np.clip(S_norm, 0.0, 1.0)


def build_input_features(
    bvals_norm: np.ndarray,    # (N_dwi,)
    bvecs: np.ndarray,         # (N_dwi, 3)
    S_norm: np.ndarray,        # (N_voxels, N_dwi)  — sinal do contexto (não mascarado)
    mask_context: np.ndarray,  # (N_dwi,) bool — True = direções visíveis
) -> np.ndarray:
    """
    Monta o tensor de features de input para o encoder.

    Feature por medição:
        [b_norm, g_x, g_y, g_z, S_obs]   → dim = 5

    Retorna: (N_voxels, N_context, 5)
    """
    b_ctx   = bvals_norm[mask_context]           # (N_ctx,)
    g_ctx   = bvecs[mask_context]                # (N_ctx, 3)
    S_ctx   = S_norm[:, mask_context]            # (N_voxels, N_ctx)

    N_vox, N_ctx = S_ctx.shape

    # Expande b e g para todos os voxels
    b_rep = np.tile(b_ctx[None, :, None], (N_vox, 1, 1))  # (N_vox, N_ctx, 1)
    g_rep = np.tile(g_ctx[None, :, :],    (N_vox, 1, 1))  # (N_vox, N_ctx, 3)
    S_rep = S_ctx[:, :, None]                              # (N_vox, N_ctx, 1)

    features = np.concatenate([b_rep, g_rep, S_rep], axis=-1)  # (N_vox, N_ctx, 5)
    return features.astype(np.float32)


def build_query_features(
    bvals_norm: np.ndarray,    # (N_dwi,)
    bvecs: np.ndarray,         # (N_dwi, 3)
    mask_query: np.ndarray,    # (N_dwi,) bool — True = direções mascaradas (targets)
) -> np.ndarray:
    """
    Features de query: [b_norm, g_x, g_y, g_z]  → dim = 4
    Retorna: (N_query, 4)
    """
    b_q = bvals_norm[mask_query][:, None]   # (N_q, 1)
    g_q = bvecs[mask_query]                 # (N_q, 3)
    return np.concatenate([b_q, g_q], axis=-1).astype(np.float32)  # (N_q, 4)


# ---------------------------------------------------------------------------
# Carregamento de um sujeito
# ---------------------------------------------------------------------------

# Nomes de arquivo suportados — tentados em ordem de prioridade
_DWI_CANDIDATES  = [
    "bgpdwis_PA_geomcorr.npy",       # numpy pré-processado (mais rápido)
    "bgpdwis_PA_geomcorr.nii",       # NIfTI sem compressão
    "bgpdwis_PA_geomcorr.nii.gz",    # NIfTI comprimido
    "dwi.nii.gz",                    # nome genérico
    "dwi.npy",
]
_MASK_CANDIDATES = [
    "bgpdwis_PA_geomcorr_mask3d.npy",
    "bgpdwis_PA_geomcorr_mask3d.nii.gz",
    "bgpdwis_PA_geomcorr_mask3d.nii",
    "brain_mask.nii.gz",
    "mask.npy",
]
_BVAL_CANDIDATES = [
    "bgpdwis_PA_geomcorr.bval",
    "bvals",
]
_BVEC_CANDIDATES = [
    "bgpdwis_PA_geomcorr.bvec",
    "bvecs",
]


def _find_file(subject_dir: Path, candidates: list) -> Path:
    for name in candidates:
        p = subject_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Nenhum dos arquivos encontrado em {subject_dir}:\n  " +
        "\n  ".join(candidates)
    )


def load_subject(subject_dir: str) -> dict:
    """
    Carrega dados de um sujeito.

    Busca os arquivos na seguinte ordem de prioridade:
      DWI  : bgpdwis_PA_geomcorr.npy  →  .nii  →  .nii.gz  →  dwi.*
      Mask : bgpdwis_PA_geomcorr_mask3d.npy  →  .nii.gz  →  brain_mask.*
      bval : bgpdwis_PA_geomcorr.bval  →  bvals
      bvec : bgpdwis_PA_geomcorr.bvec  →  bvecs

    Retorna dict com:
        'dwi'   : np.ndarray (X, Y, Z, N_dwi)  float32
        'bvals' : np.ndarray (N_dwi,)
        'bvecs' : np.ndarray (N_dwi, 3)
        'mask'  : np.ndarray (X, Y, Z) bool
        'S0'    : np.ndarray (X, Y, Z)  — média dos b0s
    """
    sd = Path(subject_dir)

    # ---- DWI ----
    dwi_path = _find_file(sd, _DWI_CANDIDATES)
    if dwi_path.suffix == ".npy":
        dwi = np.load(dwi_path).astype(np.float32)
        # Corrige shape transposto: (D, X, Y, Z) → (X, Y, Z, D)
        if dwi.ndim == 4 and dwi.shape[0] < dwi.shape[1]:
            dwi = np.transpose(dwi, (1, 2, 3, 0))
    else:
        if not HAS_NIBABEL:
            raise ImportError(f"nibabel necessário para carregar {dwi_path}. pip install nibabel")
        dwi = nib.load(dwi_path).get_fdata(dtype=np.float32)

    # ---- Mask ----
    mask_path = _find_file(sd, _MASK_CANDIDATES)
    if mask_path.suffix == ".npy":
        mask = np.load(mask_path).astype(bool)
        if mask.ndim == 4:  # (1, X, Y, Z) ou (X, Y, Z, 1) — remove dim extra
            mask = mask.squeeze()
    else:
        if not HAS_NIBABEL:
            raise ImportError(f"nibabel necessário para carregar {mask_path}. pip install nibabel")
        mask = nib.load(mask_path).get_fdata().astype(bool)

    # ---- bvals ----
    bval_path = _find_file(sd, _BVAL_CANDIDATES)
    bvals = np.loadtxt(bval_path).astype(np.float32).flatten()

    # ---- bvecs ----
    bvec_path = _find_file(sd, _BVEC_CANDIDATES)
    bvecs_raw = np.loadtxt(bvec_path).astype(np.float32)

    # FSL convention: pode ser (3, N) ou (N, 3)
    if bvecs_raw.ndim == 2 and bvecs_raw.shape[0] == 3 and bvecs_raw.shape[1] != 3:
        bvecs = bvecs_raw.T          # → (N, 3)
    else:
        bvecs = bvecs_raw

    # Normaliza bvecs para esfera unitária
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs = bvecs / np.maximum(norms, 1e-6)

    # Garante consistência de dimensões
    assert dwi.ndim == 4, f"DWI deve ser 4D (X,Y,Z,N_dwi), shape={dwi.shape}"
    N_dwi = dwi.shape[-1]
    assert len(bvals) == N_dwi, (
        f"bvals ({len(bvals)}) ≠ N_dwi ({N_dwi}) em {subject_dir}\n"
        f"  dwi.shape={dwi.shape}, bval únicos={sorted(set(bvals.round(-2).astype(int)))}\n"
        f"  Dica: o .npy pode estar com shape errado. Rode o script de diagnóstico."
    )
    assert bvecs.shape == (N_dwi, 3), \
        f"bvecs shape {bvecs.shape} ≠ ({N_dwi}, 3) em {subject_dir}"
    assert mask.shape == dwi.shape[:3], \
        f"mask shape {mask.shape} ≠ dwi shape {dwi.shape[:3]} em {subject_dir}"

    # S0: média dos volumes b≈0
    b0_idx = bvals < 50
    if b0_idx.sum() == 0:
        raise ValueError(f"Nenhum volume b0 (b<50) encontrado em {subject_dir}")
    S0 = dwi[..., b0_idx].mean(axis=-1)

    return {
        "dwi":         dwi,
        "bvals":       bvals,
        "bvecs":       bvecs,
        "mask":        mask,
        "S0":          S0,
        "subject_dir": str(subject_dir),
    }


# ---------------------------------------------------------------------------
# Dataset principal
# ---------------------------------------------------------------------------


class MaskedQSpaceDataset(Dataset):
    """
    Dataset para treino do SIREN encoder com masked q-space modeling.

    Estratégia de carregamento:
      - preload=True  (padrão): carrega TODOS os DWIs em RAM com threads
                                paralelas. Com 814 sujeitos e 436GB livres,
                                carrega tudo 1x e elimina I/O durante treino.
                                Demora ~10-30 min na primeira vez, depois voa.
      - preload=False: lazy loading com cache LRU (fallback se RAM escassa)
    """

    def __init__(
        self,
        subject_dirs: List[str],
        mask_ratio: float = 0.30,
        masking_strategy: str = "random",
        min_context: int = 6,
        b_max: float = 5000.0,
        voxels_per_subject: int = 200,
        preload: bool = True,
        ram_limit_gb: float = 100.0,  # para de preload ao atingir este limite
        cache_size: int = 5,
        n_load_workers: int = 8,
        seed: int = 42,
        augment: bool = True,
    ):
        self.mask_ratio = mask_ratio
        self.masking_strategy = masking_strategy
        self.min_context = min_context
        self.b_max = b_max
        self.voxels_per_subject = voxels_per_subject
        self.preload = preload
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        from collections import OrderedDict
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

        # ---- Fase 1: indexação leve (bval/bvec/mask) ----
        print(f"Indexando {len(subject_dirs)} sujeitos...")
        self.meta = []
        n_ok, n_fail = 0, 0

        for pid, sdir in enumerate(subject_dirs):
            try:
                sd = Path(sdir)
                bval_path = _find_file(sd, _BVAL_CANDIDATES)
                bvec_path = _find_file(sd, _BVEC_CANDIDATES)
                bvals = np.loadtxt(bval_path).astype(np.float32).flatten()
                bvecs_raw = np.loadtxt(bvec_path).astype(np.float32)
                if bvecs_raw.ndim == 2 and bvecs_raw.shape[0] == 3 and bvecs_raw.shape[1] != 3:
                    bvecs = bvecs_raw.T
                else:
                    bvecs = bvecs_raw
                norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
                bvecs = bvecs / np.maximum(norms, 1e-6)

                mask_path = _find_file(sd, _MASK_CANDIDATES)
                if mask_path.suffix == ".npy":
                    mask = np.load(mask_path).astype(bool)
                    if mask.ndim == 4:
                        mask = mask.squeeze()
                else:
                    if not HAS_NIBABEL:
                        raise ImportError("nibabel necessário para máscara NIfTI")
                    mask = nib.load(mask_path).get_fdata().astype(bool)

                valid_voxels = np.argwhere(mask).astype(np.int32)
                shells = np.unique(np.round(bvals, -2).astype(int))

                self.meta.append({
                    "subject_dir":  str(sdir),
                    "protocol_id":  pid,
                    "bvals":        bvals,
                    "bvals_norm":   normalize_bvals(bvals, b_max=b_max),
                    "bvecs":        bvecs,
                    "mask":         mask,
                    "valid_voxels": valid_voxels,
                    "shape_xyz":    mask.shape,
                    "N_dwi":        len(bvals),
                })
                n_ok += 1
                if n_ok % 50 == 0 or n_ok == 1:
                    print(f"  [{n_ok}/{len(subject_dirs)}] {Path(sdir).name}: "
                          f"{len(valid_voxels)} voxels, shells={shells}")
            except Exception as e:
                print(f"  ✗ {Path(sdir).name}: {e}")
                n_fail += 1

        print(f"Indexação: {n_ok} OK, {n_fail} falhas")

        # ---- Fase 2: preload ou lazy ----
        self.S_norm_all: dict = {}
        if preload:
            self._preload_parallel(n_load_workers, ram_limit_gb=ram_limit_gb)


        else:
            print(f"Modo lazy (cache_size={cache_size}) — I/O ocorre durante treino")

        self._build_index()

    # -----------------------------------------------------------------------
    # Preload paralelo com threads
    # -----------------------------------------------------------------------

    def _load_one(self, s_idx: int):
        """Carrega e normaliza S_norm de um sujeito. Chamado pelas threads."""
        meta = self.meta[s_idx]
        sd   = Path(meta["subject_dir"])
        try:
            dwi_path = _find_file(sd, _DWI_CANDIDATES)
            if dwi_path.suffix == ".npy":
                dwi = np.load(dwi_path).astype(np.float32)
                if dwi.ndim == 4 and dwi.shape[0] < dwi.shape[1]:
                    dwi = np.transpose(dwi, (1, 2, 3, 0))
            else:
                if not HAS_NIBABEL:
                    raise ImportError("nibabel necessário")
                dwi = nib.load(dwi_path).get_fdata(dtype=np.float32)

            b0_idx = meta["bvals"] < 50
            S0     = dwi[..., b0_idx].mean(axis=-1)
            S_norm = normalize_signal(
                dwi.reshape(-1, dwi.shape[-1]),
                S0.reshape(-1),
            )
            # Guarda em float16 — metade da RAM, precisão suficiente para [0,1]
            return s_idx, S_norm.astype(np.float16), None
        except Exception as e:
            return s_idx, None, str(e)

    def _preload_parallel(self, n_workers: int = 8, ram_limit_gb: float = 0.0):
        """
        Carrega DWIs em RAM usando ThreadPoolExecutor.
        ram_limit_gb: para de carregar ao atingir este limite (0 = sem limite).
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n = len(self.meta)
        limit_str = f", limite={ram_limit_gb:.0f}GB" if ram_limit_gb > 0 else ""
        print(f"Preloading {n} sujeitos em RAM com {n_workers} threads{limit_str}...")

        t0 = time.time()
        n_done, n_err = 0, 0
        total_gb = 0.0

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(self._load_one, i): i for i in range(n)}
            for fut in as_completed(futures):
                s_idx, S_norm, err = fut.result()
                if err:
                    print(f"  ✗ {Path(self.meta[s_idx]['subject_dir']).name}: {err}")
                    n_err += 1
                else:
                    # Para se atingiu limite de RAM
                    if ram_limit_gb > 0 and total_gb + S_norm.nbytes / 1e9 > ram_limit_gb:
                        n_remaining = n - n_done - n_err
                        print(f"  Limite de RAM ({ram_limit_gb:.0f}GB) atingido. "
                              f"{n_remaining} sujeitos restantes ficarão em lazy loading.")
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    self.S_norm_all[s_idx] = S_norm
                    total_gb += S_norm.nbytes / 1e9
                    n_done += 1
                    if n_done % 50 == 0 or n_done == 1:
                        elapsed = time.time() - t0
                        rate    = n_done / elapsed
                        eta     = (n - n_done) / rate if rate > 0 else 0
                        print(f"  [{n_done}/{n}] | {total_gb:.1f} GB | "
                              f"{elapsed/60:.1f} min | ETA {eta/60:.1f} min")

        elapsed = time.time() - t0
        n_lazy  = n - n_done - n_err
        print(f"Preload: {n_done} em RAM ({total_gb:.1f} GB), "
              f"{n_lazy} em lazy, {n_err} falhas — {elapsed/60:.1f} min")

        if n_err > 0:
            # Remove entradas sem dados do meta
            valid_keys = sorted(self.S_norm_all.keys())
            # Mantém todos os meta (lazy vai carregar os que não estão no S_norm_all)
            for new_i, old_i in enumerate(range(len(self.meta))):
                self.meta[old_i]["protocol_id"] = old_i

    # -----------------------------------------------------------------------
    # Acesso a dados
    # -----------------------------------------------------------------------

    def _get_S_norm(self, s_idx: int) -> np.ndarray:
        """Retorna S_norm — da RAM (preload) ou do disco (lazy fallback)."""
        if s_idx in self.S_norm_all:
            return self.S_norm_all[s_idx]
        # Lazy LRU — para sujeitos além do limite de RAM ou preload=False
        if s_idx in self._cache:
            self._cache.move_to_end(s_idx)
            return self._cache[s_idx]
        _, S_norm, err = self._load_one(s_idx)
        if err:
            raise RuntimeError(f"Erro ao carregar sujeito {s_idx}: {err}")
        self._cache[s_idx] = S_norm
        self._cache.move_to_end(s_idx)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return S_norm

    # -----------------------------------------------------------------------
    # Index
    # -----------------------------------------------------------------------

    def _build_index(self):
        """Agrupa voxels por sujeito — 1 leitura de disco por bloco."""
        blocks = []
        for s_idx, meta in enumerate(self.meta):
            if self.preload and s_idx not in self.S_norm_all:
                continue
            n_vox  = len(meta["valid_voxels"])
            chosen = (self.rng.choice(n_vox, self.voxels_per_subject, replace=False)
                      if n_vox > self.voxels_per_subject else np.arange(n_vox))
            blocks.append([(s_idx, int(v)) for v in chosen])

        block_order = self.rng.permutation(len(blocks))
        self.index  = []
        for b in block_order:
            self.index.extend(blocks[b])

    def resample(self):
        self._build_index()

    # -----------------------------------------------------------------------
    # Masking
    # -----------------------------------------------------------------------

    def _get_mask(self, bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
        N = len(bvals)
        if self.masking_strategy == "random":
            dwi_idx    = np.where(bvals >= 50)[0]
            n_mask     = max(1, int(len(dwi_idx) * self.mask_ratio))
            masked     = self.rng.choice(dwi_idx, n_mask, replace=False)
            query_mask = np.zeros(N, dtype=bool)
            query_mask[masked] = True
        elif self.masking_strategy == "shell":
            shells     = np.unique(np.round(bvals, -2))
            dwi_shells = shells[shells >= 50]
            target     = (self.rng.choice(dwi_shells) if len(dwi_shells) > 1
                          else dwi_shells[0])
            query_mask = (np.round(bvals, -2) == target)
        elif self.masking_strategy == "angular":
            pole       = self.rng.normal(size=3)
            pole      /= np.linalg.norm(pole)
            cos_sim    = np.abs(bvecs @ pole)
            threshold  = self.rng.uniform(0.7, 0.95)
            query_mask = (cos_sim > threshold) & (bvals >= 50)
            if query_mask.sum() == 0:
                return self._get_mask(bvals, bvecs)
        else:
            raise ValueError(f"Masking strategy desconhecida: {self.masking_strategy}")

        context_count = (~query_mask).sum()
        if context_count < self.min_context:
            masked_idx = np.where(query_mask)[0]
            n_unmask   = self.min_context - context_count
            unmask     = self.rng.choice(masked_idx,
                                         min(n_unmask, len(masked_idx)), replace=False)
            query_mask[unmask] = False
        return query_mask

    # -----------------------------------------------------------------------
    # __len__ / __getitem__
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        s_idx, v_idx = self.index[idx]
        meta   = self.meta[s_idx]
        S_norm = self._get_S_norm(s_idx)

        xyz      = meta["valid_voxels"][v_idx]
        X, Y, Z  = meta["shape_xyz"]
        flat_idx = int(xyz[0]) * Y * Z + int(xyz[1]) * Z + int(xyz[2])

        bvals  = meta["bvals"]
        bvecs  = meta["bvecs"].copy()
        b_norm = meta["bvals_norm"]
        S_all  = S_norm[flat_idx].astype(np.float32)  # float16→32 só pro voxel atual

        query_mask   = self._get_mask(bvals, bvecs)
        context_mask = ~query_mask

        if self.augment:
            bvecs = self._random_rotation(bvecs)

        x_context = np.concatenate([
            b_norm[context_mask][:, None],
            bvecs[context_mask],
            S_all[context_mask][:, None],
        ], axis=-1)

        q_query = np.concatenate([
            b_norm[query_mask][:, None],
            bvecs[query_mask],
        ], axis=-1)

        return {
            "x_context":   torch.from_numpy(x_context).float(),
            "q_query":     torch.from_numpy(q_query).float(),
            "S_target":    torch.from_numpy(S_all[query_mask]).float(),
            "protocol_id": torch.tensor(meta["protocol_id"], dtype=torch.long),
            "bvals_query": torch.from_numpy(bvals[query_mask]).float(),
        }

    def _random_rotation(self, bvecs: np.ndarray) -> np.ndarray:
        axis  = self.rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = self.rng.uniform(-np.pi / 36, np.pi / 36)
        c, s  = np.cos(angle), np.sin(angle)
        K = np.array([
            [0,       -axis[2],  axis[1]],
            [axis[2],  0,       -axis[0]],
            [-axis[1], axis[0],  0      ],
        ])
        R = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)
        return (R @ bvecs.T).T

# ---------------------------------------------------------------------------
# Collate function: lida com N_dwi variável entre sujeitos
# ---------------------------------------------------------------------------

def collate_variable_dwi(batch: List[dict]) -> dict:
    """
    Agrega batch com número variável de direções (N_ctx e N_q podem diferir).
    Usa padding + máscara de atenção.
    """
    max_ctx = max(b["x_context"].shape[0] for b in batch)
    max_q   = max(b["q_query"].shape[0]   for b in batch)

    ctx_dim = batch[0]["x_context"].shape[-1]
    q_dim   = batch[0]["q_query"].shape[-1]

    B = len(batch)

    x_ctx_pad   = torch.zeros(B, max_ctx, ctx_dim)
    ctx_mask    = torch.zeros(B, max_ctx, dtype=torch.bool)   # True = padding
    q_pad       = torch.zeros(B, max_q,   q_dim)
    S_target_pad = torch.zeros(B, max_q)
    q_mask      = torch.zeros(B, max_q,   dtype=torch.bool)
    protocol_ids = torch.zeros(B, dtype=torch.long)
    bvals_q_pad = torch.zeros(B, max_q)

    for i, item in enumerate(batch):
        n_ctx = item["x_context"].shape[0]
        n_q   = item["q_query"].shape[0]

        x_ctx_pad[i, :n_ctx]     = item["x_context"]
        ctx_mask[i, n_ctx:]      = True          # padding positions

        q_pad[i, :n_q]           = item["q_query"]
        S_target_pad[i, :n_q]    = item["S_target"]
        q_mask[i, n_q:]          = True          # padding positions
        protocol_ids[i]          = item["protocol_id"]
        bvals_q_pad[i, :n_q]     = item["bvals_query"]

    return {
        "x_context":    x_ctx_pad,
        "ctx_mask":     ctx_mask,
        "q_query":      q_pad,
        "S_target":     S_target_pad,
        "q_mask":       q_mask,
        "protocol_id":  protocol_ids,
        "bvals_query":  bvals_q_pad,
    }


def get_dataloader(
    subject_dirs: List[str],
    batch_size: int = 32,
    num_workers: int = 0,
    **dataset_kwargs,
) -> Tuple[DataLoader, MaskedQSpaceDataset]:
    """
    Factory function conveniente.

    Usa sampler sequencial (não shuffle=True) porque o _build_index já
    embaralha a ordem dos sujeitos. Isso garante que batches consecutivos
    vêm do mesmo sujeito → cache hit → sem freeze no início.
    """
    from torch.utils.data import SequentialSampler
    dataset = MaskedQSpaceDataset(subject_dirs, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),  # índice já está embaralhado por sujeito
        num_workers=num_workers,
        collate_fn=collate_variable_dwi,
        pin_memory=num_workers > 0,
        drop_last=True,
    )
    return loader, dataset