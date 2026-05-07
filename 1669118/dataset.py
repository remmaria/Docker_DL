"""
dataset.py  —  QSpaceDataset com backend .npy / .nii (sem HDF5)

Interface idêntica ao dataset.py original:
  - QSpaceDatasetCoord_KNearest_Shell
  - mesmas chaves no __getitem__
  - set_fase(), mode_val, debug_timing, etc.

Melhorias incorporadas de dataset_nii_load.py:
  - NUMA affinity / worker pinning  (numa_init_fn)
  - LRU cache com OrderedDict + max_cache_size
  - mmap_mode='r' para leitura lazy dos volumes
  - Leitura sequencial por difusão (evita page faults espalhados)
"""

import os
import time
import random

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from collections import OrderedDict
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# NUMA affinity helpers
# ---------------------------------------------------------------------------

NUMA_AFFINITY = {
    **{i: list(range(0, 64))   for i in range(0, 4)},   # GPU 0-3 → node 0
    **{i: list(range(64, 128)) for i in range(4, 8)},   # GPU 4-7 → node 1
}


def _get_target_cpus_from_env() -> list:
    """Resolve o socket NUMA sem tocar em CUDA."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        physical_id = int(visible.split(",")[0])
        return NUMA_AFFINITY.get(physical_id, list(range(0, 64)))
    return list(range(0, 64))


# Resolve UMA VEZ no processo principal, antes de qualquer fork.
_TARGET_CPUS = _get_target_cpus_from_env()


def numa_init_fn(worker_id: int):
    """
    Passar como worker_init_fn= no DataLoader.
    Pina cada worker no subset de CPUs do socket NUMA correto.
    """
    num_workers   = torch.utils.data.get_worker_info().num_workers
    cpus_per_worker = max(1, len(_TARGET_CPUS) // num_workers)
    start         = worker_id * cpus_per_worker
    worker_cpus   = _TARGET_CPUS[start: start + cpus_per_worker]

    try:
        os.sched_setaffinity(0, worker_cpus)
    except OSError:
        pass

    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Dataset principal
# ---------------------------------------------------------------------------

class QSpaceDatasetCoord_KNearest_Shell(Dataset):
    """
    Parâmetros
    ----------
    coords_csv   : CSV com colunas subject, center_x, center_y, center_z
    npy_dir      : pasta raiz onde cada sujeito tem seu subdiretório
                   (ou None se o caminho completo estiver no CSV)
    dwi_name   : nome do subdiretório / prefixo do arquivo dentro de npy_dir
    alpha        : fator de normalização
    bval_max     : valor máximo de b para normalizar coordenadas
    patch_size   : tamanho cúbico do patch
    mode         : "train" ou "val"
    k_neighbors  : número de vizinhos kNN
    mode_val     : "patch" (mantido para compatibilidade)
    debug_timing : imprime tempos por etapa
    max_cache_size : máximo de sujeitos em cache (LRU)

    Estrutura esperada por sujeito
    ------------------------------
    <npy_dir>/<subject_id>/<dwi_name>.npy   (D, X, Y, Z) float32
    <npy_dir>/<subject_id>/<dwi_name>.bval
    <npy_dir>/<subject_id>/<dwi_name>.bvec
    <npy_dir>/<subject_id>/<dwi_name>_mask3d.nii.gz      (máscara binária)
    <npy_dir>/<subject_id>/<dwi_name>_wm_mask.nii.gz     (máscara WM)

    Caso o CSV já tenha colunas com os caminhos completos, passe npy_dir=None
    e dwi_name=None; nesse caso inclua as colunas:
        dwi_npy, bval_path, bvec_path, mask_path, wm_mask_path
    """

    def __init__(
        self,
        coords_csv,
        npy_dir,
        dwi_name,
        alpha,
        bval_max,
        patch_size,
        mode="train",
        k_neighbors=6,
        mode_val="patch",
        debug_timing=False,
        max_cache_size=300,
    ):
        self.df      = pd.read_csv(coords_csv)
        self.records = self.df.to_dict("records")

        self.npy_dir        = npy_dir
        self.dwi_name     = dwi_name
        self.patch_size     = patch_size
        self.half_patch     = patch_size // 2
        self.mode           = mode
        self.k_neighbors    = k_neighbors
        self.alpha          = alpha
        self.mode_val       = mode_val
        self.bval_max       = float(bval_max)
        self.debug_timing   = debug_timing
        self.max_cache_size = max_cache_size

        self.fase = 1

        # LRU caches separados: metadados (leve) e volumes (pesado)
        self.meta_cache   = {}
        self.volume_cache = OrderedDict()

        self._logged_workers: set = set()

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.df)

    def set_fase(self, fase):
        self.fase = fase

    def _log(self, msg):
        if self.debug_timing:
            print(msg, flush=True)

    def _should_log_once(self) -> bool:
        """Imprime apenas uma vez por worker (para não poluir o stdout)."""
        if not self.debug_timing:
            return False
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else -1
        if wid not in self._logged_workers:
            self._logged_workers.add(wid)
            return True
        return False

    # ------------------------------------------------------------------
    # Resolução de caminhos
    # ------------------------------------------------------------------

    def _paths_for(self, row) -> dict:
        """
        Retorna um dicionário com os caminhos dos arquivos do sujeito.
        Suporta tanto caminhos explícitos no CSV quanto derivados de npy_dir.
        """
        if self.npy_dir is not None:
            subject_id = str(row["subject"])
            base = os.path.join(self.npy_dir, subject_id, self.dwi_name)
            return {
                "subject_id":  subject_id,
                "dwi_npy":     f"{base}.npy",
                "bval_path":   f"{base}.bval",
                "bvec_path":   f"{base}.bvec",
                "mask_path":   f"{base}_mask3d.nii.gz",
                "wm_mask_path": f"{base}_wm_mask.nii.gz",
            }
        else:
            # Caminhos completos já estão no CSV
            return {
                "subject_id":   str(row["subject"]),
                "dwi_npy":      row["dwi_npy"],
                "bval_path":    row["bval_path"],
                "bvec_path":    row["bvec_path"],
                "mask_path":    row["mask_path"],
                "wm_mask_path": row.get("wm_mask_path", ""),
            }

    # ------------------------------------------------------------------
    # Cache de metadados (bvals/bvecs/shell_map/dot — leve, sem limite)
    # ------------------------------------------------------------------

    def _load_metadata(self, paths: dict) -> dict:
        key = paths["dwi_npy"]

        if key in self.meta_cache:
            return self.meta_cache[key]

        bvals = np.loadtxt(paths["bval_path"]).astype(np.float32)
        bvecs = np.loadtxt(paths["bvec_path"]).astype(np.float32)

        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            bvecs = bvecs.T

        idx_diff      = np.where(bvals > 50)[0]
        idx_b0        = np.where(bvals <= 50)[0]
        unique_shells = np.unique(bvals[idx_diff])

        # shell_map: b-value → índices globais (mesmo esquema do dataset.py original)
        shell_map = {
            b: np.where(np.isclose(bvals, b, atol=150))[0]
            for b in unique_shells
        }

        dot_matrix = bvecs @ bvecs.T   # (D, D) — usado na busca kNN

        self.meta_cache[key] = {
            "bvals":        bvals,
            "bvecs":        bvecs,
            "idx_diff":     idx_diff,
            "idx_b0":       idx_b0,
            "unique_shells": unique_shells,
            "shell_map":    shell_map,
            "dot_matrix":   dot_matrix,
        }

        return self.meta_cache[key]

    # ------------------------------------------------------------------
    # Cache de volumes (pesado — LRU com max_cache_size)
    # ------------------------------------------------------------------

    def _load_volume(self, paths: dict) -> dict:
        key = paths["dwi_npy"]

        if key in self.volume_cache:
            self.volume_cache.move_to_end(key)
            return self.volume_cache[key]

        t0 = time.time()

        # mmap_mode='r': abre sem carregar tudo para RAM
        dwi  = np.load(paths["dwi_npy"], mmap_mode="r")   # (D, X, Y, Z)
        mask = np.asarray(
            nib.load(paths["mask_path"]).dataobj, dtype=np.float32
        )

        # WM mask é opcional — retorna zeros se não existir
        wm_path = paths["wm_mask_path"]
        if wm_path and os.path.exists(wm_path):
            wm_mask = np.asarray(
                nib.load(wm_path).dataobj, dtype=np.float32
            )
        else:
            wm_mask = np.zeros_like(mask)

        entry = {
            "dwi":     dwi,      # mmap — lazy
            "mask":    mask,
            "wm_mask": wm_mask,
            "shape":   dwi.shape,   # (D, X, Y, Z)
        }

        self.volume_cache[key] = entry

        # Evicção LRU
        while len(self.volume_cache) > self.max_cache_size:
            self.volume_cache.popitem(last=False)

        if self._should_log_once():
            self._log(
                f"[CACHE MISS VOLUME] {paths['subject_id']}: "
                f"{time.time() - t0:.2f}s"
            )

        return entry

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        start_total = time.time()

        # ---- metadados do CSV ----------------------------------------
        t0  = time.time()
        row = self.records[idx]
        paths = self._paths_for(row)
        subject_id = paths["subject_id"]
        self._log(f"[TIMER] metadata: {time.time()-t0:.4f}s")

        # ---- caches -----------------------------------------------------
        t0   = time.time()
        meta = self._load_metadata(paths)
        vol  = self._load_volume(paths)
        self._log(f"[TIMER] cache load: {time.time()-t0:.4f}s")

        bvals      = meta["bvals"]
        bvecs      = meta["bvecs"]
        dwi        = vol["dwi"]          # mmap (D, X, Y, Z)
        D, sx, sy, sz = vol["shape"]

        # ---- coordenadas do patch ----------------------------------------
        t0 = time.time()
        cx = int(row["center_x"])
        cy = int(row["center_y"])
        cz = int(row["center_z"])
        hp = self.half_patch

        xs = max(0, cx - hp);  xe = min(sx, cx + hp)
        ys = max(0, cy - hp);  ye = min(sy, cy + hp)
        zs = max(0, cz - hp);  ze = min(sz, cz + hp)
        self._log(f"[TIMER] patch coords: {time.time()-t0:.4f}s")

        # ---- máscaras -------------------------------------------------------
        t0 = time.time()
        patch_mask   = vol["mask"][xs:xe, ys:ye, zs:ze]
        patch_maskWM = vol["wm_mask"][xs:xe, ys:ye, zs:ze]
        self._log(f"[TIMER] mask load: {time.time()-t0:.4f}s")

        # ---- target ---------------------------------------------------------
        t0 = time.time()
        target_idx  = np.random.choice(meta["idx_diff"])
        target_bval = bvals[target_idx]
        target_v    = bvecs[target_idx]
        self._log(f"[TIMER] target selection: {time.time()-t0:.4f}s")

        # ---- seleção de shell -----------------------------------------------
        t0 = time.time()
        if self.fase == 1:
            input_shell = target_bval
        else:
            available = [
                b for b in meta["unique_shells"]
                if abs(b - target_bval) > 150
            ]
            input_shell = (
                np.random.choice(available)
                if len(available) > 0
                else target_bval
            )

        idx_input = meta["shell_map"][input_shell]
        self._log(f"[TIMER] shell selection: {time.time()-t0:.4f}s")

        # ---- busca kNN -------------------------------------------------------
        t0 = time.time()
        similarities = meta["dot_matrix"][target_idx, idx_input]
        sorted_idx   = idx_input[np.argsort(-similarities)]

        neighbor_indices = []
        for candidate in sorted_idx:
            if candidate == target_idx:
                continue
            neighbor_indices.append(int(candidate))
            if len(neighbor_indices) >= self.k_neighbors:
                break

        if len(neighbor_indices) < self.k_neighbors:
            print(f"--- [DEBUG] Subject {subject_id}: found {len(neighbor_indices)}/{self.k_neighbors} neighbors at b={target_bval}")
            while len(neighbor_indices) < self.k_neighbors:
                neighbor_indices.append(neighbor_indices[0]) # Repete o primeiro até encher

        self._log(f"[TIMER] neighbor search: {time.time()-t0:.4f}s")

        # ---- b0 -------------------------------------------------------------
        t0     = time.time()
        idx_b0 = meta["idx_b0"]
        selected_b0 = (
            list(
                np.random.choice(
                    idx_b0, min(1, len(idx_b0)), replace=False
                )
            )
            if self.mode == "train"
            else list(idx_b0)
        )

        needed     = sorted(
            set(neighbor_indices + [int(target_idx)] + [int(i) for i in selected_b0])
        )
        needed_arr = np.array(needed)

        
        self._log(f"[TIMER] b0 selection: {time.time()-t0:.4f}s")

        # ---- leitura HDF5→npy (sequencial por difusão) -----------------------
        # Lê cada volume necessário separadamente para minimizar page faults.
        # dwi tem shape (D, X, Y, Z) — fatiamos na dimensão D primeiro.
        t0 = time.time()
        slices_list = []
        for d in needed_arr:
            slices_list.append(
                np.array(dwi[d, xs:xe, ys:ye, zs:ze])  # (px, py, pz)
            )
        patch = np.stack(slices_list).astype(np.float32)   # (N_needed, px, py, pz)
        patch = np.clip(patch, a_min=0, a_max=None) 

        self._log(f"[TIMER] npy read: {time.time()-t0:.4f}s")

        # ---- normalização ---------------------------------------------------
        t0      = time.time()
        idx_map = {old: new for new, old in enumerate(needed)}

        b0_vols = patch[[idx_map[i] for i in selected_b0]]     # (n_b0, px, py, pz)
        mean_b0 = np.mean(b0_vols, axis=0, keepdims=True)   # (1, px, py, pz)
        mean_b0 = mean_b0 * patch_mask

        # ---- stack de vizinhos ----------------------------------------------
        t0 = time.time()
        # Máscara binária do patch: (1, px, py, pz) — usada para zerar background
        patch_mask_bin = (patch_mask > 0.5)[np.newaxis]          # (1, px, py, pz)

        # Valores de b0 APENAS onde a máscara é 1 (dentro do cérebro)
        b0_in_mask = mean_b0[:, patch_mask == 1] 

        # Valores de b0 onde a máscara é 0 (fundo/ar)
        b0_out_mask = mean_b0[:, patch_mask == 0]

        # 1. Cria uma máscara booleana (True para negativos, False para o resto)
        negativos_mask = b0_in_mask < 0

        # 2. Conta quantos True existem
        qtd_negativos = negativos_mask.sum()

        # 3. (Opcional) Calcula a porcentagem para ver a gravidade
        total_voxels_mask = b0_in_mask.size
        porcentagem = (qtd_negativos / total_voxels_mask) * 100

        #if porcentagem > 1 or b0_in_mask.min() < -80:
        if b0_in_mask.min() < 0:
            print(f"DEBUG IN-MASK {subject_id}: min={b0_in_mask.min()}, max={b0_in_mask.max()}, mean={b0_in_mask.mean()}")
            print(f"Voxels negativos dentro da mask: {qtd_negativos} ({porcentagem:.3f}%)")
      
        #if b0_in_mask.size > 0:
            #print(f"DEBUG IN-MASK: min={b0_in_mask.min()}, max={b0_in_mask.max()}, mean={b0_in_mask.mean()}")
        #if b0_out_mask.size > 0:
            #print(f"DEBUG OUT-MASK (fundo): min={b0_out_mask.min()}, max={b0_out_mask.max()}")

        mean_b0 = np.clip(mean_b0, a_min=0, a_max=None)
        denom   = mean_b0 * self.alpha
        self._log(f"[TIMER] normalization: {time.time()-t0:.4f}s")

        neighbor_list = []
        for n_idx in neighbor_indices:
            #print("patch_neighbor: ",patch[idx_map[n_idx]].min(),patch[idx_map[n_idx]].max())
            #print("denom: ",denom.min(),denom.max())

            p_norm = np.divide(patch[idx_map[n_idx]][np.newaxis], denom, out=np.zeros_like(patch[idx_map[n_idx]][np.newaxis]), where=denom > 0)

            #print("p_norm: ",p_norm.min(),p_norm.max())            
            
            p_norm = p_norm * patch_mask_bin                      # zera background
            neighbor_list.append(p_norm)

        # (k, 1, px, py, pz)
        stacked_neighbors = np.stack(neighbor_list)
        self._log(f"[TIMER] neighbor stack: {time.time()-t0:.4f}s")

        # ---- target normalizado ----------------------------------------------
        t0 = time.time()
        target_norm = np.divide(patch[idx_map[int(target_idx)]][np.newaxis], denom, out=np.zeros_like(patch[idx_map[int(target_idx)]][np.newaxis]), where=denom > 0)
        #print("target_norm: ",target_norm.min(),target_norm.max())            
        self._log(f"[TIMER] target normalization: {time.time()-t0:.4f}s")

        # ---- coordenadas dos vizinhos ----------------------------------------
        t0 = time.time()
        neighbors_coords = []
        for n_idx in neighbor_indices:
            v = bvecs[n_idx].copy()
            if np.dot(v, target_v) < 0:
                v = -v
            neighbors_coords.append([bvals[n_idx] / self.bval_max, *v])
        self._log(f"[TIMER] neighbor coords: {time.time()-t0:.4f}s")

        # ---- conversão para torch -------------------------------------------
        t0 = time.time()
        result = {
            # (k, 1, px, py, pz) — mesma forma que o original após permute
            "source_neighbors": torch.from_numpy(
                np.ascontiguousarray(stacked_neighbors)
            ).float(),

            "neighbors_coords": torch.tensor(
                neighbors_coords, dtype=torch.float32
            ),

            # (1, px, py, pz)
            "target_real": torch.from_numpy(
                np.ascontiguousarray(target_norm)
            ).float(),

            "target_query": torch.tensor(
                [target_bval / self.bval_max, *target_v],
                dtype=torch.float32
            ),

            "mask": torch.from_numpy(patch_mask).unsqueeze(0).float(),

            "maskWM": torch.from_numpy(patch_maskWM).unsqueeze(0).float(),

            "is_cross_shell": bool(abs(input_shell - target_bval) > 150),

            "id":          subject_id,
            "origin_bval": float(input_shell),
            "target_bval": float(target_bval),
            "target_idx": torch.tensor(
                target_idx,
                dtype=torch.long
            ),
            "neighbor_indices": torch.tensor(
                neighbor_indices,
                dtype=torch.long
            ),
        }

        self._log(f"[TIMER] torch conversion: {time.time()-t0:.4f}s")
        self._log(
            f"[TOTAL] idx={idx}: {time.time()-start_total:.4f}s\n"
        )

        return result