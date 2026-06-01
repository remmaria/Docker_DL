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
from scipy.ndimage import binary_erosion

# FIX: número de iterações de erosão deve ser idêntico ao make_patches.py
MASK_EROSION_ITERS = 2

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
    <npy_dir>/<session_id>/<dwi_name>.npy   (D, X, Y, Z) float32
    <npy_dir>/<session_id>/<dwi_name>.bval
    <npy_dir>/<session_id>/<dwi_name>.bvec
    <npy_dir>/<session_id>/<dwi_name>_mask3d.nii.gz      (máscara binária)
    <npy_dir>/<session_id>/<dwi_name>_wm_mask.nii.gz     (máscara WM)

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
        print(self.df.columns)
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
            session_id = str(row["SessionID"])
            base = os.path.join(self.npy_dir, session_id, self.dwi_name)
            return {
                "session_id":  session_id,
                "dwi_npy":     f"{base}.npy",
                "bval_path":   f"{base}.bval",
                "bvec_path":   f"{base}.bvec",
                "mask_path":   f"{base}_mask3d.nii.gz",
                "wm_mask_path": f"{base}_wm_mask.nii.gz",
            }
        else:
            # Caminhos completos já estão no CSV
            return {
                "session_id":   str(row["subject"]),
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

        self.meta_cache[key] = {
            "bvals":        bvals,
            "bvecs":        bvecs,
            "idx_diff":     idx_diff,
            "idx_b0":       idx_b0,
            "unique_shells": unique_shells,
            "shell_map":    shell_map,
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

        # FIX: binariza explicitamente (> 0.5) para evitar valores float
        # intermediários que fazem patch_mask == 1 retornar array vazio.
        # FIX: aplica erosão idêntica ao make_patches.py (MASK_EROSION_ITERS=2)
        # para garantir que a máscara em runtime seja a mesma usada para
        # selecionar os patches no CSV.
        mask_raw = nib.load(paths["mask_path"]).get_fdata() > 0
        mask_eroded = binary_erosion(mask_raw, iterations=MASK_EROSION_ITERS)
        mask = mask_eroded.astype(np.float32)

        # WM mask é opcional — retorna zeros se não existir
        # FIX: também binariza a WM mask pelo mesmo motivo
        wm_path = paths["wm_mask_path"]
        if wm_path and os.path.exists(wm_path):
            wm_mask = (nib.load(wm_path).get_fdata() > 0).astype(np.float32)
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
                f"[CACHE MISS VOLUME] {paths['session_id']}: "
                f"{time.time() - t0:.2f}s"
            )

        return entry

    # ------------------------------------------------------------------
    # Seleção de vizinhos — estratégia por fase
    # ------------------------------------------------------------------

    def _select_neighbors(
        self,
        idx_input_valid: np.ndarray,
        distancias: np.ndarray,
        pool_vecs_hemi: np.ndarray,
        target_v: np.ndarray,
        k: int,
        fase: int,
    ) -> list:
        """
        Seleciona K vizinhos com estratégia adaptada à fase de treino.

        Fase 1 — Concentrado (same-shell, interpolação angular pura)
        ------------------------------------------------------------
        Pega os K vizinhos angularmente mais próximos do target.
        Garante w_dominante > 0.4, preserva estrutura direcional fina
        e dá ao decoder resíduo real para aprender.
        A estratégia anterior de bins distribuía por 360° e forçava
        w_max ~0.15, suavizando toda a anisotropia do tecido.

        Fase 2 — Misto (cross-shell: k_close próximos + k_diverse diversos)
        -------------------------------------------------------------------
        Para harmonização cross-shell, o decoder precisa:
          - Vizinhos angularmente próximos: âncora para a direção do sinal
          - Vizinhos angularmente diversos: contexto do perfil de decaimento
            T2 (necessário para estimar delta_b corretamente)
        Divide K em: metade próximos + metade diversos (quadrantes de 90°).

        Fase 3 — Igual à fase 2 (refinamento, mesma estratégia)
        """
        ordem = np.argsort(distancias)  # crescente por distância angular

        if fase == 1:
            # Fase 1: K mais próximos angularmente
            selecionados = ordem[:k]
            return [int(idx_input_valid[i]) for i in selecionados]

        else:
            # Fases 2 e 3: metade próximos + metade diversos
            k_close   = k // 2       # ex: k=8 -> 4 próximos
            k_diverse = k - k_close  # ex: k=8 -> 4 diversos

            close_indices = [int(idx_input_valid[i]) for i in ordem[:k_close]]

            # Diversos: quadrantes no plano tangente ao target
            v_ref  = np.array([1, 0, 0]) if abs(target_v[0]) < 0.9 else np.array([0, 1, 0])
            orto_x = np.cross(target_v, v_ref)
            orto_x /= np.linalg.norm(orto_x) + 1e-8
            orto_y  = np.cross(target_v, orto_x)

            angulos = np.degrees(np.arctan2(
                np.dot(pool_vecs_hemi, orto_y),
                np.dot(pool_vecs_hemi, orto_x),
            )) % 360

            bins = np.linspace(0, 360, k_diverse + 1)
            diverse_indices = []
            close_set = set(close_indices)

            for i_bin in range(len(bins) - 1):
                mask_bin = (angulos >= bins[i_bin]) & (angulos < bins[i_bin + 1])
                candidatos = [c for c in np.where(mask_bin)[0]
                              if int(idx_input_valid[c]) not in close_set]
                if candidatos:
                    melhor = candidatos[int(np.argmin(distancias[candidatos]))]
                    idx_global = int(idx_input_valid[melhor])
                    if idx_global not in close_set and idx_global not in diverse_indices:
                        diverse_indices.append(idx_global)

            # fallback: completa com os mais próximos restantes
            if len(diverse_indices) < k_diverse:
                usados = close_set | set(diverse_indices)
                for i in ordem:
                    if len(diverse_indices) >= k_diverse:
                        break
                    cand = int(idx_input_valid[i])
                    if cand not in usados:
                        diverse_indices.append(cand)
                        usados.add(cand)

            return (close_indices + diverse_indices)[:k]

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        start_total = time.time()

        # ---- metadados do CSV ----------------------------------------
        t0  = time.time()
        row = self.records[idx]
        paths = self._paths_for(row)
        session_id = paths["session_id"]
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

        # FIX: valida que o patch tem o tamanho esperado.
        # Se o .npy tiver shape diferente do .nii usado no make_patches.py,
        # os bounds podem ser truncados e o patch fica menor que patch_size,
        # causando erros silenciosos no batch (tensores de shapes diferentes).
        expected = self.patch_size
        if (xe - xs) != expected or (ye - ys) != expected or (ze - zs) != expected:
            raise ValueError(
                f"Patch com shape errado para session={session_id} idx={idx}: "
                f"got ({xe-xs},{ye-ys},{ze-zs}), esperado ({expected},{expected},{expected}). "
                f"Volume shape={vol['shape']}, center=({cx},{cy},{cz})"
            )
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

        # ---- busca kNN consistente com inferência --------------------------
        t0 = time.time()

        # remove b0
        idx_input_valid = idx_input[bvals[idx_input] > 50]

        # remove target explicitamente (evita leakage)
        idx_input_valid = idx_input_valid[
            idx_input_valid != target_idx
        ]

        # fallback extremo
        if len(idx_input_valid) == 0:
            idx_input_valid = np.array(
                [target_idx] * self.k_neighbors
            )

        # hemisfério alinhado
        pool_vecs_hemi = np.array([
            v if np.dot(v, target_v) >= 0 else -v
            for v in bvecs[idx_input_valid]
        ])

        # DISTÂNCIA ANGULAR REAL para todos os candidatos
        dots = np.abs(np.sum(pool_vecs_hemi * target_v, axis=1))
        dots = np.clip(dots, -1.0, 1.0)
        distancias = np.arccos(dots)   # radianos, 0 = idêntico

        neighbor_indices = self._select_neighbors(
            idx_input_valid=idx_input_valid,
            distancias=distancias,
            pool_vecs_hemi=pool_vecs_hemi,
            target_v=target_v,
            k=self.k_neighbors,
            fase=self.fase,
        )

        # fallback extremo
        if len(neighbor_indices) == 0:
            neighbor_indices = [int(target_idx)] * self.k_neighbors

        # completa se faltar
        while len(neighbor_indices) < self.k_neighbors:
            neighbor_indices.append(neighbor_indices[0])

        self._log(
            f"[TIMER] neighbor search: "
            f"{time.time()-t0:.4f}s"
        )
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
        # FIX: removido "mean_b0 * patch_mask" aqui — era redundante com o
        # filtro patch_mask == 1 abaixo, e causava distorção quando a máscara
        # tinha valores float (ex: 0.7 * sinal ao invés de sinal puro).

        # ---- stack de vizinhos ----------------------------------------------
        t0 = time.time()
        # Máscara binária do patch: (1, px, py, pz) — usada para zerar background
        patch_mask_bin = (patch_mask > 0.5)[np.newaxis]          # (1, px, py, pz)

        # FIX: guard — se a máscara estiver toda zero, o patch não deveria
        # estar no CSV. Lança erro explícito em vez de crash obscuro no numpy.
        if patch_mask_bin.sum() == 0:
            raise ValueError(
                f"Patch sem voxels na máscara para session={session_id} idx={idx}. "
                f"Verifique se MASK_EROSION_ITERS={MASK_EROSION_ITERS} está igual ao make_patches.py."
            )

        # Valores de b0 APENAS onde a máscara é 1 (dentro do cérebro)
        b0_in_mask = mean_b0[:, patch_mask_bin[0]]   # FIX: usa patch_mask_bin (bool) em vez de patch_mask == 1

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
        target_norm = target_norm * patch_mask_bin
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

        # ---- peso dominante (para monitorar qualidade da seleção) -----------
        # Calcula os mesmos pesos que o modelo vai usar (temperatura 0.1, fase 1)
        # para expor o w_dominante no batch e detectar patches com seleção ruim.
        nc_arr   = np.array(neighbors_coords, dtype=np.float32)  # (K, 4)
        neigh_vs = nc_arr[:, 1:]                                  # (K, 3)
        dots_w   = np.abs(neigh_vs @ target_v)
        dots_w   = np.clip(dots_w, 0.0, 1.0)
        scores_w = dots_w / 0.1
        exp_w    = np.exp(scores_w - scores_w.max())
        softmax_w = exp_w / exp_w.sum()
        w_dominant = float(softmax_w.max())
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

            "id":          session_id,
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

            # Peso do vizinho dominante — útil para filtrar patches ruins no debug
            # e monitorar se a seleção por K-próximos está funcionando.
            # Com bins angulares: w_dominant ~ 0.15
            # Com K-próximos:     w_dominant > 0.40
            "w_dominant": torch.tensor(w_dominant, dtype=torch.float32),
        }

        self._log(f"[TIMER] torch conversion: {time.time()-t0:.4f}s")
        self._log(
            f"[TOTAL] idx={idx}: {time.time()-start_total:.4f}s\n"
        )

        return result