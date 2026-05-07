import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class QSpaceDatasetCoord_KNearest_Shell(Dataset):
    """
    Dataset para interpolação/harmonização de DWI no espaço-q.

    Parâmetros
    ----------
    coords_csv  : caminho para o CSV com as colunas subject, dwi_path,
                  center_x, center_y, center_z.
    alpha       : fator de desnormalização (usado externamente para plots).
    patch_size  : tamanho do patch cúbico.
    mode        : 'train' ou 'val' (reservado para futuras augmentações).
    k_neighbors : número de vizinhos angulares a selecionar.
    mode_val    : 'patch' extrai um patch centrado nas coordenadas do CSV;
                  'full'  retorna o volume inteiro do sujeito.
    bval_max    : valor máximo de b para normalização das coordenadas q.                  
    fase        : fase de treinamento (1=same-shell, 2=mix, 3=cross-shell).
                  Pode ser alterada em tempo de execução via set_fase().
    """

    def __init__(
        self,
        coords_csv,
        alpha,
        bval_max,
        patch_size=(64, 64, 64),
        mode="train",
        k_neighbors=4,
        mode_val="patch",
    ):
        self.df = pd.read_csv(coords_csv)
        self.patch_size = patch_size
        self.mode = mode
        self.half_patch = patch_size[0] // 2
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.mode_val = mode_val
        self.bval_max = float(bval_max)

        # Fase inicial padrão (1 = interpolação angular, mesmo shell)
        self.fase = 1

    def __len__(self):
        return len(self.df)

    def set_fase(self, fase):
        """Muda a fase de amostragem em tempo de execução (chamado pelo loop de treino)."""
        self.fase = fase

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = str(row['subject'])
        dwi_path_nii = row['dwi_path']

        # 1. MAPEAMENTO DE CAMINHOS (.nii -> .npy)
        dwi_path   = dwi_path_nii.replace('.nii.gz', '.npy').replace('.nii', '.npy')
        mask_path  = dwi_path_nii.replace('.nii.gz', '_brain_mask.npy').replace('.nii', '_brain_mask.npy')
        maskWM_path = dwi_path_nii.replace('.nii.gz', '_maskWM.npy').replace('.nii', '_maskWM.npy')

        # 2. CARREGAMENTO COM MMAP
        try:
            full_volume = np.load(dwi_path, mmap_mode='r')
            full_mask   = (
                np.load(mask_path, mmap_mode='r')
                if os.path.exists(mask_path)
                else np.ones(full_volume.shape[:3], dtype=np.uint8)
            )
            full_maskWM = (
                np.load(maskWM_path, mmap_mode='r')
                if os.path.exists(maskWM_path)
                else np.ones(full_volume.shape[:3], dtype=np.uint8)
            )

            base_path = dwi_path_nii.replace('.nii.gz', '').replace('.nii', '')
            bvals = np.loadtxt(base_path + '.bval')
            bvecs = np.loadtxt(base_path + '.bvec').T

        except Exception as e:
            print(f"❌ Erro ao carregar dados para {subject_id}: {e}", flush=True)
            return self.__getitem__(np.random.randint(0, len(self.df)))

        bval_norm = self.bval_max

        # 3. SELEÇÃO DINÂMICA BASEADA NA FASE
        idx_diff = np.where(bvals > 50)[0]
        target_idx  = np.random.choice(idx_diff)
        target_bval = bvals[target_idx]
        target_v    = bvecs[target_idx]

        unique_shells = np.unique(bvals[bvals > 50])

        # --- LÓGICA DE FASES ---
        if self.fase == 1:
            # FASE 1: Interpolação Angular (Mesmo Shell)
            input_shell = target_bval
        elif self.fase == 2:
            # FASE 2: Harmonização (Mix de Same-Shell e Cross-Shell)
            if np.random.rand() < 0.5:
                input_shell = target_bval
            else:
                available = [b for b in unique_shells if not np.isclose(b, target_bval, atol=150)]
                input_shell = np.random.choice(available) if available else target_bval
        else:
            # FASE 3: Fine-tuning (Foco em Cross-Shell complexo)
            available = [b for b in unique_shells if not np.isclose(b, target_bval, atol=150)]
            input_shell = np.random.choice(available) if available else target_bval

        idx_input = np.where(np.isclose(bvals, input_shell, atol=150))[0]

        # 4. SELEÇÃO DE VIZINHOS (por setores angulares no hemisfério local)
        pool_vecs_hemi = np.array([
            v if np.dot(v, target_v) >= 0 else -v for v in bvecs[idx_input]
        ])

        v_ref  = np.array([1, 0, 0]) if abs(target_v[0]) < 0.9 else np.array([0, 1, 0])
        orto_x = np.cross(target_v, v_ref)
        orto_x /= np.linalg.norm(orto_x)
        orto_y = np.cross(target_v, orto_x)

        angulos   = np.degrees(
            np.arctan2(
                np.dot(pool_vecs_hemi, orto_y),
                np.dot(pool_vecs_hemi, orto_x)
            )
        ) % 360
        distancias = np.minimum(
            np.linalg.norm(bvecs[idx_input] - target_v, axis=1),
            np.linalg.norm(bvecs[idx_input] + target_v, axis=1)
        )

        # Exclui o próprio target da busca
        for i, orig_idx in enumerate(idx_input):
            if orig_idx == target_idx or distancias[i] < 0.01:
                distancias[i] = np.inf

        # Seleção por Setores (Bins angulares)
        bins = np.linspace(0, 360, self.k_neighbors + 1)
        neighbor_indices = []

        for i in range(len(bins) - 1):
            mask_q = (angulos >= bins[i]) & (angulos < bins[i + 1])
            if np.any(mask_q):
                idx_q = np.where(mask_q)[0]
                best  = idx_q[np.argmin(distancias[idx_q])]
                if distancias[best] != np.inf:
                    neighbor_indices.append(idx_input[best])

        # Preenchimento se faltar vizinhos
        if len(neighbor_indices) < self.k_neighbors:
            faltam       = self.k_neighbors - len(neighbor_indices)
            mask_valida  = distancias != np.inf
            for ja in neighbor_indices:
                mask_valida &= (idx_input != ja)

            if np.any(mask_valida):
                sobras_idx  = idx_input[mask_valida]
                sobras_dist = distancias[mask_valida]
                extras      = sobras_idx[np.argsort(sobras_dist)]
                neighbor_indices.extend(extras[:faltam].tolist())

        if len(neighbor_indices) < self.k_neighbors:
            return self.__getitem__(np.random.randint(0, len(self.df)))

        # 5. EXTRAÇÃO DO VOLUME/PATCH
        if self.mode_val == "patch":
            cx, cy, cz = int(row['center_x']), int(row['center_y']), int(row['center_z'])
            hp = self.half_patch
            x_s, x_e = cx - hp, cx + hp
            y_s, y_e = cy - hp, cy + hp
            z_s, z_e = cz - hp, cz + hp

            vol_data    = full_volume[x_s:x_e, y_s:y_e, z_s:z_e]
            patch_mask  = full_mask[x_s:x_e, y_s:y_e, z_s:z_e].astype(np.float32)
            patch_maskWM = full_maskWM[x_s:x_e, y_s:y_e, z_s:z_e].astype(np.float32)
        else:
            # MODO FULL: volume inteiro para validação end-to-end
            vol_data    = full_volume[...]
            patch_mask  = full_mask[...].astype(np.float32)
            patch_maskWM = full_maskWM[...].astype(np.float32)

        # 6. NORMALIZAÇÃO PELO B0
        idx_b0 = np.where(bvals < 50)[0]
        # patch_b0_all agora contém APENAS os volumes de b0 deste sujeito
        patch_b0_all = vol_data[..., idx_b0].astype(np.float32) 
        n_b0_disponiveis = patch_b0_all.shape[-1] # Pode ser 4, 9, etc.

        if self.mode == "train":
            # Sorteamos índices RELATIVOS (de 0 até n_b0_disponiveis - 1)
            max_to_sample = min(n_b0_disponiveis, 3) 
            n_b0_to_use = np.random.randint(1, max_to_sample + 1)
            
            # CORREÇÃO AQUI: Escolhemos índices de 0 a n_disponivel, 
            # não os índices globais do arquivo DWI.
            sel_b0_relativo = np.random.choice(range(n_b0_disponiveis), n_b0_to_use, replace=False)
        else:
            # Na validação, usamos todos os b0s disponíveis
            sel_b0_relativo = slice(None) 

        # Cálculo da média usando os índices relativos ao patch_b0_all
        mean_b0 = np.mean(patch_b0_all[..., sel_b0_relativo], axis=-1, keepdims=True)
        mean_b0 = np.maximum(mean_b0, 1e-8)

        # 7. TARGET E VIZINHOS NORMALIZADOS
        # Agora o mean_b0 tem o shape espacial correto e a média correta.
        
        # Extração do Target (usamos slice para manter a dimensão de canal)
        target_patch = vol_data[..., target_idx:target_idx + 1].astype(np.float32)
        target_norm  = np.clip(target_patch / (mean_b0 * self.alpha), 0, 1)

        neighbor_list    = []
        neighbors_coords = []

        for n_idx in neighbor_indices[:self.k_neighbors]:
            p      = vol_data[..., n_idx:n_idx + 1].astype(np.float32)
            p_norm = np.clip(p / (mean_b0 * self.alpha), 0, 1)
            neighbor_list.append(p_norm)
            
            # Rebatimento para hemisfério do target
            v_vec = bvecs[n_idx].copy()
            if np.dot(v_vec, target_v) < 0:
                v_vec = -v_vec

            # CORRIGIDO: normalização usa self.bval_max dinamicamente
            neighbors_coords.append([bvals[n_idx] / bval_norm, *v_vec])

        return {
            # [K, 1, H, W, D] — vizinhos empilhados
            "source_neighbors": torch.from_numpy(
                np.stack(neighbor_list)
            ).permute(0, 4, 1, 2, 3).contiguous().half(),

            # [K, 4] — coordenadas q de cada vizinho
            "neighbors_coords": torch.tensor(neighbors_coords, dtype=torch.float16),

            # [1, H, W, D] — target a ser predito
            "target_real": torch.from_numpy(target_norm).permute(3, 0, 1, 2).contiguous().half(),

            # [4] — coordenada q do target (b_norm, gx, gy, gz)
            "target_query": torch.tensor(
                [target_bval / bval_norm, *target_v], dtype=torch.float16
            ),

            # Máscaras
            "mask":   torch.from_numpy(patch_mask).unsqueeze(0).half(),
            "maskWM": torch.from_numpy(patch_maskWM).unsqueeze(0).half(),

            # Metadados
            "id":            subject_id,
            "origin_bval":   float(input_shell),
            "target_bval":   float(target_bval),
            "dwi_path":      dwi_path_nii,
            "plot_target_idx":      int(target_idx),
            "plot_neighbor_indices": torch.tensor(neighbor_indices[:self.k_neighbors]),
        }