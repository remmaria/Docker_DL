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
                 min_tile_coverage: float = 0.0, sh_q_out: int = 0, ensemble_m: int = 0,
                 log_worker_loads: bool = False):
        """
        only_valid: quando True (default), so usa trincas com `valid=True`
            (residuo de colinearidade dentro do teto usado em
            02b_build_rrin_triplets.py --max-residual-deg) -- treinar em
            trincas geometricamente sem sentido so ensinaria a rede a
            "chutar" nesses casos, sem sinal real de fluxo pra aprender.
            Sujeitos sem NENHUMA trinca valida para este (shell,n_level)
            sao descartados do dataset inteiro (ver __init__).

        sh_q_out: (default 0 = desligado, comportamento identico a antes)
            quando > 0, cada item TAMBEM devolve um "feixe" de ate
            `sh_q_out` trincas adicionais do MESMO sujeito, na MESMA
            posicao espacial (ox,oy,oz) do item principal -- usado por
            scripts/04b_train_rrin.py para montar o termo de loss
            angular/SH (ver utils/sh_angular_loss.py e protocolo secao
            14.5/15), que precisa de VARIAS direcoes-alvo simultaneas por
            "exemplo" pra ajustar uma base SH (analogo ao q_out do RCAE,
            mas aqui cada trinca do feixe ainda passa pelo RRIN uma de
            cada vez -- o feixe so agrupa as previsoes DEPOIS, no script de
            treino, pra rodar a mesma compute_sh_angular_loss do RCAE sem
            modifica-la). Se o sujeito tiver menos de `sh_q_out` trincas
            validas disponiveis, o feixe e preenchido ate onde der e o
            restante marcado invalido em "sh_mask" (compute_sh_angular_loss
            ja sabe ignorar posicoes invalidas, mesmo mecanismo do padding
            de collate_variable_targets no RCAE).

        ensemble_m: (default 0 = desligado, comportamento identico a antes)
            quando > 0, cada item TAMBEM devolve um feixe de ate
            `ensemble_m` pares de entrada DIVERSOS para a MESMA
            direcao-alvo do item principal -- nao um feixe de trincas
            diferentes, como sh_q_out (ver acima), mas MULTIPLOS PARES
            candidatos pra UM SO alvo (ver protocolo secao 14.5 item 1/
            addendum 2026-08-27, "ensemble em estrela", e
            utils/gradients.py:find_star_ensemble_batch/
            model/rrin3d_star.py:RRIN3DStar, que funde as M predicoes por
            um softmax aprendido). Exige que o `<tag>_rrin_triplets.npz`
            tenha sido gerado com `scripts/02b_build_rrin_triplets.py
            --ensemble-m <M>` (M pode ser >= ensemble_m; usamos so os
            primeiros `ensemble_m` slots do feixe gravado) -- levanta
            RuntimeError cedo e com mensagem clara se os campos `ens_*`
            nao existirem no npz, em vez de KeyError confuso mais tarde
            dentro do DataLoader. Um sujeito pode ter menos de
            `ensemble_m` pares REAIS disponiveis para um dado alvo (a
            mascara `ens_valid` gravada por 02b ja marca isso) -- as
            posicoes sem par real ficam zeradas e `ensemble_mask=False`,
            mesmo mecanismo de padding de sh_mask/target_mask.

        log_worker_loads: (default False, ADITIVO -- ver mesmo parametro
            em utils/pairflow_ssl_dataset.py:PairFlowSSLDataset, adicionado
            junto no diagnostico de gargalo de dataloading de 2026-09-02)
            quando True, imprime `worker_id`/`subject_tag` a cada carga
            REAL de disco (cache MISS) em `_load_subject` -- usado pra
            confirmar se o mesmo sujeito esta sendo recarregado por
            multiplos workers na mesma epoca (ver
            utils.dataset.SubjectGroupedSampler.__init__ para o mecanismo
            suspeito).
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
        self.sh_q_out = sh_q_out
        self.ensemble_m = ensemble_m
        self.log_worker_loads = log_worker_loads
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
            if ensemble_m > 0 and f"{key}__ens_pair_a" not in trip.files:
                raise RuntimeError(
                    f"ensemble_m={ensemble_m} pedido, mas {trip_path} nao tem os campos "
                    f"'{key}__ens_pair_a' etc. -- rode scripts/02b_build_rrin_triplets.py de "
                    f"novo com --ensemble-m >= {ensemble_m} (aditivo, nao precisa apagar nada "
                    f"do que ja existe, so acrescenta os campos '__ens_*' novos)."
                )
            if ensemble_m > 0:
                # BUG CORRIGIDO (2026-08-28): fatiar `[:, :ensemble_m]` num
                # array com MENOS colunas que o M pedido nao levanta erro em
                # numpy (so devolve as colunas que existem) -- isso deixava o
                # mismatch passar batido aqui e so estourar bem mais tarde,
                # num IndexError criptico dentro de um worker do DataLoader
                # (`_ensemble_tensors`, ao tentar acessar um slot que nao
                # existe). Checagem early e explicita: o `.npz` precisa ter
                # sido gravado com --ensemble-m >= ensemble_m pedido aqui.
                actual_m = trip[f"{key}__ens_pair_a"].shape[1]
                if actual_m < ensemble_m:
                    raise RuntimeError(
                        f"ensemble_m={ensemble_m} pedido, mas {trip_path} foi gravado com "
                        f"apenas M={actual_m} pares no feixe (campo '{key}__ens_pair_a' tem "
                        f"shape (.,{actual_m})) -- rode scripts/02b_build_rrin_triplets.py de "
                        f"novo com --ensemble-m >= {ensemble_m} apontando pro MESMO --out-dir "
                        f"usado aqui (--triplets-dir), sobrescrevendo o .npz com o M maior."
                    )
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

        if self.log_worker_loads:
            # Mesmo diagnostico de utils/pairflow_ssl_dataset.py:
            # PairFlowSSLDataset._load_subject (2026-09-02) -- ver
            # docstring la e de SubjectGroupedSampler.__init__ pra o
            # mecanismo suspeito (redundancia de carga entre workers).
            worker_info = torch.utils.data.get_worker_info()
            wid = worker_info.id if worker_info is not None else -1
            print(f"[rrin_dataset][worker-load] worker={wid} subject={tag}", flush=True)

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

        ens_pair_a = ens_pair_b = ens_t_frac = ens_residual_deg = ens_gap_deg = ens_valid = None
        if self.ensemble_m > 0:
            # ADITIVO -- so le/filtra estes campos se ensemble_m>0 (ver
            # __init__, ja confirmou que existem). Slots alem de ensemble_m
            # no npz (se M gravado por 02b > ensemble_m pedido aqui) sao
            # simplesmente ignorados.
            ens_pair_a = trip[f"{key}__ens_pair_a"][:, : self.ensemble_m]
            ens_pair_b = trip[f"{key}__ens_pair_b"][:, : self.ensemble_m]
            ens_t_frac = trip[f"{key}__ens_t_frac"][:, : self.ensemble_m]
            ens_residual_deg = trip[f"{key}__ens_residual_deg"][:, : self.ensemble_m]
            ens_gap_deg = trip[f"{key}__ens_gap_deg"][:, : self.ensemble_m]
            ens_valid = trip[f"{key}__ens_valid"][:, : self.ensemble_m]

        if self.only_valid:
            keep = valid
            target_idx = target_idx[keep]
            pair_a = pair_a[keep]
            pair_b = pair_b[keep]
            t_frac = t_frac[keep]
            residual_deg = residual_deg[keep]
            gap_deg = gap_deg[keep]
            if self.ensemble_m > 0:
                ens_pair_a = ens_pair_a[keep]
                ens_pair_b = ens_pair_b[keep]
                ens_t_frac = ens_t_frac[keep]
                ens_residual_deg = ens_residual_deg[keep]
                ens_gap_deg = ens_gap_deg[keep]
                ens_valid = ens_valid[keep]
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
            "ens_pair_a": ens_pair_a,
            "ens_pair_b": ens_pair_b,
            "ens_t_frac": ens_t_frac,
            "ens_residual_deg": ens_residual_deg,
            "ens_gap_deg": ens_gap_deg,
            "ens_valid": ens_valid,
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

    def _triplet_tensors(self, d, k, ox, oy, oz, mask_patch, xmax):
        """Extrai (vol_a, vol_b, target, bvec_a, bvec_b, bvec_t, t_frac,
        quality) para UMA trinca `k` do sujeito `d`, na posicao espacial
        (ox,oy,oz) -- fatorado de __getitem__ para ser reusado tanto pelo
        item principal quanto pelo feixe `sh_q_out` (mesma logica, so muda
        qual `k`/posicao e passado)."""
        a_idx, b_idx, t_idx = int(d["pair_a"][k]), int(d["pair_b"][k]), int(d["target_idx"][k])
        t_frac = float(d["t_frac"][k])
        # normalizados por 90 (maximo possivel com simetria antipodal, ver
        # utils/gradients.py) -- usados so quando RRIN3D(use_quality_cond=True)
        # (ver model/rrin3d.py); sempre calculados aqui (custo desprezivel),
        # quem nao usar so ignora o campo "quality" do item.
        quality = np.array([d["residual_deg"][k] / 90.0, d["gap_deg"][k] / 90.0],
                            dtype=np.float32)

        vol_a = (self._extract(d["dwi"][..., [a_idx]], ox, oy, oz) / xmax) * mask_patch
        vol_b = (self._extract(d["dwi"][..., [b_idx]], ox, oy, oz) / xmax) * mask_patch
        target = (self._extract(d["dwi"][..., [t_idx]], ox, oy, oz) / xmax) * mask_patch

        # (ps,ps,ps,1) -> (1,ps,ps,ps)
        vol_a = np.moveaxis(vol_a, -1, 0).astype(np.float32)
        vol_b = np.moveaxis(vol_b, -1, 0).astype(np.float32)
        target = np.moveaxis(target, -1, 0).astype(np.float32)

        return {
            "vol_a": vol_a, "vol_b": vol_b, "target": target,
            "bvec_a": d["bvecs"][a_idx], "bvec_b": d["bvecs"][b_idx],
            "bvec_t": d["bvecs"][t_idx], "t_frac": t_frac, "quality": quality,
        }

    def _ensemble_tensors(self, d, k, ox, oy, oz, mask_patch, xmax):
        """Extrai o feixe "em estrela" de ate `self.ensemble_m` pares de
        entrada DIVERSOS para a MESMA direcao-alvo da trinca `k` (ver
        __init__/ensemble_m e protocolo secao 14.5 item 1/addendum
        2026-08-27) -- ao contrario de _triplet_tensors (que extrai UM par),
        aqui vol_a/vol_b/bvec_a/bvec_b variam entre as M posicoes do feixe,
        mas o alvo (target/bvec_t) e o MESMO em todas (e' o mesmo alvo,
        varios pares candidatos pra prever ELE). Posicoes de padding (sem
        par real disponivel -- ver ens_valid) ficam com tensores zerados e
        `ensemble_mask[slot]=False`, mesmo mecanismo do sh_mask/target_mask
        ja usados no resto do modulo."""
        M = self.ensemble_m
        t_idx = int(d["target_idx"][k])
        ps = self.patch_size
        vol_a_ens = np.zeros((M, 1, ps, ps, ps), dtype=np.float32)
        vol_b_ens = np.zeros((M, 1, ps, ps, ps), dtype=np.float32)
        bvec_a_ens = np.zeros((M, 3), dtype=np.float32)
        bvec_b_ens = np.zeros((M, 3), dtype=np.float32)
        bvec_t_ens = np.zeros((M, 3), dtype=np.float32)
        t_frac_ens = np.zeros((M,), dtype=np.float32)
        quality_ens = np.zeros((M, 2), dtype=np.float32)
        ensemble_mask = np.zeros((M,), dtype=bool)

        bvec_t = d["bvecs"][t_idx]
        # NAO reextraimos o volume-alvo aqui -- e' o MESMO alvo do item
        # principal ("target", ja calculado por _triplet_tensors em
        # __getitem__), reextrair so duplicaria IO/memoria a toa.

        for slot in range(M):
            if not bool(d["ens_valid"][k, slot]):
                continue  # padding -- ja zerado/False acima
            a_idx = int(d["ens_pair_a"][k, slot])
            b_idx = int(d["ens_pair_b"][k, slot])
            if a_idx < 0 or b_idx < 0:
                continue  # defensivo -- ens_valid ja deveria cobrir isso
            vol_a_patch = (self._extract(d["dwi"][..., [a_idx]], ox, oy, oz) / xmax) * mask_patch
            vol_b_patch = (self._extract(d["dwi"][..., [b_idx]], ox, oy, oz) / xmax) * mask_patch
            vol_a_ens[slot] = np.moveaxis(vol_a_patch, -1, 0).astype(np.float32)
            vol_b_ens[slot] = np.moveaxis(vol_b_patch, -1, 0).astype(np.float32)
            bvec_a_ens[slot] = d["bvecs"][a_idx]
            bvec_b_ens[slot] = d["bvecs"][b_idx]
            bvec_t_ens[slot] = bvec_t
            t_frac_ens[slot] = float(d["ens_t_frac"][k, slot])
            quality_ens[slot] = [d["ens_residual_deg"][k, slot] / 90.0,
                                  d["ens_gap_deg"][k, slot] / 90.0]
            ensemble_mask[slot] = True

        return {
            "vol_a_ens": vol_a_ens, "vol_b_ens": vol_b_ens,
            "bvec_a_ens": bvec_a_ens, "bvec_b_ens": bvec_b_ens, "bvec_t_ens": bvec_t_ens,
            "t_frac_ens": t_frac_ens, "quality_ens": quality_ens,
            "ensemble_mask": ensemble_mask,
        }

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

        mask_patch = self._extract(d["mask"].astype(np.float32), ox, oy, oz)[..., None]
        xmax = d["xmax"]
        main = self._triplet_tensors(d, k, ox, oy, oz, mask_patch, xmax)

        item = {
            "vol_a": torch.from_numpy(main["vol_a"]),
            "vol_b": torch.from_numpy(main["vol_b"]),
            "target": torch.from_numpy(main["target"]),
            "bvec_a": torch.from_numpy(main["bvec_a"]),
            "bvec_b": torch.from_numpy(main["bvec_b"]),
            "bvec_t": torch.from_numpy(main["bvec_t"]),
            "t_frac": torch.tensor(main["t_frac"], dtype=torch.float32),
            "quality": torch.from_numpy(main["quality"]),
            "subject_tag": tag,
        }

        if self.ensemble_m > 0:
            ens = self._ensemble_tensors(d, k, ox, oy, oz, mask_patch, xmax)
            item.update({
                "vol_a_ens": torch.from_numpy(ens["vol_a_ens"]),
                "vol_b_ens": torch.from_numpy(ens["vol_b_ens"]),
                "bvec_a_ens": torch.from_numpy(ens["bvec_a_ens"]),
                "bvec_b_ens": torch.from_numpy(ens["bvec_b_ens"]),
                "bvec_t_ens": torch.from_numpy(ens["bvec_t_ens"]),
                "t_frac_ens": torch.from_numpy(ens["t_frac_ens"]),
                "quality_ens": torch.from_numpy(ens["quality_ens"]),
                "ensemble_mask": torch.from_numpy(ens["ensemble_mask"]),
            })
            # o alvo do feixe e' o MESMO "target"/"bvec_t" do item principal
            # (main, acima) -- ver docstring de _ensemble_tensors -- entao
            # nao ha um "target_ens" separado no item.

        if self.sh_q_out > 0:
            K = self.sh_q_out
            n_valid_sh = min(K, n_triplets)
            if self.training:
                sh_idxs = self._rng.choice(n_triplets, size=n_valid_sh, replace=False)
            else:
                # deterministico e reprodutivel entre epocas (val_loss
                # comparavel), mas espalha o ponto de partida por item para
                # nao amostrar sempre as mesmas n_valid_sh primeiras trincas
                # em todo tile do mesmo sujeito.
                start = idx % n_triplets
                sh_idxs = np.unique((np.arange(n_valid_sh) + start) % n_triplets)
                if sh_idxs.size < n_valid_sh:  # colisao rara do modulo -- completa com arange
                    sh_idxs = np.arange(n_valid_sh)

            ps = self.patch_size
            vol_a_sh = np.zeros((K, 1, ps, ps, ps), dtype=np.float32)
            vol_b_sh = np.zeros((K, 1, ps, ps, ps), dtype=np.float32)
            target_sh = np.zeros((K, 1, ps, ps, ps), dtype=np.float32)
            bvec_a_sh = np.zeros((K, 3), dtype=np.float32)
            bvec_b_sh = np.zeros((K, 3), dtype=np.float32)
            bvec_t_sh = np.zeros((K, 3), dtype=np.float32)
            t_frac_sh = np.zeros((K,), dtype=np.float32)
            quality_sh = np.zeros((K, 2), dtype=np.float32)
            sh_mask = np.zeros((K,), dtype=bool)
            for slot, k2 in enumerate(sh_idxs):
                t = self._triplet_tensors(d, int(k2), ox, oy, oz, mask_patch, xmax)
                vol_a_sh[slot] = t["vol_a"]
                vol_b_sh[slot] = t["vol_b"]
                target_sh[slot] = t["target"]
                bvec_a_sh[slot] = t["bvec_a"]
                bvec_b_sh[slot] = t["bvec_b"]
                bvec_t_sh[slot] = t["bvec_t"]
                t_frac_sh[slot] = t["t_frac"]
                quality_sh[slot] = t["quality"]
                sh_mask[slot] = True

            item.update({
                "vol_a_sh": torch.from_numpy(vol_a_sh),
                "vol_b_sh": torch.from_numpy(vol_b_sh),
                "target_sh": torch.from_numpy(target_sh),
                "bvec_a_sh": torch.from_numpy(bvec_a_sh),
                "bvec_b_sh": torch.from_numpy(bvec_b_sh),
                "bvec_t_sh": torch.from_numpy(bvec_t_sh),
                "t_frac_sh": torch.from_numpy(t_frac_sh),
                "quality_sh": torch.from_numpy(quality_sh),
                "sh_mask": torch.from_numpy(sh_mask),
            })

        return item