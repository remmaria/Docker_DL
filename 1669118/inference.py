import torch
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import gc

sys.path.append("/ix1/tibrahim/rmm270/UTILITIES/env_dl")
import nibabel as nib
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.direction import peaks_from_model
from dipy.data import get_sphere
from skimage.metrics import peak_signal_noise_ratio as psnr
from monai.inferers import sliding_window_inference

from sub.select_pair_rep_centroid import sub_sample
from model import QSpaceAttentionNetwork


# ---------------------------------------------------------------------------
# UTILITÁRIOS
# ---------------------------------------------------------------------------

def get_protocol_string(bvals, tolerance=50):
    bvals_r = np.array([int(round(b / tolerance) * tolerance) for b in bvals])
    return "_".join([f"{s}-{np.sum(bvals_r == s)}" for s in sorted(np.unique(bvals_r))])


def get_neighbors_info(all_bvecs, all_bvals, target_v, target_b, idx_input, k_neighbors, bmax_norm):
    """
    Busca os K vizinhos angulares mais próximos dentro do sub-protocolo.
    Retorna os índices globais e as coordenadas q no formato [b_norm, gx, gy, gz].
    """
    idx_input = np.array(idx_input).flatten().astype(int)
    # Exclui b0s do pool de vizinhos (igual ao dataset.py: idx_diff = where(bvals > 50))
    idx_input = idx_input[all_bvals[idx_input] > 50]

    if len(idx_input) == 0:
        return [], np.zeros((0, 4), dtype=np.float32)

    pool_vecs_hemi = np.array([
        v if np.dot(v, target_v) >= 0 else -v for v in all_bvecs[idx_input]
    ])

    v_ref  = np.array([1, 0, 0]) if abs(target_v[0]) < 0.9 else np.array([0, 1, 0])
    orto_x = np.cross(target_v, v_ref)
    orto_x /= (np.linalg.norm(orto_x) + 1e-8)
    orto_y = np.cross(target_v, orto_x)

    angulos = np.degrees(np.arctan2(
        np.dot(pool_vecs_hemi, orto_y),
        np.dot(pool_vecs_hemi, orto_x)
    )) % 360

    distancias = np.minimum(
        np.linalg.norm(all_bvecs[idx_input] - target_v, axis=1),
        np.linalg.norm(all_bvecs[idx_input] + target_v, axis=1)
    )

    bins = np.linspace(0, 360, k_neighbors + 1)
    neighbor_indices = []
    for i in range(len(bins) - 1):
        mask_q = (angulos >= bins[i]) & (angulos < bins[i + 1])
        if np.any(mask_q):
            idx_q = np.where(mask_q)[0]
            best  = idx_q[np.argmin(distancias[idx_q])]
            neighbor_indices.append(idx_input[best])

    if len(neighbor_indices) < k_neighbors:
        faltam       = k_neighbors - len(neighbor_indices)
        mask_valida  = np.ones(len(idx_input), dtype=bool)
        for i, val in enumerate(idx_input):
            if val in neighbor_indices:
                mask_valida[i] = False
        if np.any(mask_valida):
            sobras = idx_input[mask_valida]
            extras = sobras[np.argsort(distancias[mask_valida])]
            neighbor_indices.extend(extras[:faltam].tolist())

    neighbor_indices = neighbor_indices[:k_neighbors]

    # FORMATO CORRETO: [b_norm, gx, gy, gz] — igual ao dataset.py
    n_coords = [
        [all_bvals[idx] / bmax_norm,
         all_bvecs[idx][0], all_bvecs[idx][1], all_bvecs[idx][2]]
        for idx in neighbor_indices
    ]

    return neighbor_indices, np.array(n_coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# DTI
# ---------------------------------------------------------------------------

def Quick_DTI(path_nii, path_bval, path_bvec, out_prefix, inf_folder):
    img   = nib.load(path_nii)
    data  = img.get_fdata().astype(np.float32)
    gtab  = gradient_table(path_bval, path_bvec)

    tenmodel = dti.TensorModel(gtab)
    tenfit   = tenmodel.fit(data)

    fa = tenfit.fa.astype(np.float32)
    v1 = tenfit.evecs[..., 0].astype(np.float32)

    nib.save(nib.Nifti1Image(fa, img.affine), f"{inf_folder}/{out_prefix}_FA.nii.gz")
    nib.save(nib.Nifti1Image(v1, img.affine), f"{inf_folder}/{out_prefix}_V1.nii.gz")

    del data, tenmodel, tenfit
    gc.collect()
    return fa, v1


# ---------------------------------------------------------------------------
# CSD
# ---------------------------------------------------------------------------

def Get_CSD_Peaks_Optimized(path_nii, path_bval, path_bvec, mask_data):
    """Calcula picos CSD em fatias para economizar memória."""
    img  = nib.load(path_nii)
    data = img.get_fdata().astype(np.float32)
    gtab = gradient_table(path_bval, path_bvec)

    H, W, D, _ = data.shape

    # Verifica consistência de shape com a máscara ANTES de processar
    assert mask_data.shape == (H, W, D), (
        f"Shape da máscara {mask_data.shape} não bate com o volume {(H, W, D)}. "
        "Certifique-se de usar o volume GT cortado (vol_GT_cropped) e não o original."
    )

    peak_dirs = np.zeros((H, W, D, 3), dtype=np.float32)

    print("  Estimando função de resposta...")
    response, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    csd_model   = ConstrainedSphericalDeconvModel(gtab, response)
    sphere      = get_sphere('repulsion724')

    SLICE_BATCH = 5
    for z_start in range(0, D, SLICE_BATCH):
        z_end = min(z_start + SLICE_BATCH, D)
        print(f"  Processando fatias {z_start}-{z_end}/{D}", end='\r')

        data_slice = data[:, :, z_start:z_end, :].copy()
        mask_slice = mask_data[:, :, z_start:z_end].copy()

        csd_peaks = peaks_from_model(
            model=csd_model,
            data=data_slice,
            sphere=sphere,
            relative_peak_threshold=.5,
            min_separation_angle=25,
            mask=mask_slice,
            return_odf=False,
            return_sh=False,
            normalize_peaks=True,
            parallel=False,
        )
        peak_dirs[:, :, z_start:z_end, :] = csd_peaks.peak_dirs[..., 0, :]

        del data_slice, mask_slice, csd_peaks
        gc.collect()

    print()
    del data, csd_model
    gc.collect()
    return peak_dirs


def Compare_CSD_Metrics_Optimized(gt_peaks, test_peaks, mask, name):
    H, W, D, _ = gt_peaks.shape
    angular_errors = []

    BATCH_SIZE = 10
    for z_start in range(0, D, BATCH_SIZE):
        z_end = min(z_start + BATCH_SIZE, D)

        gt_batch   = gt_peaks[:, :, z_start:z_end, :]
        test_batch = test_peaks[:, :, z_start:z_end, :]
        mask_batch = mask[:, :, z_start:z_end]

        dot = np.abs(np.einsum('ijkl,ijkl->ijk', gt_batch, test_batch))
        dot = np.clip(dot, 0, 1)
        angular_error = np.degrees(np.arccos(dot))

        angular_errors.append(angular_error[mask_batch])

        del gt_batch, test_batch, mask_batch, dot, angular_error
        gc.collect()

    all_errors = np.concatenate(angular_errors)
    print(f"--- CSD Angular Error ({name}) ---")
    print(f"Erro Médio:   {np.mean(all_errors):.2f}°")
    print(f"Erro Mediano: {np.median(all_errors):.2f}°")
    return all_errors


# ---------------------------------------------------------------------------
# MÉTRICAS E PLOTS
# ---------------------------------------------------------------------------

def Calculate_Success_Rate(angular_error_array, threshold=15.0):
    return (np.sum(angular_error_array < threshold) / len(angular_error_array)) * 100


def Plot_Error_Distribution(errors_dict, folder, region_name):
    plt.figure(figsize=(10, 6))
    colors    = {'DL': '#2ca02c', 'SUB': '#d62728', 'MEDIA': '#ff7f0e'}
    threshold = 15.0 if "White Matter" in region_name else 25.0

    for name, data in errors_dict.items():
        success_rate = np.mean(data < threshold) * 100
        mediana      = np.median(data)
        plt.hist(data, bins=100, range=(0, 60), density=True,
                 histtype='step', linewidth=2, color=colors[name],
                 label=f"{name} (SR: {success_rate:.1f}%, Med: {mediana:.1f}°)")

    plt.axvline(threshold, color='black', linestyle='--', alpha=0.5)
    plt.title(f"Distribuição do Erro Angular - {region_name}")
    plt.xlabel("Erro Angular (Graus)")
    plt.ylabel("Densidade de Voxels")
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{folder}/Histograma_{region_name.replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close()


def get_mask_WM(fa_data, threshold=0.4):
    from scipy import ndimage
    mask = (fa_data > threshold).astype(np.uint8)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
    return mask


def Compare_Maps(gt_fa, test_fa, gt_v1, test_v1, mask, name, affine, inf_folder):
    rmse_fa  = np.sqrt(np.mean((gt_fa[mask] - test_fa[mask]) ** 2))
    psnr_val = psnr(gt_fa[mask], test_fa[mask], data_range=1.0)

    dot           = np.abs(np.einsum('ijkl,ijkl->ijk', gt_v1, test_v1))
    dot           = np.clip(dot, 0, 1)
    angular_error = np.degrees(np.arccos(dot))
    mean_ang      = np.mean(angular_error[mask])

    diff_map = np.abs(gt_fa - test_fa)
    nib.save(nib.Nifti1Image(diff_map.astype(np.float32), affine),
             f"{inf_folder}/DIFF_FA_{name}.nii.gz")
    nib.save(nib.Nifti1Image(angular_error.astype(np.float32), affine),
             f"{inf_folder}/DIFF_ANGULAR_{name}.nii.gz")

    print(f"\n--- Métricas: GT vs {name} ---")
    print(f"FA RMSE:              {rmse_fa:.4f}")
    print(f"FA PSNR:              {psnr_val:.2f} dB")
    print(f"V1 Erro Angular Médio: {mean_ang:.2f}°")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', type=str, default='local')
args   = parser.parse_args()

job_id     = args.job_id
inf_folder = f'inf_debug/{job_id}'
os.makedirs(inf_folder, exist_ok=True)
print(f"Job ID: {job_id}", flush=True)

# --- CONFIGURAÇÕES ---
device          = torch.device("cuda")
checkpoint_path = "/ix1/tibrahim/rmm270/Docker/harm_dl/QSpaceAttentionNetwork_full2/checkpoints/1669031/model_best.pt"
K               = 8
alpha           = 2.0
bmax            = 3000.0
print(f"CHECKPOINT: {checkpoint_path}")
print(f"k:{K} | alpha:{alpha} | bmax:{bmax}")

# --- CARREGAR MODELO ---
model = QSpaceAttentionNetwork(k_neighbors=K).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- CARREGAR DADOS ---
BASE = "/ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias"
ID   = "20170831165760_603"
print(f"ID: {ID}")

sl_start = 2
sl_end   = -2

dwi_path      = f"{BASE}/{ID}/bgpdwis_PA_geomcorr.nii"
mask_path     = f"{BASE}/{ID}/bgpdwis_PA_geomcorr_mask3d.nii.gz"
maskWM_path   = f"{BASE}/{ID}/bgpdwis_PA_geomcorr_maskseg_WM_e1.nii.gz"
mask_CSO_path = f"{BASE}/{ID}/JHU-ICBM-CSO-1mm_warped_s.nii.gz"
bval_path     = f"{BASE}/{ID}/bgpdwis_PA_geomcorr.bval"
bvec_path     = f"{BASE}/{ID}/bgpdwis_PA_geomcorr.bvec"

img       = nib.load(dwi_path)
# FIX: full_data já é o volume CORTADO em Z — todas as comparações usam este mesmo volume
full_data = img.get_fdata()[:, :, sl_start:sl_end, :].astype(np.float32)

# Atualiza o affine para refletir o corte em Z
new_affine         = img.affine.copy()
new_affine[2, 3]  += sl_start * img.header.get_zooms()[2]

# Máscaras — todas cortadas no mesmo range de Z
mask_brain = nib.load(mask_path).get_fdata().astype(bool)[:, :, sl_start:sl_end]
mask_WM    = nib.load(maskWM_path).get_fdata().astype(bool)[:, :, sl_start:sl_end]
mask_CSO   = nib.load(mask_CSO_path).get_fdata().astype(bool)[:, :, sl_start:sl_end]

bvals = np.loadtxt(bval_path)
bvecs = np.loadtxt(bvec_path).T  # shape [N, 3]

subset_sub = '0-1_1000-32'

# --- SUB-PROTOCOLO ---
sub_indices, _, _ = sub_sample(dwi_path, bval_path, bvec_path,
                                get_protocol_string(bvals), subset_sub,
                                sub_folder="temp_sub")
print(f"Sub-protocolo selecionado: {len(sub_indices)} direções.")

# --- B0 MEAN (calculado uma vez sobre o volume completo cortado) ---
idx_b0      = np.where(bvals < 50)[0]
mean_b0_vol = np.mean(full_data[..., idx_b0], axis=-1)  # [H, W, D]
mean_b0_vol = np.maximum(mean_b0_vol, 1e-8)

# =====================================================================
# GERAR VOLUME GT CORTADO (para CSD e DTI comparáveis)
# FIX: sem isso o GT teria D fatias enquanto máscaras têm D-4 → crash no CSD
# =====================================================================
gt_cropped_path = f"{inf_folder}/vol_GT_cropped.nii.gz"
nib.save(nib.Nifti1Image(full_data, new_affine), gt_cropped_path)
print("✅ GT cortado salvo.")

# =====================================================================
# GERAR VOLUME E DTI DO SUBSET (SUB)
# =====================================================================
print("➡️ Gerando volume do protocolo reduzido (SUB)...")

all_sub_idx = np.unique(np.concatenate((idx_b0, sub_indices))).astype(int)
sub_data    = full_data[..., all_sub_idx]
sub_bvals   = bvals[all_sub_idx]
# bvecs tem shape [N, 3] após o .T no carregamento
# Para salvar no formato FSL (3 × N), transpomos
sub_bvecs_save = bvecs[all_sub_idx, :].T  # shape [3, n_sub]

sub_bval_path = f"{inf_folder}/sub.bval"
sub_bvec_path = f"{inf_folder}/sub.bvec"
np.savetxt(sub_bval_path, sub_bvals[None, :], fmt="%d")       # [1, n_sub]
np.savetxt(sub_bvec_path, sub_bvecs_save,     fmt="%.6f")     # [3, n_sub]

sub_img_path = f"{inf_folder}/vol_SUB.nii.gz"
nib.save(nib.Nifti1Image(sub_data.astype(np.float32), new_affine), sub_img_path)
sub_fa, sub_v1 = Quick_DTI(sub_img_path, sub_bval_path, sub_bvec_path, "SUB", inf_folder)

del sub_data
gc.collect()

# =====================================================================
# LOOP DE RECONSTRUÇÃO 4D
# =====================================================================
final_4d = []
media_4d  = []

for i in range(len(bvals)):
    target_q_vec = bvecs[i]       # shape [3]
    target_b_val = bvals[i]

    # b0s passam direto — o modelo não foi treinado para predizê-los
    if target_b_val < 50:
        final_4d.append(full_data[..., i].copy())
        media_4d.append(full_data[..., i].copy())
        continue

    # A. Busca vizinhos
    neighbor_idx_in_full, n_coords_raw = get_neighbors_info(
        bvecs, bvals, target_q_vec, target_b_val, sub_indices, K, bmax
    )

    if len(neighbor_idx_in_full) < K:
        # Fallback: copia o volume mais próximo disponível
        print(f"⚠️  Direção {i} (b={target_b_val}): apenas {len(neighbor_idx_in_full)} vizinhos, pulando modelo.")
        fallback = full_data[..., neighbor_idx_in_full[0]] if neighbor_idx_in_full else full_data[..., i]
        final_4d.append(fallback.copy())
        media_4d.append(fallback.copy())
        continue

    # B. Tensores de coordenadas
    # FIX: formato [b_norm, gx, gy, gz] — igual ao dataset.py
    n_coords_tensor = torch.from_numpy(n_coords_raw).float().unsqueeze(0).to(device)  # [1, K, 4]
    target_q_tensor = torch.tensor([
        target_b_val / bmax,
        target_q_vec[0], target_q_vec[1], target_q_vec[2],
    ], dtype=torch.float32).unsqueeze(0).to(device)  # [1, 4]

    # C. Normalização dos vizinhos (igual ao dataset.py)
    volumes_vizinhos = []
    for idx in neighbor_idx_in_full:
        vol = full_data[..., idx] / (mean_b0_vol * alpha)
        vol = np.clip(vol, 0, 1).astype(np.float32)
        volumes_vizinhos.append(vol)

    # Média dos vizinhos desnormalizada (baseline para comparação)
    media_norm = np.mean(volumes_vizinhos, axis=0)
    media_4d.append((media_norm * mean_b0_vol * alpha).astype(np.float32))

    # D. Preditor MONAI
    # FIX: empacotamos os K vizinhos como [1, K, H, W, D] para o sliding_window.
    # O predictor recebe [sw_B, K, ph, pw, pd] e insere a dim de canal C=1.
    input_for_monai = (
        torch.from_numpy(np.stack(volumes_vizinhos))  # [K, H, W, D]
        .float()
        .unsqueeze(0)                                  # [1, K, H, W, D]
        .to(device)
    )

    def predictor(patch_from_monai):
        # patch_from_monai: [sw_B, K, ph, pw, pd]
        # Insere dim de canal C=1: [sw_B, K, 1, ph, pw, pd]
        p_neighbors = patch_from_monai.unsqueeze(2)
        sw_B        = p_neighbors.shape[0]
        q_exp       = target_q_tensor.expand(sw_B, -1)    # [sw_B, 4]
        n_exp       = n_coords_tensor.expand(sw_B, -1, -1) # [sw_B, K, 4]
        output, _, _ = model(p_neighbors, q_exp, n_exp)
        return output.float()  # [sw_B, 1, ph, pw, pd]

    # E. Inferência
    print(f"➡️  Reconstruindo direção {i+1}/{len(bvals)} (b={int(target_b_val)})", end="\r")
    with torch.no_grad():
        pred_3d = sliding_window_inference(
            inputs=input_for_monai,
            roi_size=(64, 64, 64),
            sw_batch_size=4,
            predictor=predictor,
            overlap=0.25,
            # FIX: mode="constant" evita bias gaussiano nas bordas dos patches.
            # Se quiser suavização, use "gaussian" mas saiba que pode introduzir
            # um shift sistemático se o modelo for inconsistente entre centro e borda.
            mode="constant",
        )

    # FIX: desnormalização correta — pred está em [0,1], volta para escala original
    pred_np = pred_3d.cpu().numpy().squeeze()  # [H, W, D]
    final_4d.append((pred_np * mean_b0_vol * alpha).astype(np.float32))

    del input_for_monai, pred_3d
    torch.cuda.empty_cache()

print()

# =====================================================================
# SALVAR VOLUMES
# =====================================================================
final_4d_array = np.stack(final_4d, axis=-1)
nib.save(nib.Nifti1Image(final_4d_array, new_affine), f"{inf_folder}/vol_DL.nii.gz")
print("✅ vol_DL.nii.gz salvo.")

media_4d_array = np.stack(media_4d, axis=-1)
media_path     = f"{inf_folder}/vol_MEDIA.nii.gz"
nib.save(nib.Nifti1Image(media_4d_array, new_affine), media_path)
print("✅ vol_MEDIA.nii.gz salvo.")

del final_4d, final_4d_array, media_4d, media_4d_array
gc.collect()
torch.cuda.empty_cache()

# =====================================================================
# DTI
# =====================================================================
print("\nCalculando DTI...")

# FIX: GT usa o volume CORTADO (gt_cropped_path), não o original
# para garantir shape idêntico ao DL e às máscaras
gt_fa,  gt_v1  = Quick_DTI(gt_cropped_path,            bval_path,    bvec_path,    "GT",    inf_folder)
dl_fa,  dl_v1  = Quick_DTI(f"{inf_folder}/vol_DL.nii.gz",  bval_path, bvec_path,   "DL",    inf_folder)
med_fa, med_v1 = Quick_DTI(media_path,                  bval_path,    bvec_path,    "MEDIA", inf_folder)

# sub_fa/sub_v1 já foram calculados acima e salvos
sub_fa = nib.load(f"{inf_folder}/SUB_FA.nii.gz").get_fdata().astype(np.float32)
sub_v1 = nib.load(f"{inf_folder}/SUB_V1.nii.gz").get_fdata().astype(np.float32)

# =====================================================================
# COMPARAÇÃO DTI
# =====================================================================
print("\nComparando DTI...")

for region_name, mask in [("Whole Brain", mask_brain),
                           ("White Matter", mask_WM),
                           ("Centrum Semiovale", mask_CSO)]:
    print(f"\n{'='*50}")
    print(region_name)
    Compare_Maps(gt_fa, dl_fa,  gt_v1, dl_v1,  mask, "DL",    new_affine, inf_folder)
    Compare_Maps(gt_fa, med_fa, gt_v1, med_v1, mask, "MEDIA", new_affine, inf_folder)
    Compare_Maps(gt_fa, sub_fa, gt_v1, sub_v1, mask, "SUB",   new_affine, inf_folder)

# =====================================================================
# CSD
# =====================================================================
print("\n" + "=" * 60)
print("CALCULANDO CSD PEAKS")
print("=" * 60)

nib.save(nib.Nifti1Image(mask_WM.astype(np.float32), new_affine),
         f"{inf_folder}/mask_WM.nii.gz")

# FIX: GT agora usa gt_cropped_path (shape idêntico às máscaras cortadas)
print("\n🧠 GT...")
gt_v1_csd = Get_CSD_Peaks_Optimized(gt_cropped_path, bval_path, bvec_path, mask_WM)

print("🧠 SUB...")
sub_v1_csd = Get_CSD_Peaks_Optimized(
    f"{inf_folder}/vol_SUB.nii.gz", sub_bval_path, sub_bvec_path, mask_WM
)
e_sub_wb  = Compare_CSD_Metrics_Optimized(gt_v1_csd, sub_v1_csd, mask_brain, "SUB - Whole Brain")
e_sub_wm  = Compare_CSD_Metrics_Optimized(gt_v1_csd, sub_v1_csd, mask_WM,    "SUB - White Matter")
e_sub_cso = Compare_CSD_Metrics_Optimized(gt_v1_csd, sub_v1_csd, mask_CSO,   "SUB - CSO")
del sub_v1_csd; gc.collect()

print("🧠 MEDIA...")
media_v1_csd = Get_CSD_Peaks_Optimized(
    f"{inf_folder}/vol_MEDIA.nii.gz", bval_path, bvec_path, mask_WM
)
e_med_wb  = Compare_CSD_Metrics_Optimized(gt_v1_csd, media_v1_csd, mask_brain, "MEDIA - Whole Brain")
e_med_wm  = Compare_CSD_Metrics_Optimized(gt_v1_csd, media_v1_csd, mask_WM,    "MEDIA - White Matter")
e_med_cso = Compare_CSD_Metrics_Optimized(gt_v1_csd, media_v1_csd, mask_CSO,   "MEDIA - CSO")
del media_v1_csd; gc.collect()

print("🧠 DL...")
dl_v1_csd = Get_CSD_Peaks_Optimized(
    f"{inf_folder}/vol_DL.nii.gz", bval_path, bvec_path, mask_WM
)
e_dl_wb  = Compare_CSD_Metrics_Optimized(gt_v1_csd, dl_v1_csd, mask_brain, "DL - Whole Brain")
e_dl_wm  = Compare_CSD_Metrics_Optimized(gt_v1_csd, dl_v1_csd, mask_WM,    "DL - White Matter")
e_dl_cso = Compare_CSD_Metrics_Optimized(gt_v1_csd, dl_v1_csd, mask_CSO,   "DL - CSO")
del dl_v1_csd, gt_v1_csd; gc.collect()

# =====================================================================
# TABELA FINAL CSD
# =====================================================================
print("\n" + "=" * 65)
print(f"{'📊 CSD ANGULAR ERROR COMPARISON':^65}")
print("=" * 65)
print(f"{'Region':<20} | {'SUB (32d)':>10} | {'MEDIA':>12} | {'DL (IA)':>10}")
print("-" * 65)

GREEN = '\033[92m'; YELLOW = '\033[93m'; END = '\033[0m'; BOLD = '\033[1m'

def print_row(region, sub, med, dl):
    dl_color = GREEN if (dl < sub and dl < med) else YELLOW
    print(f"{BOLD}{region:<20}{END} | {sub:>10.2f}° | {med:>12.2f}° | {dl_color}{dl:>10.2f}°{END}")

print_row("Whole Brain",       np.mean(e_sub_wb),  np.mean(e_med_wb),  np.mean(e_dl_wb))
print_row("White Matter",      np.mean(e_sub_wm),  np.mean(e_med_wm),  np.mean(e_dl_wm))
print_row("Centrum Semiovale", np.mean(e_sub_cso), np.mean(e_med_cso), np.mean(e_dl_cso))
print("=" * 65)

# =====================================================================
# TAXAS DE SUCESSO
# =====================================================================
for thresh, region_name, e_sub, e_med, e_dl in [
    (15, "WM",  e_sub_wm,  e_med_wm,  e_dl_wm),
    (10, "WM",  e_sub_wm,  e_med_wm,  e_dl_wm),
    (25, "CSO", e_sub_cso, e_med_cso, e_dl_cso),
    (20, "CSO", e_sub_cso, e_med_cso, e_dl_cso),
]:
    sr_sub = Calculate_Success_Rate(e_sub, thresh)
    sr_med = Calculate_Success_Rate(e_med, thresh)
    sr_dl  = Calculate_Success_Rate(e_dl,  thresh)
    print(f"\n📈 Taxa de Sucesso (erro < {thresh}° na {region_name}):")
    print(f"   SUB: {sr_sub:.1f}% | MEDIA: {sr_med:.1f}% | DL: {sr_dl:.1f}%")

# =====================================================================
# HISTOGRAMAS
# =====================================================================
Plot_Error_Distribution({'DL': e_dl_wm,  'SUB': e_sub_wm,  'MEDIA': e_med_wm},  inf_folder, "White Matter")
Plot_Error_Distribution({'DL': e_dl_cso, 'SUB': e_sub_cso, 'MEDIA': e_med_cso}, inf_folder, "Centrum Semiovale")

print("\n✅ Pronto!")