import sys
import os
import numpy as np
import nibabel as nib

def convert_subject(folder_path):
    # Dicionário de arquivos e seus tipos ideais
    targets = {
        #"bgpdwis_PA_geomcorr.nii": np.float32,
        "bgpdwis_PA_geomcorr_mask3d.nii.gz": np.uint8,
    }
    
    print(f"📂 Processando: {os.path.basename(folder_path)}")
    
    for filename, dtype_target in targets.items():
        nii_path = os.path.join(folder_path, filename)
        
        if os.path.exists(nii_path):
            basename = filename.replace('.gz', '').replace('.nii', '')
            npy_path = os.path.join(folder_path, f"{basename}.npy")
            if not os.path.exists(npy_path):
                try:
                    img = nib.load(nii_path)
                    # get_fdata sempre em float32 primeiro para evitar erro de precisão
                    data = img.get_fdata(dtype=np.float32)
                    
                    # ========================================================
                    # CORREÇÃO AQUI: Transpõe de (X, Y, Z, D) para (D, X, Y, Z)
                    # ========================================================
                    if len(data.shape) > 3:
                        data = np.transpose(data, (3, 0, 1, 2))
                    
                    # Converte para o tipo econômico e salva
                    np.save(npy_path, data.astype(dtype_target))
                    print(f"   ✅ {filename} -> {basename}.npy ({dtype_target}) com shape {data.shape}")
                except Exception as e:
                    print(f"   ❌ Erro em {filename}: {e}")
        else:
            print(f"   ⚠️ Arquivo não encontrado: {filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_subject(sys.argv[1])