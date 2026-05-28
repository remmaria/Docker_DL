import os
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

# =========================================================
# CONFIGURAÇÃO DO QC
# =========================================================
CSV_PATH = "/ix1/tibrahim/rmm270/DATA/DWIs/studies/COORDS_DL_new/master.csv"
QC_OUT_FOLDER = "/ix1/tibrahim/rmm270/DATA/DWIs/studies/COORDS_DL_new/QC_RESULTS"
N_SAMPLES_TO_VISUALIZE = 20  # Número de patches aleatórios para gerar imagens
PATCH_SIZE = 32
HALF = PATCH_SIZE // 2

os.makedirs(QC_OUT_FOLDER, exist_ok=True)


def load_middle_b0(dwi_path, bval_path):
    """Carrega o DWI e extrai o primeiro volume b0 encontrado para servir de background."""
    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()
    bvals = np.loadtxt(bval_path)

    # Tenta pegar o primeiro b0 (b-value < 50)
    b0_indices = np.where(bvals < 50)[0]
    if len(b0_indices) > 0:
        return dwi_data[..., b0_indices[0]]
    else:
        # Se não achar b0, pega o primeiro volume mesmo
        return dwi_data[..., 0]


# =========================================================
# 1. ANÁLISE DE MÉTRICAS E ESTATÍSTICAS
# =========================================================
print("📊 Lendo o arquivo master.csv...", flush=True)
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("\n=== METRICAS GERAIS DELA ESTRUTURA DOS PATCHES ===")
total_patches = len(df)
total_subjects = df["SessionID"].nunique()
print(f"• Total de patches gerados: {total_patches}")
print(f"• Total de sujeitos únicos : {total_subjects}")
print(
    f"• Média de patches por sujeito: {total_patches / total_subjects:.2f}"
)

# Patches por sujeito (identificar outliers/falhas de segmentação)
patches_per_sub = df["SessionID"].value_counts()
print("\n• Distribuição de Patches por Sujeito:")
print(patches_per_sub.describe())

# Verificar se há protocolos misturados
print("\n• Distribuição por Protocolo de DWI:")
print(df["protocol"].value_counts())

# Salva relatório de métricas em texto
with open(f"{QC_OUT_FOLDER}/metrics_report.txt", "w") as f:
    f.write(f"Total Patches: {total_patches}\n")
    f.write(f"Total Subjects: {total_subjects}\n")
    f.write(f"Patches/Subj Descriptives:\n{patches_per_sub.describe().to_string()}\n")

# Histograma de distribuição de patches
plt.figure(figsize=(8, 4))
plt.hist(patches_per_sub, bins=20, color="skyblue", edgecolor="black")
plt.title("Distribuição do Número de Patches por Sujeito")
plt.xlabel("Quantidade de Patches")
plt.ylabel("Frequência (Sujeitos)")
plt.grid(axis="y", alpha=0.75)
plt.savefig(f"{QC_OUT_FOLDER}/patches_distribution_histogram.png", dpi=150)
plt.close()


# =========================================================
# 2. GERAÇÃO DE IMAGENS DE INSPEÇÃO VISUAL (QC)
# =========================================================
print(
    f"\n🖼️ Gerando visualizações para {N_SAMPLES_TO_VISUALIZE} patches aleatórios...",
    flush=True,
)

# Sorteia linhas aleatórias do CSV para validar
sampled_df = df.sample(n=min(N_SAMPLES_TO_VISUALIZE, total_patches)).copy()

for idx, row in sampled_df.iterrows():
    subject = row["SessionID"]
    cx, cy, cz = int(row["center_x"]), int(row["center_y"]), int(row["center_z"])

    print(f"   ↳ Renderizando Patch do {subject} nas coordenadas ({cx}, {cy}, {cz})")

    try:
        # Carrega dados médicos correspondentes
        background_b0 = load_middle_b0(row["dwi_path"], row["bval_path"])
        mask_data = nib.load(row["mask_path"]).get_fdata() > 0

        # Define os limites do bounding box do patch
        xs, xe = cx - HALF, cx + HALF
        ys, ye = cy - HALF, cy + HALF
        zs, ze = cz - HALF, cz + HALF

        # Cria a figura com 3 visões (Axial, Sagital, Coronal)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Subject: {subject} | Center: ({cx}, {cy}, {cz})",
            fontsize=14,
            fontweight="bold",
        )

        # -------------------------------------------------
        # VISÃO AXIAL (Plano XY na fatia Z central)
        # -------------------------------------------------
        ax = axes[0]
        ax.imshow(background_b0[:, :, cz].T, cmap="gray", origin="lower")
        # Desenha a bounding box do patch
        rect = plt.Rectangle(
            (xs, ys),
            PATCH_SIZE,
            PATCH_SIZE,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
            label="Patch Bounds",
        )
        ax.add_patch(rect)
        ax.plot(cx, cy, "ro", markersize=4)  # Centro
        ax.set_title(f"Axial (Z={cz})")
        ax.axis("off")

        # -------------------------------------------------
        # VISÃO SAGITAL (Plano YZ na fatia X central)
        # -------------------------------------------------
        ax = axes[1]
        ax.imshow(background_b0[cx, :, :].T, cmap="gray", origin="lower")
        rect = plt.Rectangle(
            (ys, zs),
            PATCH_SIZE,
            PATCH_SIZE,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.plot(cy, cz, "ro", markersize=4)
        ax.set_title(f"Sagital (X={cx})")
        ax.axis("off")

        # -------------------------------------------------
        # VISÃO CORONAL (Plano XZ na fatia Y central)
        # -------------------------------------------------
        ax = axes[2]
        ax.imshow(background_b0[:, cy, :].T, cmap="gray", origin="lower")
        rect = plt.Rectangle(
            (xs, zs),
            PATCH_SIZE,
            PATCH_SIZE,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.plot(cx, cz, "ro", markersize=4)
        ax.set_title(f"Coronal (Y={cy})")
        ax.axis("off")

        # Ajusta layout e salva
        plt.tight_layout()
        output_img_path = (
            f"{QC_OUT_FOLDER}/QC_{subject}_c{cx}_{cy}_{cz}.png"
        )
        plt.savefig(output_img_path, bbox_inches="tight", dpi=100)
        plt.close()

    except Exception as e:
        print(f"   ❌ Erro ao gerar imagem para o patch {idx}: {e}")

print(f"\n✅ Controle de qualidade concluído! Resultados salvos em: {QC_OUT_FOLDER}")
print(
    f"Verifique o histograma e as imagens geradas para validar visualmente o alinhamento."
)