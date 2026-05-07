from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from scipy.ndimage import binary_erosion


# =========================================================
# CONFIG
# =========================================================

ROOT = Path("/ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias")

OUT_CSV = "/ix1/tibrahim/rmm270/DATA/DWIs/studies/COORDS_DL/master.csv"

PATCH_SIZE = 32
STRIDE = 16

HALF = PATCH_SIZE // 2

EROSION_ITERS = 2

MIN_MASK_RATIO = 0.5
# exige pelo menos 50% do patch dentro da máscara


# =========================================================
# PROTOCOL
# =========================================================

def extract_protocol_from_bval(
    bval_path,
    shell_tolerance=50,
):

    bvals = np.loadtxt(bval_path)

    rounded = (
        np.round(bvals / shell_tolerance)
        * shell_tolerance
    ).astype(int)

    # tudo abaixo de 50 vira b0
    rounded[rounded < 50] = 0

    protocol_parts = []

    for shell in sorted(np.unique(rounded)):

        n_dirs = np.sum(rounded == shell)

        protocol_parts.append(
            f"{shell}-{n_dirs}"
        )

    return "_".join(protocol_parts)


# =========================================================
# MAIN
# =========================================================

rows = []

subjects_ok = 0
subjects_failed = 0

for subj_dir in sorted(ROOT.iterdir()):

    if not subj_dir.is_dir():
        continue

    subject = subj_dir.name

    print(f"\n📦 Subject: {subject}", flush=True)

    try:

        dwi = next(
            subj_dir.glob("bgpdwis_PA_geomcorr.nii"),
            None
        )

        mask = next(
            subj_dir.glob("bgpdwis_PA_geomcorr_mask3d.nii.gz"),
            None
        )

        bval = next(
            subj_dir.glob("bgpdwis_PA_geomcorr.bval"),
            None
        )

        bvec = next(
            subj_dir.glob("bgpdwis_PA_geomcorr.bvec"),
            None
        )

        if any(x is None for x in [dwi, mask, bval, bvec]):

            print("❌ Missing files", flush=True)
            subjects_failed += 1
            continue

        # =================================================
        # LOAD DWI HEADER ONLY
        # =================================================

        dwi_img = nib.load(dwi)

        sx, sy, sz, ndwi = dwi_img.shape

        if len(dwi_img.shape) != 4:

            print(f"❌ Invalid DWI shape: {dwi_img.shape}")
            subjects_failed += 1
            continue

        # =================================================
        # BVAL VALIDATION
        # =================================================

        bvals = np.loadtxt(bval)

        if len(bvals) != ndwi:

            print(
                f"❌ bval mismatch | "
                f"DWIs={ndwi} "
                f"bvals={len(bvals)}"
            )

            subjects_failed += 1
            continue
        # =================================================
        # MASK
        # =================================================

        mask_data = (
            nib.load(mask)
            .get_fdata()
            > 0
        )

        # erosão
        mask_data = binary_erosion(
            mask_data,
            iterations=EROSION_ITERS,
        )

        shape = mask_data.shape

        # =================================================
        # PROTOCOL
        # =================================================

        protocol = extract_protocol_from_bval(
            bval
        )

        print(
            f"   shape={shape} "
            f"vols={ndwi} "
            f"protocol={protocol}",
            flush=True,
        )

        n_subject_patches = 0

        # =================================================
        # GRID
        # =================================================

        for x in range(
            HALF,
            shape[0] - HALF,
            STRIDE,
        ):

            for y in range(
                HALF,
                shape[1] - HALF,
                STRIDE,
            ):

                for z in range(
                    HALF,
                    shape[2] - HALF,
                    STRIDE,
                ):

                    # centro precisa estar na máscara
                    if not mask_data[x, y, z]:
                        continue

                    # patch bounds
                    xs = x - HALF
                    xe = x + HALF

                    ys = y - HALF
                    ye = y + HALF

                    zs = z - HALF
                    ze = z + HALF

                    # segurança extra
                    if (
                        xs < 0
                        or ys < 0
                        or zs < 0
                        or xe > shape[0]
                        or ye > shape[1]
                        or ze > shape[2]
                    ):
                        continue

                    # =================================================
                    # PATCH MASK COVERAGE
                    # =================================================

                    patch_mask = mask_data[
                        xs:xe,
                        ys:ye,
                        zs:ze,
                    ]

                    # garante shape correto
                    if patch_mask.shape != (
                        PATCH_SIZE,
                        PATCH_SIZE,
                        PATCH_SIZE,
                    ):
                        continue

                    # fração dentro da máscara
                    ratio = patch_mask.mean()

                    if ratio < MIN_MASK_RATIO:
                        continue

                    rows.append({

                        "subject": subject,

                        "dwi_path": str(dwi),

                        "bval_path": str(bval),

                        "bvec_path": str(bvec),

                        "mask_path": str(mask),

                        "center_x": int(x),

                        "center_y": int(y),

                        "center_z": int(z),

                        "protocol": protocol,

                        "n_volumes": int(ndwi),

                        "shape_x": int(shape[0]),

                        "shape_y": int(shape[1]),

                        "shape_z": int(shape[2]),

                    })

                    n_subject_patches += 1

        print(
            f"   ✅ patches={n_subject_patches}",
            flush=True,
        )

        subjects_ok += 1

    except Exception as e:

        print(
            f"❌ ERROR {subject}: {e}",
            flush=True,
        )

        subjects_failed += 1


# =========================================================
# SAVE
# =========================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUT_CSV,
    index=False,
)

print("\n==============================")
print(f"✅ Subjects OK     : {subjects_ok}")
print(f"❌ Subjects Failed : {subjects_failed}")
print(f"🧠 Total patches   : {len(df)}")
print(f"👤 Total subjects  : {df['subject'].nunique()}")
print("==============================")