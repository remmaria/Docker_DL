import numpy as np
import os
import sys
import pandas as pd
from scipy.spatial import ConvexHull

from sub.calc_sequence import calc_sequence

import nibabel as nib
import plotly.graph_objects as go

# Remove duplicados (pares espelhados)
def remove_opposite_duplicates(bvecs):
    unique = []
    for v in bvecs:
        if not any(np.allclose(v, u) or np.allclose(v, -u) for u in unique):
            unique.append(v)
    return np.array(unique)


# Funções de distância e energia
def angular_distance(v1, v2):
    return np.arccos(np.clip(np.abs(np.dot(v1, v2)), -1.0, 1.0))

def repulsion_energy(vectors):
    energy = 0.0
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            theta = angular_distance(vectors[i], vectors[j])
            energy += 1.0 / (theta + 1e-6)
    return energy

def select_greedy_retas(bvecs, n_retas, center_init=None, alpha=1.0):
    """
    alpha: peso entre espalhamento angular (repulsão) e distância ao centro de massa original.
    alpha = 1.0 => mais uniforme, alpha = 0.0 => só centro de massa
    """
    selected = [bvecs[0]]
    candidates = [v for v in bvecs[1:] if not np.allclose(v, -selected[0])]

    while len(selected) < n_retas:
        best_score = np.inf
        best_vec = None

        for c in candidates:
            new_sel = selected + [c]
            repulsion = repulsion_energy(new_sel)
            if center_init is not None:
                center_final = np.mean(new_sel, axis=0)
                dist_cm = np.linalg.norm(center_final - center_init)
            else:
                dist_cm = 0.0

            score = alpha * repulsion + (1 - alpha) * dist_cm  # balanceamento

            if score < best_score:
                best_score = score
                best_vec = c

        selected.append(best_vec)
        candidates = [v for v in candidates if not np.allclose(v, best_vec) and not np.allclose(v, -best_vec)]

    return np.array(selected)

def electrostatic_energy(bvecs):
    energy = 0.0
    for i in range(len(bvecs)):
        for j in range(i+1, len(bvecs)):
            dist = np.linalg.norm(bvecs[i] - bvecs[j])
            energy += 1.0 / (dist + 1e-6)
    return energy

def angle_distribution(bvecs):
    angles = []
    for i in range(len(bvecs)):
        for j in range(i+1, len(bvecs)):
            angle = np.arccos(np.clip(np.abs(np.dot(bvecs[i], bvecs[j])), -1, 1))
            angles.append(np.degrees(angle))
    return np.array(angles)

def select_best(bvecs_base, bvecs_unique, n_retas, original_indices, bvals, sub_folder=None):
    center_init = np.mean(bvecs_base, axis=0)
    cm_init = np.linalg.norm(center_init)
    ee_init = electrostatic_energy(bvecs_base)
    # EE = minimizá-la (quanto menor, melhor), pois uma 
    # boa distribuição tem maior separação angular ⇒ menor energia de repulsão (inverso da distância angular).

    best_alpha = None
    best_selected = None
    results = []
    alphas_range = np.arange(0,1.1,0.1)
    for alpha_sel in alphas_range:
        selected_dirs = select_greedy_retas(bvecs_unique, n_retas, center_init=center_init, alpha=alpha_sel)
        
        center_final = np.mean(selected_dirs, axis=0)
        cm_final = np.linalg.norm(center_final)  # distância até a origem
        ee_final = repulsion_energy(selected_dirs)
        betw_angles = angle_distribution(selected_dirs)

        results.append({
            'alpha': alpha_sel,
            'cm_init': cm_init,
            'ee_init': ee_init,
            'cm_final': cm_final,
            'ee_final': ee_final,
            'angle_min': np.min(betw_angles),
            'angle_max': np.max(betw_angles),
            'angle_mean': np.mean(betw_angles),
            'angle_std': np.std(betw_angles),
        })

    # Normalização min-max
    df = pd.DataFrame(results)
    df["cm_norm"] = (df["cm_final"] - df["cm_final"].min()) / (df["cm_final"].max() - df["cm_final"].min())
    df["ee_norm"] = (df["ee_final"] - df["ee_final"].min()) / (df["ee_final"].max() - df["ee_final"].min())

    # Ajuste o peso conforme sua tolerância ao "custo em energia"
    w_cm = 0.7
    w_ee = 0.3
    df["score_weighted"] = w_cm * df["cm_norm"] + w_ee * df["ee_norm"]

    best_idx = df["score_weighted"].idxmin()
    best_alpha = df.loc[best_idx, "alpha"]
    if sub_folder is not None:
        df.to_csv(f"{sub_folder}/sel_{n_retas}_score.csv", index=False, sep='\t')

    best_selected = select_greedy_retas(bvecs_unique, n_retas, center_init=center_init, alpha=best_alpha)

    # Encontrar índices originais correspondentes aos vetores selecionados
    best_selected_indices = []
    for v in best_selected:
        for i, ref_v in enumerate(bvecs_base):
            if np.allclose(v, ref_v) or np.allclose(v, -ref_v):
                best_selected_indices.append(original_indices[i])
                break

    best_center = np.mean(best_selected, axis=0)
    cm_best = np.linalg.norm(best_center)
    print(f'INFO: Módulo do centro de massa inicial (alpha={best_alpha}): {cm_init:.4f}')
    print(f'INFO: Módulo do centro de massa final (alpha={best_alpha}):   {cm_best:.4f}')

    if sub_folder is not None:
        df = pd.DataFrame(results)
        df.to_csv(f"{sub_folder}/sel_{n_retas}retas_alphas.csv", index=False, sep='\t')

        # === PLOT ===
        hover_texts = [
            f"Index: {idx}<br>bval: {bval}<br>x: {v[0]:.3f}<br>y: {v[1]:.3f}<br>z: {v[2]:.3f}"
            for idx, bval, v in zip(best_selected_indices, bvals[best_selected_indices], best_selected)
        ]

        fig = go.Figure()

        # 1. Pontos originais (azul)
        fig.add_trace(go.Scatter3d(
            x=bvecs_base[:, 0], y=bvecs_base[:, 1], z=bvecs_base[:, 2],
            mode='markers',
            marker=dict(size=6, color='lightblue'),
            name='Originais'
        ))

        # 2. Pontos espelhados (laranja)
        fig.add_trace(go.Scatter3d(
            x=-bvecs_base[:, 0], y=-bvecs_base[:, 1], z=-bvecs_base[:, 2],
            mode='markers',
            marker=dict(size=6, color='lightgreen'),
            name='Espelhados'
        ))

        # 3. Pontos selecionados finais (v) – vermelho escuro, cross, com hover
        fig.add_trace(go.Scatter3d(
            x=best_selected[:, 0], y=best_selected[:, 1], z=best_selected[:, 2],
            mode='markers',
            marker=dict(size=8, color='darkblue', symbol='cross'),
            name='Selecionados (v)',
            text=hover_texts,
            hoverinfo='text'
        ))

        # 4. Espelhados dos selecionados (-v) – verde escuro, cross
        selected_neg = -best_selected
        fig.add_trace(go.Scatter3d(
            x=selected_neg[:, 0], y=selected_neg[:, 1], z=selected_neg[:, 2],
            mode='markers',
            marker=dict(size=8, color='darkgreen', symbol='cross'),
            name='Selecionados espelhados (-v)'
        ))

        # 5. Retas selecionadas (com espelhados)
        for i, v in enumerate(best_selected):
            fig.add_trace(go.Scatter3d(
                x=[-v[0], v[0]],
                y=[-v[1], v[1]],
                z=[-v[2], v[2]],
                mode='lines',
                line=dict(width=4, color='red', dash='dash'),
                name=f'Reta {i+1}',
                showlegend=True
            ))

        # 6. Origem
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers+text',
            marker=dict(size=7, color='black'),
            textposition='top center',
            name=f'Origin (0,0,0)'
        ))

        # 6. Centro de massa inicial (esfera preta)
        fig.add_trace(go.Scatter3d(
            x=[center_init[0]], y=[center_init[1]], z=[center_init[2]],
            mode='markers+text',
            marker=dict(size=6, color='orange'),
            textposition='top center',
            name=f'cm0:{cm_init:.3f}'
        ))

        # 7. Centro de massa final (esfera verde)
        fig.add_trace(go.Scatter3d(
            x=[best_center[0]], y=[best_center[1]], z=[best_center[2]],
            mode='markers+text',
            marker=dict(size=6, color='green'),
            textposition='top center',
            name=f'cm1:{cm_best:.3f}'
        ))
        if True:

            # 8. Esfera de fundo
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.1, showscale=False, colorscale='Greys',hoverinfo='skip'))

            all_selected = np.vstack([best_selected, -best_selected])
            intensity_vals = np.linalg.norm(all_selected, axis=1)


            hull = ConvexHull(all_selected)

            # Criar mesh a partir dos triângulos do hull
            fig.add_trace(go.Mesh3d(
                x=all_selected[:, 0],
                y=all_selected[:, 1],
                z=all_selected[:, 2],
                i=hull.simplices[:, 0],
                j=hull.simplices[:, 1],
                k=hull.simplices[:, 2],
                opacity=0.7,
                intensity=intensity_vals,
                colorscale='Viridis',
                name='Convex Hull (v + -v)',
                showscale=False,
                hoverinfo='skip'
            ))

        # Layout
        fig.update_layout(
            title='Distribuição de vetores, retas e centros de massa',
            scene=dict(aspectmode='data'),
            legend=dict(itemsizing='constant'),
            showlegend=True
        )

        fig.write_html(f"{sub_folder}/sel_{n_retas}retas.html")
        print(f"INFO: Selected vectors plot saved in {sub_folder}/sel_{n_retas}retas.html")

    return best_selected, best_selected_indices

def sub_sample(dwi_path, bval_path, bvec_path, subset_original, subset_sub, sub_folder=None):

    #alpha = 1.0: apenas espalhamento angular (como antes).
    #alpha = 0.0: apenas minimização da distância entre centros de massa.
    #alpha = 0.7: valor intermediário (recomendado para começar).

    bvecs = np.loadtxt(bvec_path).T

    bvals, dic_seq, dic_seq_str, unique_values, val_tolerance = calc_sequence(bval_path)

    if subset_original is not None:
        subset_original
        if subset_original != dic_seq_str:
            print(f"ERROR: Original subset {subset_original} does not match the input sequence {dic_seq_str}.", flush=True)
            sys.exit(1)            

    dic_n = {}
    n_pts = subset_sub.split("_")
    #print(n_pts, file=sys.stderr)
    n_pts = [b_part.split("-") for b_part in n_pts]
    #print(n_pts, file=sys.stderr)
    for b_n in n_pts:
        dic_n[int(b_n[0])]=int(b_n[1])
    #print(dic_n, file=sys.stderr)

    #print(dic_n, flush=True, file=sys.stderr) #{0: 1, 1000: 65}
    #print(dic_seq, flush=True, file=sys.stderr) #{0: 1, 1000: 64}

    for key in dic_n.keys():
        if key not in dic_seq:
            print(f"ERROR: b-value {key} not found in the input sequence: {dic_seq}.", flush=True)
            sys.exit(1)
        if dic_n[key] > dic_seq[key]:
            print(f"ERROR: Requested number of directions {dic_n[key]} for b-value {key} exceeds available {dic_seq[key]} in sequence {dic_seq}.", flush=True)
            sys.exit(1)

    #print(dic_seq, flush=True, file=sys.stderr)

    if all(key in unique_values for key in dic_n.keys()):
        
        os.makedirs(sub_folder, exist_ok=True)

        val_tolerance = np.array(val_tolerance)

        dwis_nib = nib.load(dwi_path)
        dwis = dwis_nib.get_fdata()

        #print(bvecs.shape, file=sys.stderr)
        #print(bvals.shape, file=sys.stderr)
        #print(dwis.shape, file=sys.stderr)    

        bvals_dic = {}
        bvecs_dic = {}
        dwis_dic = {}
        idxs_ref_dic = {}
        selected_data = []
        selected_indices_all = []

        if 0 in dic_n:
            n_b0_solicitados = dic_n[0] # Pega o '1' de '0-1'
            idx_b0_total = np.where(val_tolerance == 0)[0]
            
            # Pega apenas os N primeiros b0s (ou use np.random.choice para aleatórios)
            idx_b0_selecionados = idx_b0_total[:n_b0_solicitados] 
            
            selected_indices_all.extend(idx_b0_selecionados.tolist())

        for c, b in enumerate([k for k in dic_n.keys() if k != 0]):

            n_pts = dic_n[b]

            dwis_dic[b] = dwis[:,:,:,val_tolerance == b]
            idxs_ref_dic[b] = np.where(val_tolerance == b)[0] #índices para cada b-value

            bvals_dic[b] = bvals[val_tolerance == b]
            bvecs_dic[b] = bvecs[val_tolerance == b,:]

            # bvecs unitários na esfera (N direções)
            bvecs_base = bvecs[val_tolerance == b,:]
            bvecs_all = np.vstack([bvecs_base, -bvecs_base])  # 2*N direções
            bvecs_unique = remove_opposite_duplicates(bvecs_all)
            best_selected, selected_indices_global = select_best(bvecs_base, bvecs_unique, n_pts, idxs_ref_dic[b], bvals)
                   
            # Índices globais no volume (para DWI, bvals e bvecs)
            selected_indices_global = []
            for v in best_selected:
                for i, ref_v in enumerate(bvecs_dic[b]):
                    if np.allclose(v, ref_v) or np.allclose(v, -ref_v):
                        selected_indices_global.append(idxs_ref_dic[b][i])
                        break

            # Acumular todos para salvar ao fim
            selected_indices_all.extend(selected_indices_global)

            for v in best_selected:
                for i, ref_v in enumerate(bvecs_dic[b]):
                    if np.allclose(v, ref_v) or np.allclose(v, -ref_v):
                        original_idx = int(idxs_ref_dic[b][i])
                        selected_data.append({
                            'original_index': original_idx,
                            'bval': bvals[original_idx],
                            'x': v[0],
                            'y': v[1],
                            'z': v[2]
                        })
                        break

        if False:
            # Ordenar por índice original
            df_all = pd.DataFrame(selected_data).sort_values("original_index").reset_index(drop=True)

            # Salvar CSV final
            csv_out = f"{sub_folder}/selected_vectors_all.csv"
            df_all.to_csv(csv_out, index=False)
            print(f"INFO: Selected vectors saved in {sub_folder}/selected_vectors_all.csv")

    #print(selected_indices_all,file=sys.stderr)
    selected_indices_all = sorted(selected_indices_all) #deixar na mesma ordem de aquisição

    bvals_selected = bvals[selected_indices_all]
    bvecs_selected = bvecs[selected_indices_all,:]

    if sub_folder is not None:
        file_name = bvec_path.split('/')[-1].split('.')[0]

        np.savetxt(f"{sub_folder}/{file_name}.bval", bvals_selected[None,:], fmt="%d")
        np.savetxt(f"{sub_folder}/{file_name}.bvec", bvecs_selected.T, fmt="%.10f")  # Transpor de volta

        ext='.nii.gz' if dwi_path.endswith('.nii.gz') else '.nii'
        dwis_selected = dwis[:,:,:,selected_indices_all]
        dwis_file = nib.Nifti1Image(dwis_selected, affine=dwis_nib.affine, header=dwis_nib.header)
        nib.save(dwis_file, f"{sub_folder}/{file_name}{ext}")
        print(f"INFO: New DWIs, bvec and bval saved in {sub_folder}")

    return selected_indices_all, bvals_selected, bvecs_selected