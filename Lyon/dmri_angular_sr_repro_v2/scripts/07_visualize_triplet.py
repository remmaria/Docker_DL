#!/usr/bin/env python3
"""
Etapa 7 (diagnostico visual, nao faz parte do pipeline de treino/avaliacao):
para um sujeito e um par (shell_b, n_level) especifico, plota TODOS os bvecs
adquiridos naquela shell na esfera unitaria (com seus espelhos antipodais --
dMRI trata v e -v como a MESMA direcao fisica, mesma convencao ja usada em
utils/gradients.py:spherical_triplet_residual/farthest_point_sampling),
destacando quais foram selecionados como direcoes de ENTRADA pelo esquema de
subamostragem (etapa 2, scripts/02_subsample_directions.py) para aquele
n_level, e marca uma trinca especifica (par de entrada a,b + alvo t, etapa
2b, scripts/02b_build_rrin_triplets.py) usada pelo RRIN3D/AMT3D/HFD3D.
Desenha o arco de circulo maximo (geodesica, via slerp) entre bvec_a e
bvec_b -- a "trajetoria" que a hipotese de fluxo assume existir -- e mostra
onde o bvec-alvo REAL cai em relacao a esse arco (o desvio visual e
literalmente o `residual_deg` ja calculado por
utils.gradients.spherical_triplet_residual, ver protocolo secao 10.2).

Tambem plota as fatias 2D reais das DWIs dessa trinca: vol_a, vol_b, alvo
real, e, como referencia, o blend ingenuo (1-t_frac)*vol_a + t_frac*vol_b
que e exatamente o que um metodo de fluxo produziria SEM nenhum
refinamento/warping (a "hipotese nula" mais simples possivel) -- e o mapa
de erro absoluto |alvo_real - blend_ingenuo|, para visualizar concretamente
o quanto essa hipotese erra naquele voxel/trinca especifico.

Se o `<tag>_rrin_triplets.npz` foi gerado com
`scripts/02b_build_rrin_triplets.py --ensemble-m M` (ver protocolo secao
14.5 item 1/addendum 2026-08-27, "ensemble em estrela", e
utils/gradients.py:find_star_ensemble_batch/model/rrin3d_star.py), a figura
da esfera TAMBEM desenha os ate M pares candidatos DIVERSOS do feixe
daquela mesma trinca (arcos coloridos mais finos, alem do arco preto
tracejado do par-unico canonico) -- util pra visualizar concretamente a
diversidade angular que o ensemble usa pra fundir predicoes (ver
--ensemble-m/--no-ensemble abaixo). Sem esses campos no npz (par-unico
apenas, comportamento de sempre), a figura fica identica a antes.

Selecao da trinca-exemplo (--example, default "typical"): dentre as trincas
VALIDAS (valid=True) daquele (shell_b,n_level), escolhe a que tem
`gap_deg` mais proximo da MEDIANA das validas ("typical" -- um caso comum,
nao um extremo escolhido a dedo); "best" escolhe a de menor `residual_deg`
(par mais bem alinhado com o alvo); "worst" escolhe a de maior `gap_deg`
entre TODAS as trincas (incluindo invalidas, se existirem no esquema) --
util para ilustrar visualmente o regime de falha da secao 13 do protocolo.
--triplet-index sobrepoe a selecao automatica e usa o indice literal (na
ordem do array, nao um indice de bvec global) dentro do array de trincas
daquele (shell_b,n_level).

Uso:
    python scripts/07_visualize_triplet.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --out triplet_shell1000_n16.png

Por padrao escolhe o primeiro sujeito do split "train" que tenha
scheme.npz + trincas para esse (shell_b, n_level) -- use --subject para
escolher manualmente. Nao requer PyTorch/GPU -- so numpy/matplotlib/nibabel,
roda em CPU (inclusive fora do cluster, se os arquivos de dados estiverem
acessiveis localmente).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, load_dwi, split_shells
from utils.masking import load_or_build_mask
# NAO importamos utils.dataset._resolve_shell_key aqui de proposito: aquele
# modulo importa torch no topo do arquivo (e' um Dataset do PyTorch), o que
# tornaria este script -- que deliberadamente nao precisa de GPU/torch, so
# numpy/matplotlib/nibabel -- dependente de uma instalacao de torch so pra
# rodar uma funcao de 10 linhas. Reimplementada aqui, identica em
# comportamento (mesma logica de utils/dataset.py:_resolve_shell_key).


def _resolve_shell_key(shells: dict, shell_b: float, tol: float) -> float:
    """Acha a chave de `shells` (dict de split_shells) mais proxima de
    `shell_b`, ignorando b0 (chave 0). Levanta erro se nao achar nada
    dentro de `tol`. (Copia deliberada de utils.dataset._resolve_shell_key
    -- ver comentario acima do import.)"""
    best_key, best_diff = None, None
    for k in shells:
        if k == 0:
            continue
        diff = abs(k - shell_b)
        if best_diff is None or diff < best_diff:
            best_key, best_diff = k, diff
    if best_key is None or best_diff > tol:
        raise RuntimeError(f"shell {shell_b} nao encontrada (tol={tol}), shells disponiveis: {list(shells)}")
    return best_key


def _tag_of(e):
    return e.subject if not e.session else f"{e.subject}_{e.session}"


def _fix_sign(reference: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Escolhe o representante antipodal de `v` (v ou -v) mais proximo de
    `reference` -- MESMA convencao de sinal usada por
    utils.gradients.spherical_triplet_residual (flip se dot < 0), pra que o
    arco/marcadores plotados sejam geometricamente consistentes com o que o
    pipeline realmente calculou (residual_deg/gap_deg/t_frac)."""
    return v if np.dot(reference, v) >= 0 else -v


def _slerp_arc(a: np.ndarray, b: np.ndarray, n: int = 60) -> np.ndarray:
    """Interpolacao esferica (slerp) entre dois vetores unitarios `a`,`b`
    -- desenha o arco de circulo maximo (geodesica) entre eles, a
    "trajetoria" que qualquer metodo de fluxo/warping bilateral assume
    implicitamente existir entre duas direcoes de gradiente. Retorna
    (n,3)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(dot)
    t = np.linspace(0.0, 1.0, n)
    if theta < 1e-8:
        return np.tile(a, (n, 1))
    sin_theta = np.sin(theta)
    w_a = np.sin((1 - t) * theta) / sin_theta
    w_b = np.sin(t * theta) / sin_theta
    return w_a[:, None] * a[None, :] + w_b[:, None] * b[None, :]


def _pick_subject_and_key(entries, triplets_dir: Path, shell_b: float, n_level: int,
                           wanted_subject: str = None):
    key = f"{shell_b}__{n_level}"
    candidates = [e for e in entries if e.split == "train"] if wanted_subject is None else entries
    for e in candidates:
        tag = _tag_of(e)
        if wanted_subject is not None and tag != wanted_subject:
            continue
        trip_path = triplets_dir / f"{tag}_rrin_triplets.npz"
        scheme_path = triplets_dir / f"{tag}_scheme.npz"
        if not trip_path.exists() or not scheme_path.exists():
            continue
        trip = np.load(trip_path)
        if f"{key}__target" not in trip.files:
            continue
        return e, tag, key
    raise SystemExit(f"Nenhum sujeito encontrado com scheme+trincas para shell_b={shell_b} "
                      f"n_level={n_level}" + (f" (procurando especificamente {wanted_subject})"
                                              if wanted_subject else " no split 'train'") + ".")


def _write_sphere_html(out_html: Path, shell_bvecs: np.ndarray, shell_all_idx: np.ndarray,
                        is_input_of_shell: np.ndarray, bvec_a: np.ndarray, bvec_b: np.ndarray,
                        bvec_t: np.ndarray, idx_a: int, idx_b: int, idx_t: int, arc: np.ndarray,
                        tag: str, shell_key: float, n_level: int, ti: int, example_label: str,
                        is_valid: bool, residual_deg: float, gap_deg: float, t_frac: float,
                        ensemble_arcs: list = None) -> None:
    """Gera uma versao HTML interativa (Plotly, rotacionavel com o mouse) da
    MESMA figura da esfera desenhada estaticamente em `main()` -- mesmos
    dados/cores/geometria, so o backend de renderizacao muda. Nao depende do
    pacote Python `plotly` (indisponivel neste ambiente de desenvolvimento,
    ver docstring do modulo) -- so escreve um HTML autocontido que carrega a
    biblioteca JS do Plotly via CDN (https://cdnjs.cloudflare.com, mesma
    convencao usada em outros HTMLs deste projeto) e embute os dados via
    JSON puro (`json.dumps` sobre listas, sem `plotly.graph_objects`).
    Requer conexao com a internet no navegador que ABRIR o arquivo (nao no
    cluster que o GERA -- a geracao aqui e so escrita de texto)."""
    import json

    non_input = shell_bvecs[~is_input_of_shell]
    non_input_idx = shell_all_idx[~is_input_of_shell]
    input_pts = shell_bvecs[is_input_of_shell]
    input_idx_arr = shell_all_idx[is_input_of_shell]

    def _pack_with_mirror(vecs):
        both = np.concatenate([vecs, -vecs], axis=0)
        return both[:, 0].tolist(), both[:, 1].tolist(), both[:, 2].tolist()

    def _hover_labels(idx_arr):
        return [f"bvec #{i}" for i in idx_arr] + [f"bvec #{i} (espelho -v)" for i in idx_arr]

    x_ni, y_ni, z_ni = _pack_with_mirror(non_input)
    x_in, y_in, z_in = _pack_with_mirror(input_pts)

    uu, vv = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    xs = (np.cos(uu) * np.sin(vv)).tolist()
    ys = (np.sin(uu) * np.sin(vv)).tolist()
    zs = np.cos(vv).tolist()

    traces = [
        {
            "type": "surface", "x": xs, "y": ys, "z": zs,
            "opacity": 0.12, "showscale": False,
            "colorscale": [[0, "#cccccc"], [1, "#cccccc"]],
            "hoverinfo": "skip", "name": "esfera unitária",
        },
        {
            "type": "scatter3d", "mode": "markers", "x": x_ni, "y": y_ni, "z": z_ni,
            "marker": {"size": 3, "color": "lightsteelblue", "opacity": 0.6},
            "text": _hover_labels(non_input_idx), "hoverinfo": "text",
            "name": f"adquiridas, não-entrada (shell {shell_key})",
        },
        {
            "type": "scatter3d", "mode": "markers", "x": x_in, "y": y_in, "z": z_in,
            "marker": {"size": 5, "color": "royalblue", "opacity": 0.9},
            "text": _hover_labels(input_idx_arr), "hoverinfo": "text",
            "name": f"entrada (n_level={n_level})",
        },
        {
            "type": "scatter3d", "mode": "lines",
            "x": arc[:, 0].tolist(), "y": arc[:, 1].tolist(), "z": arc[:, 2].tolist(),
            "line": {"color": "black", "width": 5, "dash": "dash"},
            "hoverinfo": "skip", "name": "geodésica a→b (par único canônico)",
        },
    ]
    for ens_arc, ens_color, ens_a_idx, ens_b_idx in (ensemble_arcs or []):
        traces.append({
            "type": "scatter3d", "mode": "lines",
            "x": ens_arc[:, 0].tolist(), "y": ens_arc[:, 1].tolist(), "z": ens_arc[:, 2].tolist(),
            "line": {"color": ens_color, "width": 3},
            "hoverinfo": "skip", "name": f"ensemble: par (#{ens_a_idx},#{ens_b_idx})",
        })
    for v, vidx, color, label in [(bvec_a, idx_a, "limegreen", f"bvec_a (entrada, #{idx_a})"),
                                   (bvec_b, idx_b, "limegreen", f"bvec_b (entrada, #{idx_b})"),
                                   (bvec_t, idx_t, "crimson", f"bvec_alvo (real, #{idx_t})")]:
        both = np.stack([v, -v])
        traces.append({
            "type": "scatter3d", "mode": "markers+text",
            "x": both[:, 0].tolist(), "y": both[:, 1].tolist(), "z": both[:, 2].tolist(),
            "marker": {"size": [12, 7], "color": color, "opacity": [1.0, 0.5],
                       "line": {"color": "black", "width": 1.5}},
            "text": [label, label + " (espelho -v)"], "textposition": "top center",
            "hoverinfo": "text", "name": label,
        })

    ensemble_suffix = f" — ensemble: {len(ensemble_arcs) + 1} par(es)" if ensemble_arcs else ""
    title = (f"{tag} — shell {shell_key} — n_level={n_level}<br>"
             f"trinca #{ti} ({example_label}, valid={is_valid}) — residual={residual_deg:.1f}°, "
             f"gap={gap_deg:.1f}°, t={t_frac:.2f}{ensemble_suffix}<br>"
             f"pontos claros/translúcidos = espelho antipodal (-v, mesma direção física)")
    layout = {
        "title": {"text": title, "font": {"size": 14}},
        "scene": {"aspectmode": "cube",
                  "xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False}},
        "legend": {"x": 0.0, "y": 1.0},
        "margin": {"l": 0, "r": 0, "t": 110, "b": 0},
        "height": 850,
    }

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{tag} -- esfera de bvecs (interativo)</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>
<style>body {{ margin: 0; font-family: sans-serif; }}</style>
</head>
<body>
<div id="plot" style="width:100%;height:100vh;"></div>
<script>
var data = {json.dumps(traces)};
var layout = {json.dumps(layout)};
Plotly.newPlot('plot', data, layout, {{responsive: true}});
</script>
</body>
</html>
"""
    Path(out_html).write_text(html, encoding="utf-8")


def _load_ensemble_pairs(trip, key: str, ti: int, max_m: int = None):
    """Le o feixe "ensemble em estrela" (ver utils/gradients.py:
    find_star_ensemble_batch e scripts/02b_build_rrin_triplets.py
    --ensemble-m) da trinca `ti`, se os campos `{key}__ens_*` existirem no
    npz -- devolve lista de (idx_a, idx_b) GLOBAIS dos pares REAIS do feixe
    (posicoes de padding, `ens_valid=False` ou indice -1, sao descartadas),
    ou None se o npz nao tem esses campos (par-unico apenas, sem
    --ensemble-m -- comportamento de sempre, figura fica identica a antes).
    `max_m` (opcional) limita quantas posicoes do feixe ler (default: todas
    as gravadas)."""
    ens_a_key = f"{key}__ens_pair_a"
    if ens_a_key not in trip.files:
        return None
    ens_pair_a = trip[ens_a_key][ti]
    ens_pair_b = trip[f"{key}__ens_pair_b"][ti]
    ens_valid = trip[f"{key}__ens_valid"][ti]
    if max_m is not None:
        ens_pair_a = ens_pair_a[:max_m]
        ens_pair_b = ens_pair_b[:max_m]
        ens_valid = ens_valid[:max_m]
    pairs = []
    for a_idx, b_idx, ok in zip(ens_pair_a.tolist(), ens_pair_b.tolist(), ens_valid.tolist()):
        if not ok or a_idx < 0 or b_idx < 0:
            continue
        pairs.append((int(a_idx), int(b_idx)))
    return pairs


# paleta pros arcos do feixe "ensemble em estrela" -- distinta do preto
# tracejado do par-unico canonico e do verde-limao/carmesim dos marcadores
# a/b/alvo, pra nao competir visualmente com eles.
_ENSEMBLE_ARC_COLORS = ["darkorange", "mediumpurple", "teal", "goldenrod",
                        "deeppink", "steelblue", "olive", "brown"]


def _select_triplet(valid, gap_deg, residual_deg, example: str, triplet_index: int = None):
    n = valid.shape[0]
    if triplet_index is not None:
        if not (0 <= triplet_index < n):
            raise SystemExit(f"--triplet-index {triplet_index} fora do intervalo [0,{n-1}]")
        return triplet_index
    if example == "worst":
        return int(np.argmax(gap_deg))
    valid_idx = np.where(valid)[0]
    if valid_idx.size == 0:
        print("[aviso] nenhuma trinca VALIDA neste (shell_b,n_level) -- usando a de menor "
              "residual_deg entre todas (provavelmente ainda assim ruim, ver protocolo secao 13).",
              flush=True)
        return int(np.argmin(residual_deg))
    if example == "best":
        best_local = np.argmin(residual_deg[valid_idx])
        return int(valid_idx[best_local])
    # "typical" (default): a valida com gap_deg mais proximo da mediana das validas.
    median_gap = np.median(gap_deg[valid_idx])
    typical_local = np.argmin(np.abs(gap_deg[valid_idx] - median_gap))
    return int(valid_idx[typical_local])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="pasta com <tag>_scheme.npz (etapa 2) e <tag>_rrin_triplets.npz "
                          "(etapa 2b) -- tipicamente work_dir/subsampling")
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--subject", default=None,
                     help="tag do sujeito (subject ou subject_session) -- default: primeiro "
                          "sujeito do split 'train' com dados disponiveis para este shell/n_level")
    ap.add_argument("--example", choices=["typical", "best", "worst"], default="typical",
                     help="qual trinca escolher automaticamente (ver docstring do modulo). "
                          "Ignorado se --triplet-index for passado.")
    ap.add_argument("--triplet-index", type=int, default=None,
                     help="indice literal dentro do array de trincas deste (shell_b,n_level) "
                          "-- sobrepoe --example")
    ap.add_argument("--shell-tol", type=float, default=100.0)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--slice-axis", type=int, choices=[0, 1, 2], default=2,
                     help="eixo do array 3D (i,j,k) ao longo do qual cortar a fatia 2D exibida "
                          "(nao ha info de orientacao anatomica padronizada aqui -- ajuste e "
                          "confira visualmente qual eixo corresponde a axial/coronal/sagital "
                          "para os dados especificos deste dataset)")
    ap.add_argument("--slice-index", type=int, default=None,
                     help="indice da fatia (default: meio da caixa delimitadora da mascara)")
    ap.add_argument("--no-crop", action="store_true",
                     help="nao recortar as fatias 2D pela caixa delimitadora da mascara (mostra "
                          "o FOV inteiro, com mais fundo preto)")
    ap.add_argument("--elev", type=float, default=20.0, help="elevacao da vista 3D da esfera")
    ap.add_argument("--azim", type=float, default=45.0, help="azimute da vista 3D da esfera")
    ap.add_argument("--ensemble-m", type=int, default=None,
                     help="quantos pares do feixe 'ensemble em estrela' desenhar na esfera, se "
                          "o npz tiver sido gerado com --ensemble-m (ver "
                          "scripts/02b_build_rrin_triplets.py e protocolo secao 14.5 item 1). "
                          "Default: desenha TODOS os pares gravados no feixe. Ignorado (sem "
                          "efeito, sem erro) se o npz nao tiver esses campos.")
    ap.add_argument("--no-ensemble", action="store_true",
                     help="nao desenhar o feixe 'ensemble em estrela' mesmo que o npz tenha "
                          "esses campos -- volta a mostrar so o par-unico canonico.")
    ap.add_argument("--no-html", action="store_true",
                     help="nao gerar a versao HTML interativa da esfera (rotacionavel com o "
                          "mouse, via Plotly carregado por CDN -- ver _write_sphere_html). Por "
                          "padrao a esfera SEMPRE tambem sai como "
                          "<out>_esfera_interativo.html, alem do PNG estatico de sempre.")
    ap.add_argument("--out", required=True, help="caminho do PNG de saida")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registra a projecao 3d
    # (necessario em algumas versoes mais antigas do matplotlib; import
    # redundante/inofensivo nas mais novas, onde a projecao 3d ja e
    # registrada automaticamente ao importar mpl_toolkits.mplot3d)

    entries = load_manifest(args.manifest)
    triplets_dir = Path(args.triplets_dir)
    e, tag, key = _pick_subject_and_key(entries, triplets_dir, args.shell_b, args.n_level,
                                         wanted_subject=args.subject)
    print(f"[info] sujeito: {tag} (split={e.split})", flush=True)

    scheme = np.load(triplets_dir / f"{tag}_scheme.npz")
    input_idx = scheme[f"{key}__input"]
    target_idx_all = scheme[f"{key}__target"]

    trip = np.load(triplets_dir / f"{tag}_rrin_triplets.npz")
    pair_a_all = trip[f"{key}__pair_a"]
    pair_b_all = trip[f"{key}__pair_b"]
    target_all = trip[f"{key}__target"]
    t_frac_all = trip[f"{key}__t_frac"]
    valid_all = trip[f"{key}__valid"]
    residual_all = trip[f"{key}__residual_deg"]
    gap_all = trip[f"{key}__gap_deg"]

    ti = _select_triplet(valid_all, gap_all, residual_all, args.example, args.triplet_index)
    idx_a, idx_b, idx_t = int(pair_a_all[ti]), int(pair_b_all[ti]), int(target_all[ti])
    t_frac = float(t_frac_all[ti])
    is_valid = bool(valid_all[ti])
    residual_deg = float(residual_all[ti])
    gap_deg = float(gap_all[ti])
    print(f"[info] trinca escolhida (indice {ti}, criterio '{args.example if args.triplet_index is None else 'manual'}'): "
          f"bvec_a=#{idx_a}, bvec_b=#{idx_b}, alvo=#{idx_t}, t_frac={t_frac:.3f}, "
          f"valid={is_valid}, residual_deg={residual_deg:.2f}, gap_deg={gap_deg:.2f}", flush=True)

    ensemble_pairs = None
    if not args.no_ensemble:
        ensemble_pairs = _load_ensemble_pairs(trip, key, ti, max_m=args.ensemble_m)
        if ensemble_pairs is None:
            print("[info] npz sem campos '__ens_*' (rodado sem --ensemble-m em "
                  "scripts/02b_build_rrin_triplets.py) -- mostrando so o par-unico canonico.",
                  flush=True)
        else:
            print(f"[info] feixe 'ensemble em estrela': {len(ensemble_pairs)} par(es) real(is) "
                  f"encontrados -- {ensemble_pairs}", flush=True)

    bvals, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)
    shells = split_shells(bvals, tol=args.shell_tol)
    shell_key = _resolve_shell_key(shells, args.shell_b, args.shell_tol)
    shell_all_idx = np.asarray(shells[shell_key], dtype=int)
    shell_bvecs = bvecs[shell_all_idx]
    input_set = set(input_idx.tolist())
    is_input_of_shell = np.array([g in input_set for g in shell_all_idx])

    # ---------- figura da esfera (separada da figura das DWIs, ver abaixo) ----------
    fig = plt.figure(figsize=(10, 9))
    ax_sphere = fig.add_subplot(1, 1, 1, projection="3d")

    def _scatter_with_mirror(ax, vecs, **kwargs):
        both = np.concatenate([vecs, -vecs], axis=0)
        ax.scatter(both[:, 0], both[:, 1], both[:, 2], **kwargs)

    # esfera de referencia (wireframe leve)
    uu, vv = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    xs, ys, zs = np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)
    ax_sphere.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.3, alpha=0.4)

    _scatter_with_mirror(ax_sphere, shell_bvecs[~is_input_of_shell], color="lightsteelblue",
                          s=14, alpha=0.6, label=f"adquiridas, nao-entrada (shell {shell_key})")
    _scatter_with_mirror(ax_sphere, shell_bvecs[is_input_of_shell], color="royalblue",
                          s=26, alpha=0.9, label=f"entrada (n_level={args.n_level})")

    bvec_a = bvecs[idx_a]
    bvec_b_raw = bvecs[idx_b]
    bvec_t_raw = bvecs[idx_t]
    bvec_b = _fix_sign(bvec_a, bvec_b_raw)
    bvec_t = _fix_sign(bvec_a, bvec_t_raw)

    arc = _slerp_arc(bvec_a, bvec_b)
    ax_sphere.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="black", linewidth=2.0, linestyle="--",
                   label="geodésica a→b (hipótese de fluxo, par único canônico)")

    # arcos extras do feixe "ensemble em estrela" (ver docstring do modulo/
    # _load_ensemble_pairs) -- so desenhados se o npz tiver esses campos e
    # --no-ensemble nao tiver sido passado. O 1o par do feixe e' SEMPRE
    # identico ao par-unico canonico (ver find_star_ensemble_batch) -- pulado
    # aqui pra nao desenhar o mesmo arco duas vezes por cima.
    ensemble_arcs = []  # guardado pra reusar no HTML interativo, abaixo
    if ensemble_pairs:
        n_extra_drawn = 0
        for ens_a_idx, ens_b_idx in ensemble_pairs:
            if {ens_a_idx, ens_b_idx} == {idx_a, idx_b}:
                continue  # e' o proprio par canonico -- ja desenhado acima
            # Fixar o sinal de ens_bvec_a em relacao ao ALVO DESENHADO
            # (bvec_t), NAO em relacao ao bvec_a do par canonico -- bug
            # encontrado em 2026-08-27 (ver addendum secao 13). Cada par do
            # feixe tem sua PROPRIA convencao interna de sinal em
            # find_star_ensemble_batch (a_pairs = U[iu] cru, b e o alvo
            # fixados em relacao a ESSE "a" especifico -- nao ao "a" do par
            # canonico nem a nenhuma referencia global). Fixar contra
            # bvec_a_canonico (ou deixar a cru, sem fixar nada) escolhe uma
            # das duas representantes antipodais de ens_a_idx de forma
            # ARBITRARIA em relacao ao alvo -- quando calha de ser a
            # representante "errada", o par (a,b) inteiro fica flipado
            # (b tambem flipa em cascata, ver _fix_sign abaixo), o que
            # desloca o arco desenhado para o segmento DIAMETRALMENTE OPOSTO
            # do MESMO circulo maximo (o plano nao muda, so o segmento
            # visivel muda de lado) -- exatamente o padrao visual reportado
            # ("arcos que nao passam pelo alvo"): o arco acaba perto do
            # espelho antipodal do alvo (o ponto translucido), nao do alvo
            # opaco desenhado.
            #
            # Fixar contra bvec_t (o alvo ja desenhado, computado 3 linhas
            # acima) resolve isso: garante por construcao que a
            # representante do alvo "local" a este par (recomputada com a
            # mesma formula de sinal usada aqui) e EXATAMENTE bvec_t --
            # entao, sempre que este par estiver marcado 'between=True' no
            # npz (0<=t_frac<=1 na convencao interna de
            # find_star_ensemble_batch), o arco desenhado passa
            # geometricamente perto do bvec_t desenhado, a uma distancia
            # ditada so por residual_deg (residuo perpendicular ao plano) --
            # verificado numericamente em ~8000 casos sinteticos, distancia
            # arco-alvo sempre dentro de 0.002 do limite teorico
            # 2*sin(residual_deg/2), contra ate 1.8 (essencialmente
            # arbitrario) fixando contra bvec_a_canonico. So afeta esta
            # visualizacao -- a selecao real em find_star_ensemble_batch ja
            # usa a convencao de sinal correta por par internamente, entao
            # os .npz/treinos nao sao afetados por este bug.
            ens_bvec_a = _fix_sign(bvec_t, bvecs[ens_a_idx])
            ens_bvec_b = _fix_sign(ens_bvec_a, bvecs[ens_b_idx])
            ens_arc = _slerp_arc(ens_bvec_a, ens_bvec_b)
            color = _ENSEMBLE_ARC_COLORS[n_extra_drawn % len(_ENSEMBLE_ARC_COLORS)]
            ax_sphere.plot(ens_arc[:, 0], ens_arc[:, 1], ens_arc[:, 2], color=color,
                           linewidth=1.4, linestyle="-", alpha=0.85,
                           label=f"ensemble: par (#{ens_a_idx},#{ens_b_idx})")
            ensemble_arcs.append((ens_arc, color, ens_a_idx, ens_b_idx))
            n_extra_drawn += 1
        if n_extra_drawn == 0:
            print("[info] feixe do ensemble nao tem nenhum par ALEM do canonico pra desenhar "
                  "(so 1 par real disponivel para esta trinca).", flush=True)

    for v, color, label in [(bvec_a, "limegreen", "bvec_a (entrada)"),
                             (bvec_b, "limegreen", "bvec_b (entrada)"),
                             (bvec_t, "crimson", "bvec_alvo (real)")]:
        ax_sphere.scatter(*v, color=color, s=140, edgecolor="black", linewidth=1.2, zorder=5)
        ax_sphere.scatter(*(-v), color=color, s=60, edgecolor="black", linewidth=0.6,
                           alpha=0.5, zorder=4)
    ax_sphere.text(*(bvec_a * 1.08), "a", fontsize=11, weight="bold")
    ax_sphere.text(*(bvec_b * 1.08), "b", fontsize=11, weight="bold")
    ax_sphere.text(*(bvec_t * 1.08), "alvo", fontsize=11, weight="bold", color="crimson")

    ax_sphere.set_box_aspect((1, 1, 1))
    ax_sphere.view_init(elev=args.elev, azim=args.azim)
    ensemble_suffix = f" — ensemble: {len(ensemble_pairs)} par(es)" if ensemble_pairs else ""
    ax_sphere.set_title(
        f"{tag} — shell {shell_key} — n_level={args.n_level}\n"
        f"trinca #{ti} ({args.example if args.triplet_index is None else 'manual'}, "
        f"valid={is_valid}) — residual={residual_deg:.1f}°, gap={gap_deg:.1f}°, t={t_frac:.2f}"
        f"{ensemble_suffix}\n"
        f"pontos claros/translúcidos = espelho antipodal (-v, mesma direção física)",
        fontsize=10)
    ax_sphere.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_sphere.set_xticks([]); ax_sphere.set_yticks([]); ax_sphere.set_zticks([])

    # ---------- DWIs da trinca ----------
    data, affine, header = load_dwi(e.dwi_path)
    b0_mean = data[..., shells[0]].mean(axis=-1)
    mask = load_or_build_mask(e.dwi_path, b0_mean, mask_suffix=args.mask_suffix)
    mask_bool = mask.astype(bool)

    shell_vals = data[..., shell_all_idx][mask_bool]
    xmax = float(np.percentile(shell_vals, 99)) if shell_vals.size else 1.0
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = 1.0
    signal = data / xmax

    vol_a_full = signal[..., idx_a]
    vol_b_full = signal[..., idx_b]
    vol_t_full = signal[..., idx_t]
    blend_full = (1.0 - t_frac) * vol_a_full + t_frac * vol_b_full
    diff_full = np.abs(vol_t_full - blend_full)

    axis = args.slice_axis
    if not args.no_crop and mask_bool.any():
        # bounding box nos OUTROS dois eixos, pra recortar a fatia exibida
        other_axes = [a for a in range(3) if a != axis]
        crop_slices = [slice(None)] * 3
        for oa in other_axes:
            proj = mask_bool.any(axis=tuple(a for a in range(3) if a != oa))
            nzo = np.where(proj)[0]
            if nzo.size:
                crop_slices[oa] = slice(max(0, nzo[0] - 2), min(mask_bool.shape[oa], nzo[-1] + 3))
    else:
        crop_slices = [slice(None)] * 3

    if args.slice_index is not None:
        sidx = args.slice_index
    else:
        nz_axis = np.where(mask_bool.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        sidx = int(nz_axis[len(nz_axis) // 2]) if nz_axis.size else data.shape[axis] // 2

    def _slice2d(vol3d):
        sl = list(crop_slices)
        sl[axis] = sidx
        return np.transpose(vol3d[tuple(sl)])  # transpõe so pra exibir com origin='lower' legível

    panels = [
        ("vol_a (bvec_a, entrada)", _slice2d(vol_a_full), "gray"),
        ("vol_b (bvec_b, entrada)", _slice2d(vol_b_full), "gray"),
        ("alvo REAL (bvec_alvo)", _slice2d(vol_t_full), "gray"),
        (f"blend ingênuo\n(1-t)·a + t·b, t={t_frac:.2f}", _slice2d(blend_full), "gray"),
        ("|alvo real − blend ingênuo|", _slice2d(diff_full), "hot"),
    ]
    vmax_signal = np.percentile(np.concatenate([p[1].ravel() for p in panels[:4]]), 99.5)
    vmax_diff = np.percentile(panels[4][1].ravel(), 99.5) or 1e-6

    # Figura separada (nao um subplot da esfera) -- mais legivel que tentar
    # encaixar os 5 paineis de imagem 2D junto do plot 3D numa unica figura.
    fig2 = plt.figure(figsize=(16, 3.6))
    for col, (title, img, cmap) in enumerate(panels):
        ax_img = fig2.add_subplot(1, 5, col + 1)
        vmax = vmax_diff if cmap == "hot" else vmax_signal
        im = ax_img.imshow(img, cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        ax_img.set_title(title, fontsize=9)
        ax_img.axis("off")
        fig2.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    fig2.suptitle(f"{tag} — trinca #{ti} — fatia eixo={axis} índice={sidx}"
                  f"{' (recortada pela máscara)' if not args.no_crop else ''}", fontsize=10)
    fig2.tight_layout()

    out_path = Path(args.out)
    sphere_path = out_path.with_name(out_path.stem + "_esfera" + out_path.suffix)
    dwi_path_out = out_path.with_name(out_path.stem + "_dwis" + out_path.suffix)
    fig.tight_layout()
    fig.savefig(sphere_path, dpi=150)
    fig2.savefig(dwi_path_out, dpi=150)
    print(f"[ok] esfera salva em: {sphere_path}")
    print(f"[ok] DWIs da trinca salvas em: {dwi_path_out}")

    if not args.no_html:
        html_path = out_path.with_name(out_path.stem + "_esfera_interativo.html")
        example_label = args.example if args.triplet_index is None else "manual"
        _write_sphere_html(html_path, shell_bvecs, shell_all_idx, is_input_of_shell,
                            bvec_a, bvec_b, bvec_t, idx_a, idx_b, idx_t, arc,
                            tag, shell_key, args.n_level, ti, example_label,
                            is_valid, residual_deg, gap_deg, t_frac,
                            ensemble_arcs=ensemble_arcs)
        print(f"[ok] esfera interativa (HTML, abra no navegador) salva em: {html_path}")


if __name__ == "__main__":
    main()