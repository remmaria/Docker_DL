#!/usr/bin/env python3
"""
Etapa 2b (linha original da tese, retomada como diagnostico quantitativo --
ver protocolo, secao 10.1): para cada sujeito/shell/n_level ja presente no
esquema de subamostragem da etapa 2 (scripts/02_subsample_directions.py),
acha para cada direcao-alvo (held-out) o MELHOR par de direcoes de ENTRADA
que a "abraca" num arco geodesico comum -- o analogo, em q-space, do par
(quadro anterior, quadro seguinte) que a RRIN/RIFE usa para interpolar o
quadro do meio em video.

Ao contrario de video, tres direcoes de gradiente quaisquer normalmente NAO
sao colineares num grande circulo da esfera -- entao aqui, alem de escolher
o par que minimiza o desvio de colinearidade (residual_deg), tambem
marcamos como "invalida" (valid=False) qualquer trinca cujo residuo passe
de --max-residual-deg. A fracao de alvos SEM trinca valida, por n_level, e
em si um resultado quantitativo desta linha (ver secao 10.1 do protocolo):
mede o quanto a suposicao de "fluxo otico entre direcoes vizinhas" deixa de
fazer sentido geometrico conforme a amostragem angular fica mais esparsa.

**Atualizacao (ver protocolo secao 10.2):** colinearidade (residuo baixo)
sozinha NAO garante que o alvo esteja "entre" as duas direcoes do par --
so garante que as tres estao no mesmo grande circulo (um ponto do lado
oposto do circulo tambem teria residuo zero). Um levantamento real mostrou
que, ao minimizar so gap_deg entre os pares colineares, a selecao converge
para pares cada vez mais apertados conforme n_level cresce, mas o alvo
cada vez mais FORA do segmento entre eles (t_frac mediano ultrapassa 1 ja
em n_level=15, chega a ~3 em n_level=50) -- ou seja, extrapolacao, nao
interpolacao, que e uma premissa bem mais fraca (e diferente) do que a
tarefa que RRIN/RIFE foram desenhadas para fazer. `find_best_bracket` agora
(por padrao, `require_between=True`) so aceita como "valido" um par que,
alem de colinear dentro do teto, tambem tenha o alvo genuinamente ENTRE os
dois (0<=t_frac<=1) -- ver `--between` abaixo e o campo novo gravado no
npz.

So usa as direcoes de ENTRADA (input_idx) do n_level em questao como
candidatas ao par -- nao o conjunto completo de direcoes do sujeito -- para
manter a comparacao justa com o RCAE, que tambem so ve essas N direcoes
nesse nivel.

Uso:
    python scripts/02b_build_rrin_triplets.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --out-dir work_dir/subsampling \
        --max-residual-deg 5.0

Saida: <out-dir>/<subject>[_<session>]_rrin_triplets.npz
  Para cada combinacao (shell,n_level) presente no scheme.npz do sujeito,
  grava (na mesma ordem de target_idx daquele scheme):
    "{shell}__{level}__target"       -- copia de target_idx (indices globais)
    "{shell}__{level}__pair_a"       -- indice GLOBAL da 1a direcao do par
    "{shell}__{level}__pair_b"       -- indice GLOBAL da 2a direcao do par
    "{shell}__{level}__t_frac"       -- posicao relativa do alvo no arco (0=coincide com
                                        pair_a, 1=coincide com pair_b -- fora de [0,1] e
                                        extrapolacao, ver "between" abaixo)
    "{shell}__{level}__residual_deg" -- desvio de colinearidade (graus, 0 = perfeito)
    "{shell}__{level}__gap_deg"      -- angulo entre pair_a e pair_b (graus)
    "{shell}__{level}__between"      -- bool, 0<=t_frac<=1 (alvo genuinamente ENTRE o par,
                                        nao extrapolado -- ver nota acima)
    "{shell}__{level}__valid"        -- bool, residual_deg <= --max-residual-deg E between
                                        (as DUAS condicoes agora, nao so a primeira -- ver
                                        nota acima; --no-require-between restaura o
                                        comportamento antigo de valid = so residuo)

  Se --ensemble-m M > 0 (ADITIVO -- ver protocolo secao 14.5 item 1/addendum
  2026-08-27, "ensemble em estrela"), grava TAMBEM, para cada alvo, ate M
  pares de entrada DIVERSOS (nao so o melhor -- ver
  utils/gradients.py:find_star_ensemble_batch), na MESMA ordem de target_idx:
    "{shell}__{level}__ens_pair_a"       -- (n_alvos, M) indice GLOBAL da 1a direcao de
                                            cada par do feixe (-1 = posicao de padding)
    "{shell}__{level}__ens_pair_b"       -- idem, 2a direcao
    "{shell}__{level}__ens_t_frac"       -- (n_alvos, M) t_frac de cada par do feixe
    "{shell}__{level}__ens_residual_deg" -- (n_alvos, M) residuo de cada par do feixe
    "{shell}__{level}__ens_gap_deg"      -- (n_alvos, M) gap_deg de cada par do feixe
    "{shell}__{level}__ens_valid"        -- (n_alvos, M) bool, par real presente nesta
                                            posicao do feixe (mascara de padding -- NAO
                                            e o mesmo criterio de "valido geometricamente"
                                            de "valid" acima, so marca "existe par aqui")
  Estes campos sao INDEPENDENTES dos de par-unico acima (mesmos alvos,
  mesma ordem, mas a selecao do 1o par do feixe pode nao ser IDENTICA a
  "pair_a"/"pair_b" se --no-require-between ou --max-residual-deg
  divergirem entre as duas chamadas -- por padrao usam os MESMOS
  args.max_residual_deg/require_between do par unico desta mesma execucao
  (ver --ensemble-max-residual-deg abaixo para usar um teto DIFERENTE so
  para o feixe), entao na pratica SAO identicos ao par unico quando M>=1
  e nenhum teto separado for passado, ver docstring de
  find_star_ensemble_batch.

  **Nota importante (addendum 2026-08-27, secao 13): com o teto default de
  5.0 graus, em n_level baixo/medio o "pool aceitavel" de cada alvo
  frequentemente tem so 1 membro (nenhum outro par candidato passa no
  teto), entao o feixe efetivo colapsa para M=1 na pratica -- use
  --ensemble-max-residual-deg (mais frouxo que --max-residual-deg) se
  quiser dar ao ensemble um pool de verdade pra diversificar, SEM mexer no
  teto usado pelo par unico (preserva comparacao com RRIN3D/AMT3D/HFD3D ja
  rodados). scripts/02c_diagnose_rrin_triplets.py agora reporta a
  distribuicao de tamanho desse pool por (shell,n_level) -- rode antes de
  decidir o valor.

Ao final, imprime um resumo agregado (todos os sujeitos) da fracao de
alvos validos por (shell,n_level) -- confira antes de treinar a RRIN em
cima disso: n_level onde essa fracao for muito baixa vao gerar poucos
exemplos de treino/avaliacao utilizaveis para esse metodo.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.gradients import load_bval_bvec, find_best_bracket_batch, find_star_ensemble_batch


def _enumerate_combos(scheme_npz):
    """A partir das chaves de um scheme.npz (ver 02_subsample_directions.py),
    devolve a lista de (shell_b:str, n_level:int) presentes."""
    combos = []
    for key in scheme_npz.files:
        if key.endswith("__input"):
            base = key[: -len("__input")]
            shell_str, level_str = base.rsplit("__", 1)
            combos.append((shell_str, int(level_str)))
    return combos


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme-dir", required=True,
                     help="pasta com os <tag>_scheme.npz da etapa 2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-residual-deg", type=float, default=5.0,
                     help="tolerancia de desvio de colinearidade (graus) para considerar "
                          "uma trinca valida (default 5.0 -- bem apertado de proposito; "
                          "o objetivo e medir onde a suposicao de VFI realmente se sustenta, "
                          "nao forcar trincas ruins a passar)")
    ap.add_argument("--no-require-between", action="store_true",
                     help="desliga a exigencia de 0<=t_frac<=1 (alvo genuinamente ENTRE o "
                          "par, nao extrapolado) para marcar valid=True -- restaura o "
                          "comportamento antigo (valid = so residual_deg <= teto, mesmo que "
                          "extrapole). Ver nota no topo do arquivo/protocolo secao 10.2. "
                          "So use para comparar as duas versoes -- o default (exigir "
                          "between) e o comportamento recomendado.")
    ap.add_argument("--subjects", default=None,
                     help="lista separada por virgula de tags (subject ou subject_session) "
                          "para rodar so um subconjunto -- util pra uma PREVIA rapida (1-2 "
                          "sujeitos) antes de rodar o dataset inteiro, ex. depois de mudar "
                          "--max-residual-deg ou --no-require-between. Mesma convencao de "
                          "--subjects em 05_reconstruct_rcae.py.")
    ap.add_argument("--limit", type=int, default=None,
                     help="processa so os primeiros N sujeitos do manifesto (apos --subjects, "
                          "se ambos forem passados) -- outra forma de fazer uma previa rapida.")
    ap.add_argument("--ensemble-m", type=int, default=0,
                     help="ADITIVO (default 0 = desligado, npz identico a antes): quando > 0, "
                          "TAMBEM grava, para cada alvo, um 'feixe em estrela' de ate M pares de "
                          "entrada DIVERSOS (nao so o melhor par -- ver protocolo secao 14.5 item "
                          "1/addendum 2026-08-27, e utils/gradients.py:find_star_ensemble_batch), "
                          "usado pelo treino/reconstrucao do ensemble em estrela "
                          "(scripts/04e_train_rrin_star.py / model/rrin3d_star.py). Os campos "
                          "'pair_a'/'pair_b'/... de sempre (par UNICO, ver acima) continuam sendo "
                          "gravados exatamente como antes -- isso so ACRESCENTA campos novos "
                          "'{base}__ens_pair_a' etc., nao muda nada do que ja existe, entao os "
                          "scripts/checkpoints RRIN3D/AMT3D/HFD3D (single-pair) nao sao afetados "
                          "nem precisam ser regenerados so por causa desta flag.")
    ap.add_argument("--ensemble-max-residual-deg", type=float, default=None,
                     help="ADITIVO (default None = usa o MESMO valor de --max-residual-deg, "
                          "comportamento de sempre): teto de residuo SEPARADO, so para montar o "
                          "pool do 'feixe em estrela' (--ensemble-m), sem afetar o teto do par "
                          "UNICO (--max-residual-deg continua controlando pair_a/pair_b/valid "
                          "normalmente). Motivacao (ver addendum 2026-08-27, secao 13): com o "
                          "teto apertado de sempre (default 5.0), o pool aceitavel por alvo "
                          "costuma ter so 1 membro em n_level baixo/medio, entao o ensemble "
                          "colapsa pra M=1 na pratica -- passar um valor mais frouxo aqui (ex. "
                          "15.0) da ao ensemble candidatos de verdade pra diversificar, deixando "
                          "a cabeca de fusao aprendida (ver model/rrin3d_star.py) decidir quanto "
                          "confiar em cada par, SEM contaminar a comparacao ja feita de "
                          "RRIN3D/AMT3D no par unico (que usa --max-residual-deg, nao este). "
                          "Sem efeito se --ensemble-m nao for passado (>0).")
    args = ap.parse_args()
    require_between = not args.no_require_between

    entries = load_manifest(args.manifest)

    def _tag_of(e):
        return e.subject if not e.session else f"{e.subject}_{e.session}"

    if args.subjects:
        wanted = {t.strip() for t in args.subjects.split(",") if t.strip()}
        entries = [e for e in entries if _tag_of(e) in wanted]
        found = {_tag_of(e) for e in entries}
        missing = wanted - found
        if missing:
            print(f"[aviso] --subjects pediu {sorted(missing)}, mas nao encontrei no "
                  f"manifesto.", flush=True)
        if not entries:
            sys.exit("Nenhum dos sujeitos pedidos em --subjects foi encontrado -- nada a fazer.")
    if args.limit is not None:
        entries = entries[: args.limit]
        print(f"[previa] --limit {args.limit} -- rodando so {len(entries)} sujeito(s)", flush=True)

    scheme_dir = Path(args.scheme_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # agregados globais para o resumo final: {(shell,n_level): [n_valid, n_total]}
    agg = defaultdict(lambda: [0, 0])

    for e in entries:
        tag = e.subject if not e.session else f"{e.subject}_{e.session}"
        scheme_path = scheme_dir / f"{tag}_scheme.npz"
        if not scheme_path.exists():
            print(f"[aviso] {tag}: sem scheme.npz em {scheme_path}, pulando")
            continue

        scheme = np.load(scheme_path)
        _, bvecs = load_bval_bvec(e.bval_path, e.bvec_path)

        save_dict = {}
        for shell_str, n_level in _enumerate_combos(scheme):
            base = f"{shell_str}__{n_level}"
            input_idx = scheme[f"{base}__input"]
            target_idx = scheme[f"{base}__target"]
            if input_idx.shape[0] < 2 or target_idx.shape[0] == 0:
                continue

            input_bvecs = bvecs[input_idx]
            target_bvecs = bvecs[target_idx]

            # find_best_bracket_batch: mesma logica de find_best_bracket, mas
            # vetorizada sobre TODOS os alvos deste combo de uma vez so (~800x
            # mais rapido que chamar find_best_bracket num loop Python por
            # alvo -- ver utils/gradients.py -- e verificado numericamente
            # identico, nao so mais rapido).
            result = find_best_bracket_batch(input_bvecs, target_bvecs,
                                              max_residual_deg=args.max_residual_deg,
                                              require_between=require_between)
            pair_a = input_idx[result["i"]]
            pair_b = input_idx[result["j"]]
            t_frac = result["t_frac"]
            residual_deg = result["residual_deg"]
            gap_deg = result["gap_deg"]
            between = result["between"]
            colinear_ok = residual_deg <= args.max_residual_deg
            valid = colinear_ok & (between if require_between else True)

            save_dict[f"{base}__target"] = target_idx
            save_dict[f"{base}__pair_a"] = pair_a
            save_dict[f"{base}__pair_b"] = pair_b
            save_dict[f"{base}__t_frac"] = t_frac
            save_dict[f"{base}__residual_deg"] = residual_deg
            save_dict[f"{base}__gap_deg"] = gap_deg
            save_dict[f"{base}__between"] = between
            save_dict[f"{base}__valid"] = valid

            if args.ensemble_m > 0:
                # ADITIVO -- ver docstring do modulo e utils/gradients.py:
                # find_star_ensemble_batch. Mesmos input_bvecs/target_bvecs,
                # mesmo require_between desta mesma chamada -- so o teto de
                # residuo pode divergir do par unico se --ensemble-max-residual-deg
                # tiver sido passado (default None = usa o mesmo de sempre).
                ens_max_residual = (args.ensemble_max_residual_deg
                                     if args.ensemble_max_residual_deg is not None
                                     else args.max_residual_deg)
                ens = find_star_ensemble_batch(input_bvecs, target_bvecs, args.ensemble_m,
                                                max_residual_deg=ens_max_residual,
                                                require_between=require_between)
                ens_i, ens_j = ens["i"], ens["j"]  # (n_alvos, M), -1 = padding
                pad = ens_i < 0
                ens_pair_a = np.where(pad, -1, input_idx[np.clip(ens_i, 0, None)])
                ens_pair_b = np.where(pad, -1, input_idx[np.clip(ens_j, 0, None)])
                save_dict[f"{base}__ens_pair_a"] = ens_pair_a
                save_dict[f"{base}__ens_pair_b"] = ens_pair_b
                save_dict[f"{base}__ens_t_frac"] = ens["t_frac"]
                save_dict[f"{base}__ens_residual_deg"] = ens["residual_deg"]
                save_dict[f"{base}__ens_gap_deg"] = ens["gap_deg"]
                save_dict[f"{base}__ens_valid"] = ens["mask"]

            key = (shell_str, n_level)
            agg[key][0] += int(valid.sum())
            agg[key][1] += int(valid.shape[0])

        if not save_dict:
            print(f"[aviso] {tag}: nenhuma combinacao (shell,n_level) com >=2 direcoes "
                  f"de entrada -- nada a gravar")
            continue

        out_path = out_dir / f"{tag}_rrin_triplets.npz"
        np.savez(out_path, **save_dict)
        # 8 chaves por combo sem ensemble, 14 com --ensemble-m>0 (ver docstring
        # do modulo) -- conta pelo numero de combos distintos de verdade
        # (chave "__target"), nao por uma divisao fixa (que quebraria com o
        # numero de campos diferente entre as duas variantes).
        n_combos = sum(1 for k in save_dict if k.endswith("__target"))
        print(f"{tag}: {n_combos} combinacoes (shell,nivel) salvas em {out_path}"
              + (f" (com ensemble-m={args.ensemble_m})" if args.ensemble_m > 0 else ""))

    print(f"\n(require_between={require_between} -- ver --no-require-between "
          "e nota no topo do arquivo se quiser comparar)")
    print("Resumo (todos os sujeitos) -- fracao de alvos com trinca valida "
          f"(residual <= {args.max_residual_deg} graus"
          + (" E alvo entre o par" if require_between else "") + ") por (shell,n_level):")
    for (shell_str, n_level), (n_valid, n_total) in sorted(agg.items()):
        frac = n_valid / n_total if n_total else float("nan")
        print(f"  shell={shell_str:>10s}  n_level={n_level:>3d}  "
              f"{n_valid:5d}/{n_total:5d} validos  ({frac:.1%})")


if __name__ == "__main__":
    main()