"""
Manifesto do dataset: descoberta de sujeitos (layout proprio, nao-BIDS --
arvore studies/<estudo>/<pasta_sessao>/<nome_base><sufixo>.{nii,nii.gz,bval,bvec}),
validacao basica e split treino/val/teste por sujeito (nunca por volume).
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class SubjectEntry:
    subject: str  # identificador unico = "<estudo>__<pasta_sessao>" (ver discover_dwi_files)
    session: str  # nome cru da pasta de sessao (ex.: "20160914203805_160914-volunteer")
    study: str  # nome da subpasta de estudo (ex.: "all_bias")
    protocol: str  # "single_shell" | "multi_shell" -- derivado, so para leitura humana/QC
    dwi_path: str
    bval_path: str
    bvec_path: str
    n_b0: int
    shells: str  # ex: "700,1000,1500" (sem contar b0) -- so os b-values presentes
    shell_dirs: str  # ex: "700:32|1000:60|1500:60" -- b-value:n_direcoes por shell
    n_shells: int  # quantas shells nao-zero esse sujeito tem (1 = single, >=2 = multi)
    split: str = ""  # preenchido depois: train/val/test
    # Demograficos/acquisicao, lidos de "<data_root>/<study>/<demo_tsv_name>"
    # (ver load_study_info) e casados por SessionID == session -- vazio ("")
    # quando o estudo nao tem esse TSV ou a sessao nao foi encontrada nele.
    # patient_age fica como string (nao float) pra preservar o valor exato
    # do TSV sem arriscar formatacao/arredondamento na escrita do CSV.
    patient_sex: str = ""
    patient_age: str = ""
    acquisition_date: str = ""
    scanner_protocol: str = ""  # coluna "Study" do TSV (ex.: "BRAIN^WPC-7707") -- NAO confundir
    # com o campo `study` acima (nome da subpasta no data_root, ex.: "7TBRP")

    @property
    def is_multishell(self) -> bool:
        return self.n_shells >= 2

    def has_shell(self, b_value: float, tol: float = 25.0) -> bool:
        """Confere se esse sujeito tem uma shell dentro de `tol` s/mm^2 do
        b-value pedido -- use isso (nao comparacao exata) porque escaneres
        as vezes gravam o mesmo protocolo nominal com um b medido levemente
        diferente entre sessoes.
        """
        for pair in self.shell_dirs.split("|"):
            if not pair:
                continue
            b_str, _ = pair.split(":")
            if abs(float(b_str) - b_value) <= tol:
                return True
        return False

    def n_dirs_for_shell(self, b_value: float, tol: float = 25.0):
        for pair in self.shell_dirs.split("|"):
            if not pair:
                continue
            b_str, n_str = pair.split(":")
            if abs(float(b_str) - b_value) <= tol:
                return int(n_str)
        return None


def discover_dwi_files(data_root: str, name_suffix: str = "_geomcorr"):
    """Varre `data_root` recursivamente procurando trios nii(.gz)+bval+bvec
    cujo nome termine em `name_suffix` -- layout tipo
    studies/<estudo>/<pasta_sessao>/<qualquer_coisa><name_suffix>.{bval,bvec,nii|nii.gz}.
    Nao exige convencao BIDS nem profundidade fixa de pastas.

    So o sufixo de nome importa para o casamento (bval/bvec/nii com o mesmo
    "stem"); outros arquivos na mesma pasta (mascaras, mapas de FA/MD etc.)
    sao ignorados automaticamente porque nao tem bval/bvec companheiro com
    esse sufixo.

    Retorna lista de dicts: {subject, session, study, dwi_path, bval_path, bvec_path}.
    `subject` e "<estudo>__<pasta_sessao>" (unico mesmo se pastas de sessao
    se repetirem entre estudos); `session` e so o nome cru da pasta.
    """
    root = Path(data_root)
    found = []
    for bval_path in sorted(root.glob(f"**/*{name_suffix}.bval")):
        stem = str(bval_path)[: -len(".bval")]
        bvec_path = Path(stem + ".bvec")
        nii_path = None
        for ext in (".nii.gz", ".nii"):
            candidate = Path(stem + ext)
            if candidate.exists():
                nii_path = candidate
                break
        if not bvec_path.exists() or nii_path is None:
            print(f"[aviso] pulando {stem}: bvec ou nii(.gz) ausente ao lado do bval")
            continue

        rel_parts = bval_path.relative_to(root).parts
        session = rel_parts[-2] if len(rel_parts) >= 2 else Path(stem).name
        study = rel_parts[0] if len(rel_parts) >= 3 else ""
        subject = f"{study}__{session}" if study else session

        found.append({
            "subject": subject, "session": session, "study": study,
            "dwi_path": nii_path, "bval_path": bval_path, "bvec_path": bvec_path,
        })
    return found


def _read_tsv_by_session(path: Path) -> dict[str, dict]:
    """Le um TSV com cabecalho incluindo 'SessionID' e devolve
    {SessionID: {coluna: valor}}. Devolve {} (silenciosamente) se o arquivo
    nao existir. Colunas alem de SessionID sao repassadas como estao
    (mas com espacos nas pontas removidos) -- quem chama decide o que fazer
    com cada uma.

    Robusto a alguns problemas comuns de TSV exportado de outro sistema:
    BOM no inicio do arquivo (encoding utf-8-sig), nomes de coluna ou
    valores com espacos extras nas pontas, e linha em branco no fim.
    """
    if not path.exists():
        return {}
    rows_by_sid: dict[str, list[dict]] = {}
    n_rows = 0
    n_blank_sid = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t")
        raw_fieldnames = reader.fieldnames or []
        fieldnames = [(fn or "").strip() for fn in raw_fieldnames]
        if "SessionID" not in fieldnames:
            print(f"[aviso] {path}: sem coluna 'SessionID' no cabecalho (colunas encontradas: "
                  f"{fieldnames}) -- ignorando")
            return {}
        for raw_row in reader:
            n_rows += 1
            row = {}
            for k, v in raw_row.items():
                k2 = (k or "").strip()
                v2 = v.strip() if isinstance(v, str) else v
                row[k2] = v2
            sid = row.get("SessionID") or ""
            if not sid:
                n_blank_sid += 1
                continue
            rows_by_sid.setdefault(sid, []).append(row)

    info: dict[str, dict] = {}
    n_dup_same = 0
    n_dup_conflict = 0
    conflict_examples = []
    for sid, rows in rows_by_sid.items():
        info[sid] = rows[0]  # mantem a primeira ocorrencia
        if len(rows) > 1:
            studies = sorted({(r.get("Study") or "").strip() for r in rows})
            if len(studies) > 1:
                n_dup_conflict += 1
                if len(conflict_examples) < 10:
                    conflict_examples.append((sid, studies))
                print(f"[aviso] {path}: SessionID '{sid}' aparece {len(rows)}x no TSV com "
                      f"'Study' DIFERENTE entre as linhas {studies} -- mantendo a primeira "
                      f"ocorrencia ('{rows[0].get('Study', '')}'). Se essa sessao existir em "
                      f"mais de um estudo/protocolo, o manifest.csv pode estar atribuindo o "
                      f"'study'/'scanner_protocol' errado pra ela.")
            else:
                n_dup_same += 1
                print(f"[aviso] {path}: SessionID '{sid}' duplicado ({len(rows)}x, mesmo "
                      f"'Study') -- mantendo a primeira ocorrencia")

    print(f"[info] {path}: {n_rows} linha(s) lidas, {len(info)} SessionID unicos "
          f"({n_blank_sid} linha(s) com SessionID vazio ignoradas, {n_dup_same} SessionID "
          f"duplicado(s) com mesmo Study, {n_dup_conflict} SessionID duplicado(s) com Study "
          f"CONFLITANTE)")
    if conflict_examples:
        print(f"[aviso] exemplos de SessionID com Study conflitante (ate 10): {conflict_examples}")
    return info


def load_study_info(data_root: str, study: str, demo_tsv_name: str = "info.tsv") -> dict[str, dict]:
    """Le "<data_root>/<study>/<demo_tsv_name>" (TSV com cabecalho, ex.:
    SessionID/PatientSex/AcquisitionDate/PatientAge/Study) e devolve um dict
    {SessionID: {coluna: valor}}. Devolve {} (silenciosamente) se o arquivo
    nao existir -- nem todo estudo tem esse TSV, isso e' esperado, nao erro.
    """
    return _read_tsv_by_session(Path(data_root) / study / demo_tsv_name)


def is_global_tsv_path(demo_tsv_name: str) -> bool:
    """True quando `demo_tsv_name` e um caminho direto pra UM UNICO TSV
    global (absoluto, ex.: "/ix1/.../7TBRP/info.tsv", ou relativo mas que
    ja existe como arquivo a partir do diretorio atual) -- usado quando so
    existe um TSV de demograficos pra todos os sujeitos, independente de
    subpasta de estudo. Quando False, `demo_tsv_name` e tratado como nome
    de arquivo procurado DENTRO de cada "<data_root>/<estudo>/" (comportamento
    original, para quando cada subestudo tem seu proprio TSV).
    """
    p = Path(demo_tsv_name)
    return p.is_absolute() or p.exists()


def build_manifest(data_root: str, tol: float = 100.0, name_suffix: str = "_geomcorr",
                    demo_tsv_name: str = "info.tsv") -> list[SubjectEntry]:
    from .gradients import load_bval_bvec, split_shells

    entries = []
    use_global_tsv = is_global_tsv_path(demo_tsv_name)
    global_info: dict[str, dict] = {}
    n_demo_matched = 0
    n_demo_total_studies_with_tsv = 0

    if use_global_tsv:
        global_info = _read_tsv_by_session(Path(demo_tsv_name))
        n_demo_total_studies_with_tsv = 1 if global_info else 0
        print(f"[info] demo-tsv-name '{demo_tsv_name}' e um caminho direto -- usando como TSV "
              f"UNICO/GLOBAL (casado por SessionID em todos os sujeitos, ignorando subpasta de "
              f"estudo) em vez de procurar '{demo_tsv_name}' dentro de cada <data_root>/<estudo>/")
        global_info_lower = {sid.lower(): sid for sid in global_info}
    else:
        study_info_cache: dict[str, dict[str, dict]] = {}

    unmatched_sessions = []

    for item in discover_dwi_files(data_root, name_suffix=name_suffix):
        bvals, _ = load_bval_bvec(str(item["bval_path"]), str(item["bvec_path"]))
        shells = split_shells(bvals, tol=tol)
        n_b0 = len(shells.get(0, []))
        shell_keys = sorted(k for k in shells.keys() if k != 0)
        n_shells = len(shell_keys)
        protocol = "multi_shell" if n_shells > 1 else "single_shell"
        shell_dirs = "|".join(f"{int(k)}:{len(shells[k])}" for k in shell_keys)

        if use_global_tsv:
            demo = global_info.get(item["session"], {})
            if not demo:
                unmatched_sessions.append(item["session"])
        else:
            study = item["study"]
            if study and study not in study_info_cache:
                study_info_cache[study] = load_study_info(data_root, study, demo_tsv_name=demo_tsv_name)
                if study_info_cache[study]:
                    n_demo_total_studies_with_tsv += 1
            demo = study_info_cache.get(study, {}).get(item["session"], {})
        if demo:
            n_demo_matched += 1

        scanner_protocol = (demo.get("Study") or "").strip()
        # `study` (subpasta em disco) as vezes fica vazio quando --data-root
        # ja aponta direto pra pasta que contem as sessoes (sem nivel de
        # subpasta de estudo no meio -- ver discover_dwi_files). Nesse caso,
        # usa a coluna "Study" do TSV como fallback pra `study` tambem, pra
        # nao ficar vazio quando ha essa info disponivel.
        effective_study = item["study"] or scanner_protocol

        entries.append(SubjectEntry(
            subject=item["subject"],
            session=item["session"],
            study=effective_study,
            protocol=protocol,
            dwi_path=str(item["dwi_path"]),
            bval_path=str(item["bval_path"]),
            bvec_path=str(item["bvec_path"]),
            n_b0=n_b0,
            shells=",".join(str(int(s)) for s in shell_keys),
            shell_dirs=shell_dirs,
            n_shells=n_shells,
            patient_sex=(demo.get("PatientSex") or "").strip(),
            patient_age=(demo.get("PatientAge") or "").strip(),
            acquisition_date=(demo.get("AcquisitionDate") or "").strip(),
            scanner_protocol=scanner_protocol,
        ))

    if use_global_tsv:
        if global_info:
            print(f"[info] demograficos: {n_demo_matched}/{len(entries)} sujeitos casados via "
                  f"TSV global '{demo_tsv_name}' ({len(global_info)} SessionID no arquivo)")
            if unmatched_sessions:
                sample = unmatched_sessions[:5]
                print(f"[aviso] {len(unmatched_sessions)} sessao(oes) NAO encontradas no TSV por "
                      f"SessionID exato. Exemplo(s): {sample}")
                # tenta achar por que -- bate so ignorando maiusc/minusc?
                ci_hits = [(s, global_info_lower[s.lower()]) for s in sample if s.lower() in global_info_lower]
                if ci_hits:
                    print(f"[aviso]   destes, batem ignorando maiusculas/minusculas: {ci_hits} "
                          f"-- o TSV tem o SessionID grafado diferente (mai/minusculo) do nome da pasta")
        else:
            print(f"[aviso] demo-tsv-name '{demo_tsv_name}' aponta pra um caminho mas o TSV "
                  f"nao foi lido (nao existe ou sem coluna SessionID) -- todos os campos "
                  f"demograficos ficaram vazios")
    elif n_demo_total_studies_with_tsv:
        print(f"[info] demograficos: {n_demo_matched}/{len(entries)} sujeitos casados via "
              f"'{demo_tsv_name}' em {n_demo_total_studies_with_tsv} subestudo(s) que tinham o arquivo "
              f"(subestudos sem '{demo_tsv_name}' ficam com esses campos vazios, sem erro)")
    return entries


def _normalize_sex(patient_sex: str) -> str:
    s = (patient_sex or "").strip().upper()
    return s if s in ("M", "F") else "UNK"


def _proportional_split_group(rng: random.Random, group: list[SubjectEntry],
                               train: float, val: float):
    """Embaralha `group` e marca .split in-place (train/val/test), pelas
    proporcoes pedidas (arredondamento simples, mesma logica que o
    assign_splits original tinha por protocolo -- extraida aqui pra ser
    reusada por qualquer chave de estratificacao)."""
    idx = list(range(len(group)))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    for i, pos in enumerate(idx):
        if i < n_train:
            group[pos].split = "train"
        elif i < n_train + n_val:
            group[pos].split = "val"
        else:
            group[pos].split = "test"


def assign_splits(entries: list[SubjectEntry], train: float = 0.7, val: float = 0.15,
                   seed: int = 42, stratify_sex: bool = True, stratify_study: bool = True,
                   min_stratum_size: int = 8) -> list[SubjectEntry]:
    """Split GLOBAL por sujeito (nao por shell/protocolo especifico).

    Importante: com protocolos tao heterogeneos (b-values e n_direcoes
    variando livremente, shells de multi-shell podendo ser reaproveitadas
    como experimentos "single-shell" para aquele b-value), um mesmo sujeito
    pode participar de varios experimentos diferentes (um por b-value
    alvo). Por isso o split e feito UMA UNICA VEZ por sujeito e reusado em
    todos os experimentos -- garante que um sujeito nunca seja treino num
    experimento e teste em outro, o que complicaria a interpretacao mesmo
    sem causar vazamento estatistico direto.

    Estratificacao (hierarquica, com fallback pra grupos pequenos demais):
    sempre estratifica por `protocol` (single_shell/multi_shell, grosso).
    Quando stratify_sex=True (default), tambem cruza com `patient_sex`
    (M/F/UNK -- vazio ou outro valor vira "UNK", uma categoria propria, nao
    descartada). Quando stratify_study=True (default), tambem cruza com
    `study` (o "estudo"/protocolo de scanner de origem, ex. "BRAIN^WPC-7707"
    -- ver utils.manifest.build_manifest sobre de onde isso vem quando o
    data-root nao tem nivel de subpasta de estudo). Motivacao: com ~20
    estudos/protocolos de scanner diferentes e composicao de sexo
    desbalanceada no dataset, um split puramente aleatorio (so por
    protocol) pode por acaso ficar com composicao de sexo ou de estudo bem
    diferente entre treino/val/teste, confundindo a interpretacao se algum
    estudo/protocolo tiver efeito sistematico no sinal.

    Grupos PEQUENOS DEMAIS pra estratificar com sentido (ex.: um estudo com
    so 1-4 sujeitos -- nesse caso arredondar 15% da proporcao de val/teste
    da 0, jogando o grupo inteiro artificialmente pro treino) sao
    detectados via `min_stratum_size` (default 8 -- com round(), grupos
    menores que isso tem chance real de val OU teste saírem vazios) e
    RECAEM pra uma chave de estratificacao mais grosseira: primeiro tenta
    (protocol, sex) sem study; se AINDA assim o grupo (protocol, sex) for
    pequeno demais (raro, so aconteceria com stratify_sex=True e uma
    combinacao rara tipo multi_shell+UNK), recai para so `protocol`. Isso
    preserva estratificacao fina (por estudo) exatamente onde ela e
    estatisticamente significativa, e evita fragmentacao sem sentido nos
    estudos residuais/pequenos -- que ainda assim continuam sendo
    estratificados por sexo e protocolo, so nao por estudo individualmente.
    """
    rng = random.Random(seed)

    def key_full(e: SubjectEntry):
        parts = [e.protocol]
        if stratify_sex:
            parts.append(_normalize_sex(e.patient_sex))
        if stratify_study:
            parts.append(e.study or "")
        return tuple(parts)

    def key_medium(e: SubjectEntry):
        parts = [e.protocol]
        if stratify_sex:
            parts.append(_normalize_sex(e.patient_sex))
        return tuple(parts)

    def key_coarse(e: SubjectEntry):
        return (e.protocol,)

    # agrupa pela chave mais fina primeiro
    by_full: dict[tuple, list[SubjectEntry]] = {}
    for e in entries:
        by_full.setdefault(key_full(e), []).append(e)

    # separa grupos que sustentam a chave fina dos que precisam recair
    fine_groups = []
    fallback_entries = []
    for k, group in by_full.items():
        if len(group) >= min_stratum_size:
            fine_groups.append(group)
        else:
            fallback_entries.extend(group)

    n_fine = sum(len(g) for g in fine_groups)
    if fallback_entries:
        # tenta a chave media (protocol, sex) pros que sobraram
        by_medium: dict[tuple, list[SubjectEntry]] = {}
        for e in fallback_entries:
            by_medium.setdefault(key_medium(e), []).append(e)
        medium_groups = []
        coarse_leftover = []
        for k, group in by_medium.items():
            if len(group) >= min_stratum_size:
                medium_groups.append(group)
            else:
                coarse_leftover.append(group)
        fine_groups.extend(medium_groups)
        n_medium = sum(len(g) for g in medium_groups)

        if coarse_leftover:
            # ultimo recurso: agrupa so por protocol (praticamente sempre
            # grande o bastante, ha centenas de sujeitos por protocol)
            flat_leftover = [e for group in coarse_leftover for e in group]
            by_coarse: dict[tuple, list[SubjectEntry]] = {}
            for e in flat_leftover:
                by_coarse.setdefault(key_coarse(e), []).append(e)
            fine_groups.extend(by_coarse.values())
            n_coarse = len(flat_leftover)
        else:
            n_coarse = 0
        print(f"[splits] estratificacao: {n_fine} sujeito(s) em grupos finos "
              f"({'protocol+sexo+estudo' if stratify_study else 'protocol+sexo'}), "
              f"{n_medium} recaíram para protocol+sexo (estudo pequeno demais, "
              f"< {min_stratum_size} sujeitos), {n_coarse} recaíram so para protocol "
              f"(mesmo protocol+sexo pequeno demais)")
    else:
        print(f"[splits] estratificacao: {n_fine} sujeito(s), todos os grupos "
              f"('{'protocol+sexo+estudo' if stratify_study else 'protocol'}"
              f"{'+sexo' if stratify_sex and not stratify_study else ''}') "
              f"grandes o bastante (>= {min_stratum_size})")

    for group in fine_groups:
        _proportional_split_group(rng, group, train, val)

    return entries


def save_manifest(entries: list[SubjectEntry], out_csv: str):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(entries[0]).keys()) if entries else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in entries:
            writer.writerow(asdict(e))


def load_manifest(csv_path: str) -> list[SubjectEntry]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [SubjectEntry(**row) for row in reader]
    for r in rows:
        r.n_b0 = int(r.n_b0)
        r.n_shells = int(r.n_shells)
    return rows