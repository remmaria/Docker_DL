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


def build_manifest(data_root: str, tol: float = 100.0, name_suffix: str = "_geomcorr") -> list[SubjectEntry]:
    from .gradients import load_bval_bvec, split_shells

    entries = []
    for item in discover_dwi_files(data_root, name_suffix=name_suffix):
        bvals, _ = load_bval_bvec(str(item["bval_path"]), str(item["bvec_path"]))
        shells = split_shells(bvals, tol=tol)
        n_b0 = len(shells.get(0, []))
        shell_keys = sorted(k for k in shells.keys() if k != 0)
        n_shells = len(shell_keys)
        protocol = "multi_shell" if n_shells > 1 else "single_shell"
        shell_dirs = "|".join(f"{int(k)}:{len(shells[k])}" for k in shell_keys)
        entries.append(SubjectEntry(
            subject=item["subject"],
            session=item["session"],
            study=item["study"],
            protocol=protocol,
            dwi_path=str(item["dwi_path"]),
            bval_path=str(item["bval_path"]),
            bvec_path=str(item["bvec_path"]),
            n_b0=n_b0,
            shells=",".join(str(int(s)) for s in shell_keys),
            shell_dirs=shell_dirs,
            n_shells=n_shells,
        ))
    return entries


def assign_splits(entries: list[SubjectEntry], train: float = 0.7, val: float = 0.15,
                   seed: int = 42) -> list[SubjectEntry]:
    """Split GLOBAL por sujeito (nao por shell/protocolo especifico).

    Importante: com protocolos tao heterogeneos (b-values e n_direcoes
    variando livremente, shells de multi-shell podendo ser reaproveitadas
    como experimentos "single-shell" para aquele b-value), um mesmo sujeito
    pode participar de varios experimentos diferentes (um por b-value
    alvo). Por isso o split e feito UMA UNICA VEZ por sujeito e reusado em
    todos os experimentos -- garante que um sujeito nunca seja treino num
    experimento e teste em outro, o que complicaria a interpretacao mesmo
    sem causar vazamento estatistico direto.

    Estratifica apenas por `is_multishell` (grosso) -- com N grande (varias
    centenas a milhares de sujeitos) isso ja e suficiente para balancear os
    splits; a heterogeneidade fina de b-values/n_direcoes e tratada depois,
    por experimento, no script de QC (`01b_shell_availability_report.py`).
    """
    rng = random.Random(seed)
    by_protocol: dict[str, list[SubjectEntry]] = {}
    for e in entries:
        by_protocol.setdefault(e.protocol, []).append(e)

    for protocol, group in by_protocol.items():
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
