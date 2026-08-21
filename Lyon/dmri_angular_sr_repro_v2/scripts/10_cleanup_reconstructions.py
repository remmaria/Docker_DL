#!/usr/bin/env python3
"""
Etapa 10 (limpeza, rodar DEPOIS que as metricas de um combo ja foram
calculadas): apaga os `recon_target.nii.gz` (a parte pesada) de
baseline_recon/ e rcae_recon/ para um (shell_b, n_level) especifico, em
TODOS os sujeitos. Mantem target_idx.npy e mask.npy (leves, uteis pra
reproduzir/depurar) e nao toca em nenhum CSV de metricas.

Por que isso existe: nem o treino (etapa 4) nem a reconstrucao (etapa 5) e
etapa 6/7/8) leem de volta os arquivos de reconstrucao uns dos outros --
cada um sempre parte do dwi original + esquema de subamostragem. Os
volumes reconstruidos (recon_target.nii.gz) so existem para as etapas 6
(metricas de sinal) e 7/8 (downstream) lerem uma vez; depois de calculadas
as metricas, o volume em si nao e mais necessario. Rode esta etapa logo
apos 06 e 07 (e 08, se for usar) para o mesmo combo, ANTES de passar pro
proximo combo do experiments.tsv -- assim o pico de disco fica limitado a
poucos combos por vez, nao aos 30 acumulados.

Uso:
    python scripts/10_cleanup_reconstructions.py \
        --work-dir /caminho/work_dir \
        --shell-b 1000 --n-level 10 \
        [--baseline] [--rcae]   # por padrao limpa os dois; use so um se quiser

Seguro rodar de novo (idempotente) -- arquivos ja apagados sao ignorados.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cleanup_recon_dir(recon_dir: Path, shell_b: float, n_level: int):
    if not recon_dir.exists():
        print(f"[aviso] {recon_dir} nao existe, nada a limpar")
        return 0, 0

    freed_bytes = 0
    n_removed = 0
    pattern = f"*/shell{int(shell_b)}/n{n_level}/recon_target.nii.gz"
    for path in recon_dir.glob(pattern):
        freed_bytes += path.stat().st_size
        path.unlink()
        n_removed += 1
    return n_removed, freed_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--baseline", action="store_true", help="limpa so baseline_recon")
    ap.add_argument("--rcae", action="store_true", help="limpa so rcae_recon")
    args = ap.parse_args()

    # sem --baseline/--rcae explicitos, limpa os dois (comportamento default)
    do_baseline = args.baseline or not (args.baseline or args.rcae)
    do_rcae = args.rcae or not (args.baseline or args.rcae)

    work_dir = Path(args.work_dir)
    total_removed = 0
    total_freed = 0

    if do_baseline:
        n, freed = cleanup_recon_dir(work_dir / "baseline_recon", args.shell_b, args.n_level)
        print(f"baseline_recon: {n} arquivos removidos, {freed / 1e9:.2f} GB liberados")
        total_removed += n
        total_freed += freed

    if do_rcae:
        n, freed = cleanup_recon_dir(work_dir / "rcae_recon", args.shell_b, args.n_level)
        print(f"rcae_recon: {n} arquivos removidos, {freed / 1e9:.2f} GB liberados")
        total_removed += n
        total_freed += freed

    print(f"\nTotal: {total_removed} arquivos removidos, {total_freed / 1e9:.2f} GB liberados "
          f"(shell={args.shell_b}, n_level={args.n_level})")
    print("target_idx.npy, mask.npy e todos os CSVs de metricas foram preservados.")


if __name__ == "__main__":
    main()
