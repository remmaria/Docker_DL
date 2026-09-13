#!/bin/bash
#SBATCH --job-name=dmri_electro_calib
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-04:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/electro_calib.%J.err
#SBATCH --output=logs/electro_calib.%J.out
#
# Etapa 2e (calibracao, opcional): descobre, por (shell, n_level), o menor
# n_starts que ja fica "bom o bastante" (dentro de TOL_REL da energia
# assintotica testada) pra electrostatic_repulsion_sampling -- responde a
# ressalva ja documentada no codigo de que n_starts=80 so foi validado pra
# N=16. So LE bval/bvec de alguns sujeitos amostrados (sem GPU), mas a
# grade completa pode ser lenta -- por isso o --time generoso acima; se for
# restringir SHELLS/LEVELS a poucos combos, pode reduzir. Ver docstring de
# scripts/02e_calibrate_electrostatic_nstarts.py.
#
# Uso (canonico, grade default cobrindo todos os shells/niveis usuais):
#   sbatch slurm/02e_calibrate_electrostatic_nstarts.sh <work_dir>
#
# Uso (RECOMENDADO pra rodar mais rapido -- so os N altos onde o gap
# fps-vs-electrostatic foi observado, ver subsampling_quality*.csv):
#   SHELLS="1000 1500 2000" LEVELS="32 48 54" \
#       sbatch slurm/02e_calibrate_electrostatic_nstarts.sh <work_dir>
#
# Variaveis de ambiente (todas opcionais, com default = grade padrao do
# script python):
#   SHELLS=<lista separada por espaco>      (default: 500 700 750 1000 1500 2000)
#   LEVELS=<lista separada por espaco>      (default: 6 10 16 20 24 32 48 54)
#   N_SUBJECTS=<int>                        (default: 5, sujeitos amostrados por combo)
#   N_STARTS_GRID=<lista separada por espaco> (default: 10 20 40 80 160)
#   TOL_REL=<float>                         (default: 0.01 = 1%)
#   OUT_CSV=<caminho>                       (default: $WORK_DIR/electrostatic_nstarts_calibration.csv;
#                                             passe "-" pra nao gravar CSV)

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02e_calibrate_electrostatic_nstarts.sh <work_dir>}"

source "./00_env_common.sh"

SHELLS="${SHELLS:-500 700 750 1000 1500 2000}"
LEVELS="${LEVELS:-6 10 16 20 24 32 48 54}"
N_SUBJECTS="${N_SUBJECTS:-5}"
N_STARTS_GRID="${N_STARTS_GRID:-10 20 40 80 160}"
TOL_REL="${TOL_REL:-0.01}"
OUT_CSV="${OUT_CSV:-$WORK_DIR/electrostatic_nstarts_calibration.csv}"

echo "SHELLS=$SHELLS LEVELS=$LEVELS N_SUBJECTS=$N_SUBJECTS N_STARTS_GRID=$N_STARTS_GRID TOL_REL=$TOL_REL"

if [[ "$OUT_CSV" == "-" ]]; then
    python scripts/02e_calibrate_electrostatic_nstarts.py \
        --manifest "$WORK_DIR/manifest.csv" \
        --shells $SHELLS \
        --levels $LEVELS \
        --n-subjects "$N_SUBJECTS" \
        --n-starts-grid $N_STARTS_GRID \
        --tol-rel "$TOL_REL"
else
    python scripts/02e_calibrate_electrostatic_nstarts.py \
        --manifest "$WORK_DIR/manifest.csv" \
        --shells $SHELLS \
        --levels $LEVELS \
        --n-subjects "$N_SUBJECTS" \
        --n-starts-grid $N_STARTS_GRID \
        --tol-rel "$TOL_REL" \
        --out-csv "$OUT_CSV"
fi