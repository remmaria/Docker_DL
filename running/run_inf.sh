#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # importante para dataloader
#SBATCH --mem=64G
#SBATCH --time=0-23:00:00  
#SBATCH --account=tibrahim
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# limpa módulos antigos
module purge

# carrega anaconda
module load anaconda3/2025.7.0-2-python_3.11

# ativa conda corretamente
eval "$(conda shell.bash hook)"
conda activate dmri_dl

# impede puxar lixo do ~/.local
export PYTHONNOUSERSITE=1

# opcional: limitar threads de libs CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# roda treino
python inference.py --job_id "$SLURM_JOB_ID"