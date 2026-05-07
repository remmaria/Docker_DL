#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --cluster=gpu
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # importante para dataloader
#SBATCH --mem=128G
#SBATCH --time=0-23:00:00  
#SBATCH --account=tibrahim
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load apptainer

export APPTAINERENV_PYTHONWARNINGS="ignore::DeprecationWarning"

apptainer exec --nv --cleanenv \
  -B /ix1/tibrahim/rmm270:/ix1/tibrahim/rmm270 \
  /ix1/tibrahim/rmm270/UTILITIES/pytorch_24.12-py3.sif \
  python3 -s -u inference.py --job_id "$SLURM_JOB_ID"