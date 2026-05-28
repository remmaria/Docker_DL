#!/bin/bash
#SBATCH --job-name=qc_patches
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # importante para dataloader
#SBATCH --mem=32G
#SBATCH --time=0-23:00:00  
#SBATCH --account=tibrahim
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module purge
module load anaconda3/2025.7.0-2-python_3.11

source /ix1/tibrahim/rmm270/UTILITIES/env_container/bin/activate

python qc_patches.py