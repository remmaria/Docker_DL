#!/bin/bash
#SBATCH --job-name=env
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 # importante para dataloader
#SBATCH --mem=8G
#SBATCH --time=0-23:00:00  
#SBATCH --account=tibrahim
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# limpa módulos antigos
module purge

# carrega anaconda
module load anaconda3/2025.7.0-2-python_3.11

#python -m venv /ix1/tibrahim/rmm270/Docker_DL/running/claude/qshine_env

source /ix1/tibrahim/rmm270/Docker_DL/running/claude/qshine_env/bin/activate

#pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 # versão compatível com CUDA 12.9 (l40s e a100 - CUDA 12.9)

pip install -r requirements.txt

#import torch
#print(torch.version.cuda)        # deve mostrar 12.4
#print(torch.cuda.is_available()) # True
#print(torch.cuda.get_device_name(0))  # NVIDIA L40S