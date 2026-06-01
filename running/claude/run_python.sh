#!/bin/bash
#SBATCH --job-name=qshine
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=400G
#SBATCH --time=0-23:00:00  
#SBATCH --account=tibrahim
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out


source /ix1/tibrahim/rmm270/Docker_DL/running/claude/qshine_env/bin/activate

#python sanity_check.py


export WANDB_API_KEY='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'

#python train.py --synthetic --output_dir runs/$SLURM_JOB_ID

python train.py \
    --data_dir /ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias \
    --output_dir runs/$SLURM_JOB_ID \
    --wandb_project qshine
