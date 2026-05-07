#!/bin/bash
#SBATCH --job-name=ds_split
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

python make_split.py \
    --master_csv /ix1/tibrahim/rmm270/DATA/DWIs/studies/COORDS_DL/master.csv \
    --out_dir /ix1/tibrahim/rmm270/DATA/DWIs/studies/COORDS_DL \
    --min_subjects_per_protocol 10 \
    --outliers_txt outliers.txt \
    --val_size 0.15 \
    --test_size 0.15 \
    --seed 42