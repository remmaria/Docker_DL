#!/bin/bash
#SBATCH --job-name=train_debug
#SBATCH --cluster=htc
#SBATCH --partition=preempt
# SBATCH --gres=gpu:1
# SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/merge_csvs.%A_%a.err
#SBATCH --output=logs/merge_csvs.%A_%a.out

# sbatch 05b_merge_shard_csvs.sh 

source "./00_env_common.sh"

python /ix1/tibrahim/rmm270/Docker_DL/Lyon/dmri_angular_sr_repro_v2/scripts/merge_shard_csvs.py --dir /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir/metrics
python /ix1/tibrahim/rmm270/Docker_DL/Lyon/dmri_angular_sr_repro_v2/scripts/merge_shard_csvs.py --dir /ix1/tibrahim/rmm270/Docker_DL/Lyon/work_dir/downstream