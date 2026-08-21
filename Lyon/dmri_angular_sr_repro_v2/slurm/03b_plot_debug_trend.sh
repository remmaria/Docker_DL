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
#SBATCH --error=logs/plot_trend.%A_%a.err
#SBATCH --output=logs/plot_trend.%A_%a.out

source "./00_env_common.sh"

log_path=/ix1/tibrahim/rmm270/Docker_DL/Lyon/dmri_angular_sr_repro_v2/slurm/logs/train.3495875_4294967294.out
png_out=/ix1/tibrahim/rmm270/Docker_DL/Lyon/dmri_angular_sr_repro_v2/slurm/logs/evolve_3495875_4294967294.png
csv_out=/ix1/tibrahim/rmm270/Docker_DL/Lyon/dmri_angular_sr_repro_v2/slurm/logs/evolve_3495875_4294967294.csv

python /ix1/tibrahim/rmm270/Docker_DL/Lyon/dmri_angular_sr_repro_v2/scripts/plot_debug_trend.py --log $log_path --out $png_out --csv $csv_out