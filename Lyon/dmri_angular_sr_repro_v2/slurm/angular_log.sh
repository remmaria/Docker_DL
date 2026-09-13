#!/bin/bash
#SBATCH --job-name=ang_log
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/ang_log.%J.err
#SBATCH --output=logs/ang_log.%J.out


set -euo pipefail

source "./00_env_common.sh"

python scripts/angular_log.py