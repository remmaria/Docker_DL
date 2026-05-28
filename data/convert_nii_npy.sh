#!/bin/bash
#SBATCH --job-name=nii_npy
#SBATCH --cluster=htc
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-23:00:00  
#SBATCH --account=tibrahim
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --array=1-99

source /ix1/tibrahim/rmm270/UTILITIES/env_container/bin/activate


SUBJ_LIST="sessions.txt"
session_folder=/ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias

# Lógica para distribuir as linhas do TXT entre os jobs
NUM_JOBS=99
TOTAL_SUJEITOS=$(wc -l < "$SUBJ_LIST")

echo "Iniciando Job Array $SLURM_ARRAY_TASK_ID para um total de $TOTAL_SUJEITOS sujeitos."

for (( i=$SLURM_ARRAY_TASK_ID; i<=$TOTAL_SUJEITOS; i+=$NUM_JOBS )); do
    SESSION=$(sed -n ${i}p $SUBJ_LIST)
    SESSION_PATH=${session_folder}/${SESSION}
    if [ -d "$SESSION_PATH" ]; then
        echo  "Processando $SESSION_PATH (linha $i)"
        python3 convert_nii_npy.py $SESSION_PATH
    fi
done
