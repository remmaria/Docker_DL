#!/bin/bash
#SBATCH --job-name=dmri_compare_sh_loss
#SBATCH --cluster=htc
#SBATCH --partition=preempt
# SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/compare_sh_loss.%A.err
#SBATCH --output=logs/compare_sh_loss.%A.out
#
# Compara dois checkpoints RCAE do MESMO (shell_b, n_level) -- tipicamente
# um treinado com --angular-loss-weight 0.0 (sem SH loss) e outro com
# --angular-loss-weight > 0 (com SH loss, ver slurm/03_train_rcae.sh) --
# reconstruindo o split de teste com CADA UM e rodando o downstream
# DTI/NODDI (etapa 7) em pastas SEPARADAS, pra nao um sobrescrever o outro
# (ver aviso em slurm/03_train_rcae.sh: os dois treinos usam o MESMO
# caminho de checkpoint canonico, entao so a copia permanente em
# rcae_checkpoints/shell<B>_n<N>/runs/<job_id>/best.pt distingue os dois).
#
# Uso:
#   sbatch slurm/06_compare_sh_loss.sh <work_dir> <shell_b> <n_level> \
#       <job_id_sem_sh> <job_id_com_sh>
#
# <job_id_sem_sh>/<job_id_com_sh>: o job_id de CADA treino (o mesmo que
# aparece no nome da pasta rcae_checkpoints/shell<B>_n<N>/runs/<job_id>/ --
# rode "ls rcae_checkpoints/shell<B>_n<N>/runs/" no seu work_dir se nao
# lembrar os dois job_ids).
#
# Saida:
#   $WORK_DIR/rcae_recon_semSH/   e   $WORK_DIR/rcae_recon_comSH/   (reconstrucoes)
#   $WORK_DIR/downstream_semSH/dti_noddi_metrics_shell<B>_n<N>.csv
#   $WORK_DIR/downstream_comSH/dti_noddi_metrics_shell<B>_n<N>.csv
# Compare os dois CSVs (mesmos sujeitos/split test nos dois) pra ver se o
# termo de loss angular mudou FA/MD -- a loss de treino em si NAO e
# comparavel entre os dois runs (ver conversa: com SH loss ativo a loss
# reportada e loss_signal + lambda*loss_angular, escala diferente).
#
# RUN_NODDI=1 (variavel de ambiente) ativa --run-noddi nas duas rodadas,
# mesmo uso que em slurm/05_evaluate_and_downstream.sh.
#
# RECON_SUBJECTS="tag1,tag2" e/ou RECON_LIMIT=1 (variaveis de ambiente,
# mesmo uso que em slurm/04_reconstruct_rcae.sh) restringem a RECONSTRUCAO
# das duas variantes a poucos sujeitos -- util pra um smoke test rapido
# antes de rodar o split de teste inteiro. Ex.:
#   RECON_SUBJECTS="20170920171326_616_20170920171326_616" \
#     sbatch slurm/06_compare_sh_loss.sh <work_dir> 1000 10 <job_id_sem_sh> <job_id_com_sh>

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 06_compare_sh_loss.sh <work_dir> <shell_b> <n_level> <job_id_sem_sh> <job_id_com_sh>}"
SHELL_B="${2:?uso: sbatch 06_compare_sh_loss.sh <work_dir> <shell_b> <n_level> <job_id_sem_sh> <job_id_com_sh>}"
N_LEVEL="${3:?uso: sbatch 06_compare_sh_loss.sh <work_dir> <shell_b> <n_level> <job_id_sem_sh> <job_id_com_sh>}"
JOB_ID_SEM_SH="${4:?uso: sbatch 06_compare_sh_loss.sh <work_dir> <shell_b> <n_level> <job_id_sem_sh> <job_id_com_sh>}"
JOB_ID_COM_SH="${5:?uso: sbatch 06_compare_sh_loss.sh <work_dir> <shell_b> <n_level> <job_id_sem_sh> <job_id_com_sh>}"

source "./00_env_common.sh"

CKPT_DIR="$WORK_DIR/rcae_checkpoints/shell${SHELL_B%.*}_n${N_LEVEL}"

NODDI_FLAG=""
if [[ "${RUN_NODDI:-0}" == "1" ]]; then
    NODDI_FLAG="--run-noddi"
fi

# RECON_SUBJECTS/RECON_LIMIT (variaveis de ambiente, MESMO uso que em
# slurm/04_reconstruct_rcae.sh) -- restringem a RECONSTRUCAO (etapa 5) de
# CADA variante a poucos sujeitos, util pra um smoke test rapido antes de
# rodar o split de teste inteiro nas duas variantes. A etapa 7 (downstream)
# continua percorrendo TODOS os sujeitos do manifesto/split -- sujeitos sem
# reconstrucao 'rcae' so ficam sem linha 'rcae' no CSV (aviso no log,
# "sem reconstrucao rcae para <tag>"), nao e erro.
SUBJECTS_FLAG=()
if [[ -n "${RECON_SUBJECTS:-}" ]]; then
    SUBJECTS_FLAG=(--subjects "$RECON_SUBJECTS")
    echo "RECON_SUBJECTS=$RECON_SUBJECTS -- restringindo reconstrucao a esse(s) sujeito(s) (nas DUAS variantes)"
fi
LIMIT_FLAG=()
if [[ -n "${RECON_LIMIT:-}" ]]; then
    LIMIT_FLAG=(--limit "$RECON_LIMIT")
    echo "RECON_LIMIT=$RECON_LIMIT -- restringindo reconstrucao aos primeiros $RECON_LIMIT sujeito(s) (nas DUAS variantes)"
fi

run_variant() {
    local tag="$1"
    local job_id="$2"
    local ckpt="$CKPT_DIR/runs/$job_id/best.pt"
    if [[ ! -f "$ckpt" ]]; then
        echo "Erro: checkpoint nao encontrado em $ckpt (confira o job_id -- "
        echo "runs disponiveis em: $CKPT_DIR/runs/)"
        exit 1
    fi
    echo "=== variante '$tag' -- checkpoint: $ckpt ==="

    python scripts/05_reconstruct_rcae.py \
        --manifest "$WORK_DIR/manifest.csv" \
        --scheme-dir "$WORK_DIR/subsampling" \
        --checkpoint "$ckpt" \
        --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
        --out-dir "$WORK_DIR/rcae_recon_${tag}" \
        --split test --patch-size 24 --stride 16 \
        "${SUBJECTS_FLAG[@]}" "${LIMIT_FLAG[@]}"

    python scripts/07_downstream_dti_noddi.py \
        --manifest "$WORK_DIR/manifest.csv" \
        --baseline-dir "$WORK_DIR/baseline_recon" \
        --rcae-dir "$WORK_DIR/rcae_recon_${tag}" \
        --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
        --out-dir "$WORK_DIR/downstream_${tag}" \
        $NODDI_FLAG
}

run_variant "semSH" "$JOB_ID_SEM_SH"
run_variant "comSH" "$JOB_ID_COM_SH"

echo "Concluido. Compare:"
echo "  $WORK_DIR/downstream_semSH/dti_noddi_metrics_shell${SHELL_B%.*}_n${N_LEVEL}.csv"
echo "  $WORK_DIR/downstream_comSH/dti_noddi_metrics_shell${SHELL_B%.*}_n${N_LEVEL}.csv"