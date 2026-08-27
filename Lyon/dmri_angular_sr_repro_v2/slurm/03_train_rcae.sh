#!/bin/bash
#SBATCH --job-name=rcae_olat
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
# SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train.%A_%a.err
#SBATCH --output=logs/train.%A_%a.out
#
# Treino do RCAE (etapa 4), no mesmo padrao do seu script de treino
# original (mesma partition/conta/GPU/cpus). Cada linha de
# slurm/configs/experiments.tsv (shell_b, n_level) vira um item do array --
# assim um `sbatch --array=1-N` roda todos os experimentos em paralelo,
# um job por combinacao, cada um na sua GPU.
#
# Uso:
#   1) edite slurm/configs/experiments.tsv com os (shell_b, n_level) que quer rodar
#   2) conte quantas linhas uteis (nao-comentario) tem no arquivo, ex.:
#        N=$(grep -vc '^#' slurm/configs/experiments.tsv)
#   3) sbatch --array=1-$N slurm/03_train_rcae.sh <work_dir>
#
# Para rodar so uma combinacao especifica (sem array), passe shell_b e
# n_level direto:
#   sbatch slurm/03_train_rcae.sh <work_dir> <shell_b> <n_level>
#
# Termo de loss angular/SH opcional (ver protocolo, secao 9, prioridade 1 --
# scripts/04_train_rcae.py:compute_sh_angular_loss): DESATIVADO por padrao
# (--angular-loss-weight 0.0 no python abaixo, treino identico ao anterior).
# Pra rodar COM o termo ativo (ex.: comparar com/sem no mesmo shell/n_level),
# passe ANGULAR_LOSS_WEIGHT e/ou SH_LOSS_HIGH_ORDER_MIN (variaveis de
# ambiente) na chamada do sbatch -- nao precisa editar este arquivo:
#   ANGULAR_LOSS_WEIGHT=0.5 sbatch slurm/03_train_rcae.sh <work_dir> <shell_b> <n_level>
# Com --q-out 10 (fixo abaixo), so l<=2 e sustentavel (ver conversa no
# protocolo) -- pra rodar o experimento "barato" de l=2 (sem mudar q_out,
# sem risco de memoria/N de sujeitos), baixe o minimo pedido junto:
#   ANGULAR_LOSS_WEIGHT=0.5 SH_LOSS_HIGH_ORDER_MIN=2 sbatch slurm/03_train_rcae.sh <work_dir> <shell_b> <n_level>
# (default de SH_LOSS_HIGH_ORDER_MIN e 4 -- com q_out=10 fica sempre zerado,
# ver aviso que o proprio 04_train_rcae.py imprime no [angular-loss] do log
# se voce esquecer de baixar pra 2 aqui).
# Isso NAO muda o caminho do checkpoint (continua
# rcae_checkpoints/shell<B>_n<N>/best.pt) -- se quiser comparar os dois runs
# (com e sem) sem um sobrescrever o outro, rode em <work_dir> diferentes, ou
# guarde o best.pt de cada um antes de rodar o outro (a copia permanente em
# runs/<job_id>/best.pt tambem serve pra isso).
#
# Resume automatico de checkpoint (ver scripts/04_train_rcae.py): se
# out_dir/<shell>_<n>/last.pt ja existir (de um treino anterior que morreu
# no meio -- OOM, preempcao, timeout), o proximo sbatch retoma dali
# AUTOMATICAMENTE, sem precisar de nada extra aqui. Isso vale mesmo com
# <work_dir> ja separados por variante (com/sem SH loss) -- cada work_dir
# tem seu proprio last.pt, sem risco de misturar.
#
# Se preferir apontar EXPLICITAMENTE pro checkpoint de um job especifico
# (em vez de confiar no last.pt canonico "mais recente" -- por exemplo se
# ficar em duvida sobre qual foi o ultimo run daquele combo), passe
# RESUME_CHECKPOINT com o caminho da copia permanente
# (.../runs/<job_id>/last.pt):
#   RESUME_CHECKPOINT=/caminho/rcae_checkpoints/shell1000_n10/runs/3498743_0/last.pt \
#     sbatch slurm/03_train_rcae.sh <work_dir> 1000 10
# NO_RESUME=1 desativa o resume por completo e comeca do zero (equivalente
# a --no-resume):
#   NO_RESUME=1 sbatch slurm/03_train_rcae.sh <work_dir> <shell_b> <n_level>
#
# DECODER_TYPE=sh (variavel de ambiente, default direct) -- liga
# --decoder-type sh (ver model/rcae.py:Decoder3DSH e protocolo secoes
# 10/15): decoder preve coeficientes SH compartilhados entre as
# direcoes-alvo do item (em vez de sinal por direcao independente),
# convertidos pra sinal por uma multiplicacao pela matriz de base -- vies
# estrutural que forca as predicoes do mesmo voxel a serem consistentes
# como amostras de uma UNICA FOD continua, ideia inspirada na analogia
# OLAT/iluminacao multi-fonte da secao 10. Grava em
# out_dir/shell<B>_n<N>[_sh]_shdec/ (sufixo automatico, nao colide com a
# variante "direct" existente -- pode coexistir com --angular-loss-weight,
# que usa sufixo _sh separado, ver 04_train_rcae.py). Exige treinar do
# ZERO (parametros incompativeis com "direct" -- o script bloqueia resume
# entre as duas variantes com um erro).
#   DECODER_TYPE=sh NO_RESUME=1 sbatch slurm/03_train_rcae.sh <work_dir> <shell_b> <n_level>
# SH_DECODER_LMAX=<l> (default 4, so tem efeito com DECODER_TYPE=sh) --
# ordem SH maxima dos coeficientes previstos (ver --sh-decoder-lmax). AO
# CONTRARIO do ANGULAR_LOSS_WEIGHT, nao exige subir --q-out junto (nao ha
# piso de direcoes-alvo simultaneas -- ver docstring de Decoder3DSH).

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 03_train_rcae.sh <work_dir> [shell_b n_level]}"

EXPERIMENTS_TSV="configs/experiments.tsv"

if [[ -n "${2:-}" && -n "${3:-}" ]]; then
    SHELL_B="$2"
    N_LEVEL="$3"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    LINE=$(grep -v '^#' "$EXPERIMENTS_TSV" | sed -n "${SLURM_ARRAY_TASK_ID}p")
    if [[ -z "$LINE" ]]; then
        echo "Erro: nao ha linha $SLURM_ARRAY_TASK_ID em $EXPERIMENTS_TSV (confira --array=1-N)"
        exit 1
    fi
    SHELL_B=$(echo "$LINE" | cut -f1)
    N_LEVEL=$(echo "$LINE" | cut -f2)
else
    echo "Erro: informe shell_b/n_level como argumentos OU submeta com --array=1-N"
    echo "(N = numero de linhas uteis em $EXPERIMENTS_TSV)"
    exit 1
fi

echo "Treinando RCAE para shell_b=$SHELL_B, n_level=$N_LEVEL"

source "./00_env_common.sh"

# --val-num-workers/--val-max-cached-subjects explicitos (em vez de contar
# so com o default do script): com persistent_workers=True os workers do
# train_loader (--num-workers 8, --max-cached-subjects 6) E do val_loader
# ficam residentes ao MESMO TEMPO a partir da 1a epoca (nenhum dos dois
# morre entre epocas -- ver comentario em scripts/04_train_rcae.py:main),
# entao o pico de RAM e a SOMA dos dois caches. Val nao precisa do mesmo
# paralelismo do treino (passa 1x por epoca, sem shuffle) -- foi um job
# assim (workers de treino + workers de val subindo por cima na 1a
# validacao) que estourou o --mem=32G original (subimos pra 64G abaixo
# como margem extra, mas o ajuste de fundo continua sendo nao duplicar
# paralelismo desnecessario no val_loader).
# patch-size 10 + q-out 10 (era 24 + "resto da shell") e batch-size 4 + lr
# 1e-3 (era 2 + 1e-4) -- ajustados pra bater com os hiperparametros do
# paper (ver utils/dataset.py e scripts/04_train_rcae.py, reproducao
# completa dos 8 itens revisados contra a implementacao oficial). Patch
# menor (10^3 vs 24^3) reduz bastante o uso de memoria por patch, entao a
# folga de --mem=64G/--max-cached-subjects abaixo continua valendo.
# ANGULAR_LOSS_WEIGHT / SH_LOSS_HIGH_ORDER_MIN (variaveis de ambiente, ver
# comentario de uso no topo deste arquivo): defaults reproduzem o
# comportamento de antes quando nao informadas (termo desativado).
ANGULAR_LOSS_WEIGHT="${ANGULAR_LOSS_WEIGHT:-0.0}"
SH_LOSS_HIGH_ORDER_MIN="${SH_LOSS_HIGH_ORDER_MIN:-4}"
echo "angular-loss-weight=$ANGULAR_LOSS_WEIGHT (0.0 = desativado) | sh-loss-high-order-min=$SH_LOSS_HIGH_ORDER_MIN"

# RESUME_CHECKPOINT/NO_RESUME (variaveis de ambiente, ver comentario de uso
# no topo deste arquivo) -- por padrao (nenhuma das duas setada) o script
# python resume sozinho do last.pt canonico, se existir.
RESUME_FLAG=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_FLAG=(--resume-checkpoint "$RESUME_CHECKPOINT")
    echo "RESUME_CHECKPOINT=$RESUME_CHECKPOINT -- retomando explicitamente deste checkpoint"
elif [[ "${NO_RESUME:-0}" == "1" ]]; then
    RESUME_FLAG=(--no-resume)
    echo "NO_RESUME=1 -- ignorando qualquer last.pt existente, comecando do zero"
fi

DECODER_TYPE="${DECODER_TYPE:-direct}"
SH_DECODER_LMAX="${SH_DECODER_LMAX:-4}"
DECODER_TYPE_FLAG=()
if [[ "$DECODER_TYPE" != "direct" ]]; then
    DECODER_TYPE_FLAG=(--decoder-type "$DECODER_TYPE" --sh-decoder-lmax "$SH_DECODER_LMAX")
    echo "DECODER_TYPE=$DECODER_TYPE SH_DECODER_LMAX=$SH_DECODER_LMAX -- treinando a variante com Decoder3DSH (checkpoint em .../_shdec/, exige treino do zero)"
fi

python scripts/04_train_rcae.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --scheme-dir "$WORK_DIR/subsampling" \
    --out-dir "$WORK_DIR/rcae_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" \
    --epochs 150 --batch-size 4 --patch-size 10 --q-out 10 \
    --lr 1e-3 --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --debug-plot-every 1 --debug-plot-every-batches 200 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    --angular-loss-weight "$ANGULAR_LOSS_WEIGHT" \
    --sh-loss-high-order-min "$SH_LOSS_HIGH_ORDER_MIN" \
    "${RESUME_FLAG[@]}" "${DECODER_TYPE_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"