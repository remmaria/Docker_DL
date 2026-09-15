#!/bin/bash
#SBATCH --job-name=rrin_star
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
# SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/train_rrin_star.%A_%a.err
#SBATCH --output=logs/train_rrin_star.%A_%a.out
#
# Treino da RRIN3DStar ("ensemble em estrela", etapa 4e, ver
# scripts/04e_train_rrin_star.py e protocolo secao 14.5 item 1/addendum
# 2026-08-27) para um (shell_b, n_level) especifico -- requer que
# scripts/02b_build_rrin_triplets.py ja tenha rodado COM --ensemble-m>=M
# pra esse work_dir (ver ENSEMBLE_M abaixo). Mesmo padrao de
# slurm/04b_train_rrin.sh (array de configs/experiments.tsv OU
# shell_b/n_level explicitos).
#
# Uso:
#   sbatch --array=1-N slurm/04e_train_rrin_star.sh <work_dir>
#   sbatch slurm/04e_train_rrin_star.sh <work_dir> <shell_b> <n_level>
#
# ENSEMBLE_M=<M> (variavel de ambiente, default 3) -- quantos pares diversos
# por alvo o ensemble usa (ver utils/gradients.py:find_star_ensemble_batch e
# model/rrin3d_star.py). PRECISA bater (ou ser <=) o M usado ao rodar
# scripts/02b_build_rrin_triplets.py --ensemble-m nesse work_dir -- se o
# npz nao tiver os campos '__ens_*', o treino falha cedo com uma mensagem
# clara (ver utils/rrin_dataset.py:RRINTripletDataset.ensemble_m) em vez de
# um erro confuso no meio do loop de treino.
#   ENSEMBLE_M=5 sbatch slurm/04e_train_rrin_star.sh <work_dir> <shell_b> <n_level>
# Grava em out_dir/shell<B>_n<N>_star<M>/ (sufixo automatico, nao colide com
# nenhuma variante de scripts/04b_train_rrin.py nem entre M diferentes).
#
# Resume automatico (ver scripts/04e_train_rrin_star.py) -- mesmo mecanismo
# do RRIN3D/RCAE: RESUME_CHECKPOINT=<caminho> ou NO_RESUME=1.
#
# LR=<valor> (default 1e-3, mesma logica de sufixo automatico _lr<valor> de
# slurm/04b_train_rrin.sh -- so aplica sufixo se != 1e-3).
#
# USE_QUALITY_COND=1 / NORM_TYPE=batch / ONLY_VALID=0 -- mesmo espirito e
# mesmos sufixos automaticos (_qc/_bn/_inclinv) de slurm/04b_train_rrin.sh,
# ver docstring de scripts/04e_train_rrin_star.py para o detalhe de cada um
# aplicado ao ensemble em estrela.
#
# NAO tem ANGULAR_LOSS_WEIGHT/SH_LOSS_* aqui -- scripts/04e_train_rrin_star.py
# ainda nao porta a loss angular/SH pro ensemble em estrela (TODO, ver
# docstring do script).
#
# RESIDUAL_L2_WEIGHT=<valor> (ADITIVO, default 0.0 = desligado, addendum
# 2026-09-13 -- hipotese sobre por que o RCAE bate a familia de fluxo em
# nmse/FA: RCAE tem um prior de suavidade forte, ver addendum 2026-08-27
# secoes 20.16/20.17) -- penaliza a MAGNITUDE do residuo do RefineNet3D
# (media so sobre as posicoes REAIS do feixe), empurrando a predicao a
# ficar perto do blend/warp bruto a menos que haja evidencia forte pra se
# afastar dele. Mesmo espirito de --zero-init-refine-output (ja existente
# no PairFlowStar), so que como penalidade continua durante o treino, nao
# so na inicializacao. Ganha sufixo _resl2<valor> no run_tag.
#   RESIDUAL_L2_WEIGHT=0.01 sbatch slurm/04e_train_rrin_star.sh <work_dir> <shell_b> <n_level>
#
# CROSS_CANDIDATE_ATTENTION=1 (ADITIVO, default 0 = desligado, addendum
# 2026-09-13/2026-09-14 -- item 1: mesma ideia do item 2 do `implicit`, so
# que entre os M candidatos do ensemble em estrela em vez de entre as
# n_level direcoes) -- insere self-attention entre os M candidatos do feixe
# (por voxel) ANTES do logit de confianca da PairWeightHead3D, ver
# model/rrin3d_star.py:CrossCandidateAttention3D. Ganha sufixo _cattn no
# run_tag. CROSS_CANDIDATE_ATTN_HEADS=<N> (default 4, so tem efeito com
# CROSS_CANDIDATE_ATTENTION=1) -- precisa dividir --base-ch (16) sem resto.
#   CROSS_CANDIDATE_ATTENTION=1 sbatch slurm/04e_train_rrin_star.sh <work_dir> <shell_b> <n_level>
#
# CAPACIDADE (addendum 2026-09-14) -- ate essa data este wrapper nao expunha
# nenhuma forma de mudar a largura da rede, e o treino rodava com ~185k
# parametros (base_ch=16) contra ~6,8M do RCAE, o modelo que vinha ganhando
# a comparacao de val_loss. Quatro variaveis novas, todas ADITIVAS:
#   BASE_CH=<N>          (default 16) largura geral -> sufixo _bc<N>
#   REFINE_BASE_CH=<N>   (default: segue BASE_CH) largura SO' da RefineNet3D,
#                        que e' a unica parte do modelo nao presa a premissa
#                        de fluxo optico (secao 14.6 do protocolo) e tinha
#                        so' 8.737 parametros -> sufixo _rbc<N>
#   REFINE_DEPTH=<N>     (default 2) camadas ocultas da RefineNet3D -> _rd<N>
#   REFINE_COND=1        (default 0) injeta bvec_a/bvec_b/bvec_t/t_frac/
#                        quality na RefineNet3D, que hoje nao sabe nem QUAL
#                        direcao esta predizendo -> sufixo _rcond
# Todas mudam shape de peso => NAO retomaveis a partir de um checkpoint com
# outra config (o script falha cedo com mensagem clara, e o sufixo proprio no
# run_tag ja evita colisao de checkpoint).
#   BASE_CH=32 REFINE_BASE_CH=64 REFINE_DEPTH=4 REFINE_COND=1 \
#     sbatch slurm/04e_train_rrin_star.sh <work_dir> <shell_b> <n_level>
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04e_train_rrin_star.sh <work_dir> [shell_b n_level]}"
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
    exit 1
fi
ENSEMBLE_M="${ENSEMBLE_M:-3}"
echo "Treinando RRIN3DStar (ensemble em estrela, M=$ENSEMBLE_M) para shell_b=$SHELL_B, n_level=$N_LEVEL"
source "./00_env_common.sh"
RESUME_FLAG=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_FLAG=(--resume-checkpoint "$RESUME_CHECKPOINT")
    echo "RESUME_CHECKPOINT=$RESUME_CHECKPOINT -- retomando explicitamente deste checkpoint"
elif [[ "${NO_RESUME:-0}" == "1" ]]; then
    RESUME_FLAG=(--no-resume)
    echo "NO_RESUME=1 -- ignorando qualquer last.pt existente, comecando do zero"
fi
LR="${LR:-1e-3}"
echo "LR=$LR (default 1e-3)"
QC_FLAG=()
if [[ "${USE_QUALITY_COND:-0}" == "1" ]]; then
    QC_FLAG=(--use-quality-cond)
    echo "USE_QUALITY_COND=1 -- treinando a variante consciente da qualidade de cada par do feixe (condiciona o FlowNet3D)"
fi
WQC_FLAG=()
if [[ "${WEIGHT_QUALITY_COND:-0}" == "1" ]]; then
    WQC_FLAG=(--weight-quality-cond)
    echo "WEIGHT_QUALITY_COND=1 -- alimentando residual_deg/gap_deg direto na PairWeightHead3D (condiciona a fusao, independente de USE_QUALITY_COND)"
fi
ONLY_VALID_FLAG=()
if [[ "${ONLY_VALID:-1}" == "0" ]]; then
    ONLY_VALID_FLAG=(--no-only-valid)
    echo "ONLY_VALID=0 -- treinando/validando tambem com alvos cujo par-unico e invalido"
fi
NORM_TYPE="${NORM_TYPE:-instance}"
NORM_TYPE_FLAG=()
if [[ "$NORM_TYPE" != "instance" ]]; then
    NORM_TYPE_FLAG=(--norm-type "$NORM_TYPE")
    echo "NORM_TYPE=$NORM_TYPE -- treinando a variante com BatchNorm3d (exige treino do zero)"
fi
TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
if [[ "$TRIPLETS_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "TRIPLETS_DIR=$TRIPLETS_DIR -- lendo trincas de pasta SEPARADA da producao (subsampling/)"
fi

RESIDUAL_L2_WEIGHT="${RESIDUAL_L2_WEIGHT:-0.0}"
if [[ "$RESIDUAL_L2_WEIGHT" != "0.0" && "$RESIDUAL_L2_WEIGHT" != "0" ]]; then
    echo "RESIDUAL_L2_WEIGHT=$RESIDUAL_L2_WEIGHT -- penalizando magnitude do residuo do RefineNet3D (checkpoint em .../_resl2<valor>/)"
fi

# --- capacidade (addendum 2026-09-14) --------------------------------------
# Ate 2026-09-14 este wrapper NAO expunha --base-ch: a unica forma de mudar a
# largura da rede era editar o .py. Como a analise de capacidade mostrou que
# esta linha roda com ~185k parametros contra ~6,8M do RCAE (37x menos), a
# variavel passou a ser exposta aqui.
BASE_CH="${BASE_CH:-16}"
BASE_CH_FLAG=()
if [[ "$BASE_CH" != "16" ]]; then
    BASE_CH_FLAG=(--base-ch "$BASE_CH")
    echo "BASE_CH=$BASE_CH (default 16) -- run_tag ganha sufixo _bc$BASE_CH; NAO retomavel a partir de um checkpoint com outra largura"
fi
REFINE_BASE_CH="${REFINE_BASE_CH:-}"
REFINE_BASE_CH_FLAG=()
if [[ -n "$REFINE_BASE_CH" ]]; then
    REFINE_BASE_CH_FLAG=(--refine-base-ch "$REFINE_BASE_CH")
    echo "REFINE_BASE_CH=$REFINE_BASE_CH -- largura da RefineNet3D desacoplada do resto (run_tag ganha _rbc$REFINE_BASE_CH)"
fi
REFINE_DEPTH="${REFINE_DEPTH:-2}"
REFINE_DEPTH_FLAG=()
if [[ "$REFINE_DEPTH" != "2" ]]; then
    REFINE_DEPTH_FLAG=(--refine-depth "$REFINE_DEPTH")
    echo "REFINE_DEPTH=$REFINE_DEPTH (default 2) -- camadas ocultas da RefineNet3D (run_tag ganha _rd$REFINE_DEPTH)"
fi
REFINE_COND_FLAG=()
if [[ "${REFINE_COND:-0}" == "1" ]]; then
    REFINE_COND_FLAG=(--refine-cond)
    echo "REFINE_COND=1 -- injetando bvec_a/bvec_b/bvec_t/t_frac/quality na RefineNet3D (run_tag ganha _rcond)"
fi

CATTN_FLAG=()
if [[ "${CROSS_CANDIDATE_ATTENTION:-0}" == "1" ]]; then
    CATTN_FLAG=(--cross-candidate-attention)
    echo "CROSS_CANDIDATE_ATTENTION=1 -- treino NOVO com self-attention entre os M candidatos do feixe, ANTES do logit de confianca (run_tag ganha sufixo _cattn)"
fi
CROSS_CANDIDATE_ATTN_HEADS="${CROSS_CANDIDATE_ATTN_HEADS:-4}"
CATTN_HEADS_FLAG=()
if [[ "$CROSS_CANDIDATE_ATTN_HEADS" != "4" ]]; then
    CATTN_HEADS_FLAG=(--cross-candidate-attn-heads "$CROSS_CANDIDATE_ATTN_HEADS")
    echo "CROSS_CANDIDATE_ATTN_HEADS=$CROSS_CANDIDATE_ATTN_HEADS (default 4, so tem efeito com CROSS_CANDIDATE_ATTENTION=1) -- precisa dividir --base-ch sem resto"
fi

RESET_LR_FLAG=()
if [[ "${RESET_LR:-0}" == "1" ]]; then
    RESET_LR_FLAG=(--reset-lr)
    echo "RESET_LR=1 -- reinicio A QUENTE: carrega os pesos do checkpoint mas recria otimizador e scheduler em LR=$LR (sem esta flag, o resume restaura optimizer_state/scheduler_state e o LR da linha de comando e' ignorado na pratica). Mantem best_val, zera epochs_no_improve."
    if [[ -z "${RESUME_CHECKPOINT:-}" && "${NO_RESUME:-0}" == "1" ]]; then
        echo "  ATENCAO: RESET_LR=1 com NO_RESUME=1 nao faz reinicio a quente nenhum -- vai treinar do zero. Passe RESUME_CHECKPOINT=<best.pt> e tire o NO_RESUME."
    fi
fi

python scripts/04e_train_rrin_star.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --triplets-dir "$TRIPLETS_DIR" \
    --out-dir "$WORK_DIR/rrin_star_checkpoints" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" --ensemble-m "$ENSEMBLE_M" \
    --epochs 150 --batch-size 8 --patch-size 10 \
    --lr "$LR" --num-workers 8 --max-cached-subjects 6 --patience 15 \
    --val-num-workers 4 --val-max-cached-subjects 1 \
    --residual-l2-weight "$RESIDUAL_L2_WEIGHT" \
    "${RESUME_FLAG[@]}" "${RESET_LR_FLAG[@]}" "${QC_FLAG[@]}" "${WQC_FLAG[@]}" "${ONLY_VALID_FLAG[@]}" "${NORM_TYPE_FLAG[@]}" \
    "${CATTN_FLAG[@]}" "${CATTN_HEADS_FLAG[@]}" \
    "${BASE_CH_FLAG[@]}" "${REFINE_BASE_CH_FLAG[@]}" "${REFINE_DEPTH_FLAG[@]}" "${REFINE_COND_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"