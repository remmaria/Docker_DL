#!/bin/bash
#SBATCH --job-name=pairflow_star_iotest
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/pairflow_star_iotest.%A_%a.err
#SBATCH --output=logs/pairflow_star_iotest.%A_%a.out
#
# Mini-teste do gargalo de I/O do dataloader (pedido explicito da usuaria em
# 2026-09-03: "dá pra fazer um mini test pra ver se realmente conseguimos
# assim aumentar a % da GPU?" -- ver addendum secao 30 (gargalo medido:
# wait ~93-99% do tempo de epoca, so' 1-7% em compute) e secao 33.1
# (--freeze-subject-order trazido a scripts/04i_train_pairflow_star.py
# especificamente pra este teste).
#
# O QUE ESTE SCRIPT FAZ (dedicado a este teste -- NAO usa nem modifica
# slurm/04i_train_pairflow_star.sh, o wrapper de PRODUCAO, que fica
# intocado):
#   1. Constroi um manifesto MINI (scripts/99_make_mini_manifest.py, com
#      --filter-triplets pra so' pegar sujeitos que ja tem trincas prontas
#      pro (shell_b, n_level, ensemble_m) pedido) em
#      $WORK_DIR/manifest_mini_iotest.csv -- reconstruido do zero a cada
#      chamada (idempotente).
#   2. Roda scripts/04i_train_pairflow_star.py sobre esse manifesto mini,
#      por poucas epocas completas (EPOCHS, default 6 -- o efeito de cache
#      de pagina so' aparece DE EPOCA EM EPOCA, por isso NAO usamos
#      --max-train-batches aqui), escrevendo em
#      $WORK_DIR/pairflow_star_checkpoints_iotest{_frozen}/ (pasta SEPARADA
#      da producao -- nunca toca em checkpoints reais).
#
# COMO COMPARAR (rode as duas chamadas abaixo, mesmo work_dir/shell_b/
# n_level, e depois compare os dois logs):
#   sbatch slurm/04i_minitest_io.sh <work_dir> <shell_b> <n_level>
#   FREEZE_SUBJECT_ORDER=1 sbatch slurm/04i_minitest_io.sh <work_dir> <shell_b> <n_level>
#
#   grep -E "epoca [0-9]+ resumo" logs/pairflow_star_iotest.<JOBID_sem_freeze>.out
#   grep -E "epoca [0-9]+ resumo" logs/pairflow_star_iotest.<JOBID_com_freeze>.out
#
#   Compare o "(X%)" de wait entre as duas rodadas, e entre a epoca 1 e as
#   epocas finais (4-6) DA MESMA rodada com FREEZE_SUBJECT_ORDER=1 -- se o
#   cache de pagina do sistema de arquivos de rede esta de fato esquentando,
#   o wait% deveria CAIR ao longo das epocas so' nessa rodada.
#
# ENSEMBLE_M=<M> (default 3) -- mesma semantica de slurm/04i_train_
# pairflow_star.sh, PRECISA bater (ou ser <=) o M usado ao rodar
# scripts/02b_build_rrin_triplets.py --ensemble-m nesse work_dir.
#
# N_TRAIN/N_VAL (default 16/4) -- sujeitos no manifesto mini, por split.
# Recomendado manter N_TRAIN >= 2x NUM_WORKERS (default 8) pra garantir
# mais de 1 sujeito por grupo-de-worker (senao o particionamento vira
# trivial e a releitura entre epocas nao chega a ser testada de verdade).
#
# MAX_CACHED_SUBJECTS (default 6, MESMO default de scripts/04i_train_
# pairflow_star.py) -- capacidade do cache LRU em processo, POR WORKER
# (ver utils/rrin_dataset.py:RRINTripletDataset, param max_cached_subjects).
# ATENCAO (achado real rodando N_TRAIN=64/NUM_WORKERS=8 em 2026-09-03,
# 8 sujeitos/worker > MAX_CACHED_SUBJECTS=6 default): quando os sujeitos
# por worker (N_TRAIN/NUM_WORKERS) excedem este valor, MESMO com
# --freeze-subject-order o cache em processo nao segura todos os sujeitos
# do worker de uma epoca pra outra -- uma fracao (aqui, (8-6)/8=25% dos
# sujeitos por worker) precisa ser relida do disco toda epoca, derrubando
# o patches/s (visto: ~580/s com N_TRAIN=16 caiu pra ~270-320/s com
# N_TRAIN=64, mesmo batch/LR/freeze). Suba MAX_CACHED_SUBJECTS junto com
# N_TRAIN (ex.: MAX_CACHED_SUBJECTS=8 pra bater com N_TRAIN=64/NUM_WORKERS=8)
# pra testar se a vazao se recupera -- info util tambem pra estimar quanta
# RAM por worker a producao precisaria pra sustentar isso em escala real
# (~75 sujeitos/worker).
#
# EPOCHS (default 6) / NUM_WORKERS (default 8, mesma contagem da producao
# de proposito -- muda a granularidade do particionamento) / BATCH_SIZE
# (default 8).
#
# FREEZE_SUBJECT_ORDER=1 (default 0) -- ativa --freeze-subject-order em
# scripts/04i_train_pairflow_star.py (ver docstring la' e addendum 33.1).
#
# LR=<valor> (default 1e-4, MESMO default da producao) -- pedido explicito
# da usuaria em 2026-09-03 pra testar BATCH_SIZE maior com a LR escalada
# proporcionalmente (regra de escala linear -- ver addendum secao 33.6):
# BATCH_SIZE=32 tem 1/4 dos passos de otimizacao por epoca de BATCH_SIZE=8,
# entao a LR "equivalente" e' ~4x a original. Ex.: LR=4e-4 pra testar
# BATCH_SIZE=32 contra o baseline BATCH_SIZE=8/LR=1e-4.
#
# TRIPLETS_DIR (default $WORK_DIR/subsampling) -- mesma semantica do
# wrapper de producao.
#
# O nome de OUT_DIR agora inclui BATCH_SIZE/LR quando diferem do default,
# pra runs de comparacao (ex.: BATCH_SIZE=8 vs BATCH_SIZE=32/LR=4e-4) nao
# se sobrescreverem no mesmo checkpoints_iotest/.
#
# INIT_CHECKPOINT=<caminho> / FREEZE_FLOW=1 (pedido explicito da usuaria em
# 2026-09-03: "o mini treino com o pairflowStar, rola fazer ele com o pre
# treino dos flows nao supervisionados pra ver se melhora tb?") -- MESMA
# semantica de slurm/04i_train_pairflow_star.sh (o wrapper de producao):
# INIT_CHECKPOINT aponta pro checkpoint da Etapa 1 (SSL nao-supervisionado,
# scripts/04g_train_pairflow_ssl.py) e carrega SO' os pesos de `flow_net`
# antes do mini-treino comecar; FREEZE_FLOW=1 (requer INIT_CHECKPOINT)
# congela esse flow_net, so' treinando refine_net/weight_head. O proprio
# scripts/04i_train_pairflow_star.py ja sufixa o run_tag com
# '_pretrained'/'_frozen' automaticamente quando essas flags sao usadas,
# entao NAO precisa (nem foi adicionado) sufixo manual extra em OUT_DIR
# pra essas duas -- ja nao colide com uma rodada de controle (sem
# INIT_CHECKPOINT) no mesmo OUT_DIR base.
#
# Uso (comparar controle do zero vs com pre-treino, mesmo manifesto mini):
#   sbatch slurm/04i_minitest_io.sh <work_dir> <shell_b> <n_level>
#   INIT_CHECKPOINT=$WORK_DIR/pairflow_ssl_checkpoints/shell1000/best.pt \
#     sbatch slurm/04i_minitest_io.sh <work_dir> <shell_b> <n_level>
#   INIT_CHECKPOINT=$WORK_DIR/pairflow_ssl_checkpoints/shell1000/best.pt FREEZE_FLOW=1 \
#     sbatch slurm/04i_minitest_io.sh <work_dir> <shell_b> <n_level>
#
# VAL_NUM_WORKERS (default 0) / VAL_MAX_CACHED_SUBJECTS (default 1) --
# ate' 2026-09-03 estes dois vinham HARDCODED (0 e 1) na chamada de
# scripts/04i_train_pairflow_star.py, entao o loader de VALIDACAO rodava
# 100% sincrono (sem overlap I/O/compute) INDEPENDENTE do que se ajustava
# em MAX_CACHED_SUBJECTS (que so' afeta o loader de TREINO) -- achado real
# (secao 33.11): wait% de val ficou preso em 97-98% mesmo com
# MAX_CACHED_SUBJECTS=16 ou 32 no treino, porque val_num_workers=0 nunca
# de'ixa o dataloader adiantar leitura enquanto a GPU calcula o batch
# anterior.
#
# CUIDADO -- assimetria real com o loader de treino: o val_loader (ver
# scripts/04i_train_pairflow_star.py) usa `shuffle=False` SEM
# SubjectGroupedSampler (esse sampler so' e' aplicado ao train_loader).
# Ou seja, ao contrario do treino, NAO ha particionamento garantido de
# sujeitos por worker na validacao -- com VAL_NUM_WORKERS>1 o PyTorch
# distribui os batches round-robin entre os workers, o que pode fazer
# mais de um worker precisar do MESMO sujeito na mesma epoca (releitura
# redundante entre workers, sem a garantia de exclusividade que o treino
# tem). Mesmo assim, sair de VAL_NUM_WORKERS=0 (sincrono, sem prefetch
# algum) pra >0 deve ajudar so' pelo overlap de I/O com compute -- vale
# testar empiricamente (nao ha garantia teorica do ganho exato, ao
# contrario do treino). Ex. de ponto de partida: VAL_NUM_WORKERS=2 (ou
# ate' NUM_WORKERS, se sobrar CPU) com VAL_MAX_CACHED_SUBJECTS>=N_VAL
# (pra cobrir o pior caso de todo sujeito de val acabar pedido por mais
# de um worker).
set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 04i_minitest_io.sh <work_dir> <shell_b> <n_level>}"
SHELL_B="${2:?uso: sbatch 04i_minitest_io.sh <work_dir> <shell_b> <n_level>}"
N_LEVEL="${3:?uso: sbatch 04i_minitest_io.sh <work_dir> <shell_b> <n_level>}"

ENSEMBLE_M="${ENSEMBLE_M:-3}"
N_TRAIN="${N_TRAIN:-16}"
N_VAL="${N_VAL:-4}"
EPOCHS="${EPOCHS:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
MAX_CACHED_SUBJECTS="${MAX_CACHED_SUBJECTS:-6}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
VAL_MAX_CACHED_SUBJECTS="${VAL_MAX_CACHED_SUBJECTS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"

echo "Mini-teste de I/O (PairFlowStar) para shell_b=$SHELL_B, n_level=$N_LEVEL, M=$ENSEMBLE_M"
echo "N_TRAIN=$N_TRAIN N_VAL=$N_VAL EPOCHS=$EPOCHS NUM_WORKERS=$NUM_WORKERS BATCH_SIZE=$BATCH_SIZE LR=$LR"
echo "VAL_NUM_WORKERS=$VAL_NUM_WORKERS VAL_MAX_CACHED_SUBJECTS=$VAL_MAX_CACHED_SUBJECTS"
if [[ "$VAL_NUM_WORKERS" == "0" ]]; then
    echo "VAL_NUM_WORKERS=0 -- validacao 100% sincrona (sem overlap I/O/compute); suba pra >0 pra testar ganho (ver comentario de cabecalho sobre a assimetria com o sampler de treino)"
fi
source "./00_env_common.sh"

TRIPLETS_DIR="${TRIPLETS_DIR:-$WORK_DIR/subsampling}"
if [[ "$TRIPLETS_DIR" != "$WORK_DIR/subsampling" ]]; then
    echo "TRIPLETS_DIR=$TRIPLETS_DIR -- lendo trincas de pasta SEPARADA da producao (subsampling/)"
fi

MINI_MANIFEST="$WORK_DIR/manifest_mini_iotest.csv"
echo "[1/2] construindo manifesto mini em $MINI_MANIFEST ..."
python scripts/99_make_mini_manifest.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --out "$MINI_MANIFEST" \
    --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test 0 \
    --filter-triplets --triplets-dir "$TRIPLETS_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" --ensemble-m "$ENSEMBLE_M"

OUT_DIR="$WORK_DIR/pairflow_star_checkpoints_iotest"
if [[ "$BATCH_SIZE" != "8" ]]; then
    OUT_DIR="${OUT_DIR}_bs${BATCH_SIZE}"
fi
if [[ "$LR" != "1e-4" ]]; then
    OUT_DIR="${OUT_DIR}_lr${LR}"
fi
FREEZE_FLAG=()
if [[ "${FREEZE_SUBJECT_ORDER:-0}" == "1" ]]; then
    FREEZE_FLAG=(--freeze-subject-order)
    OUT_DIR="${OUT_DIR}_frozen"
    echo "FREEZE_SUBJECT_ORDER=1 -- ativando --freeze-subject-order (checkpoints de debug em $OUT_DIR)"
else
    echo "FREEZE_SUBJECT_ORDER nao ativado -- rodada BASELINE (ordem reembaralhada a cada epoca, checkpoints de debug em $OUT_DIR)"
fi

INIT_FLAG=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
    INIT_FLAG=(--init-checkpoint "$INIT_CHECKPOINT")
    echo "INIT_CHECKPOINT=$INIT_CHECKPOINT -- inicializando flow_net do pre-treino nao-supervisionado da Etapa 1 (scripts/04g_train_pairflow_ssl.py)"
else
    echo "INIT_CHECKPOINT nao passado -- mini-treino do ZERO (controle, sem sufixo '_pretrained' no run_tag)"
fi

FREEZE_FLOW_FLAG=()
if [[ "${FREEZE_FLOW:-0}" == "1" ]]; then
    if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
        echo "Erro: FREEZE_FLOW=1 requer INIT_CHECKPOINT (nao faz sentido congelar fluxo do zero)"
        exit 1
    fi
    FREEZE_FLOW_FLAG=(--freeze-flow)
    echo "FREEZE_FLOW=1 -- congelando flow_net pre-treinado, so treinando refine_net/weight_head (sufixo '_frozen' automatico no run_tag)"
fi

if [[ "$MAX_CACHED_SUBJECTS" != "6" ]]; then
    echo "MAX_CACHED_SUBJECTS=$MAX_CACHED_SUBJECTS (default seria 6) -- teste de capacidade do cache em processo por worker (compare com N_TRAIN/NUM_WORKERS = sujeitos-por-worker)"
fi

if [[ "$WARMUP_STEPS" != "0" ]]; then
    echo "WARMUP_STEPS=$WARMUP_STEPS -- LR sobe linearmente de 0.1*LR ate LR ao longo dos primeiros $WARMUP_STEPS passos de otimizador (ver addendum secao 33.19)"
fi

ZERO_INIT_FLAG=()
if [[ "${ZERO_INIT_REFINE_OUTPUT:-0}" == "1" ]]; then
    ZERO_INIT_FLAG=(--zero-init-refine-output)
    echo "ZERO_INIT_REFINE_OUTPUT=1 -- zerando ultima camada de RefineNet3D (predicao inicial = blend exato, ver addendum secao 33.19)"
fi

echo "[2/2] rodando $EPOCHS epoca(s) completas no manifesto mini (batch_size=$BATCH_SIZE, lr=$LR, max_cached_subjects=$MAX_CACHED_SUBJECTS) ..."
python scripts/04i_train_pairflow_star.py \
    --manifest "$MINI_MANIFEST" \
    --triplets-dir "$TRIPLETS_DIR" \
    --out-dir "$OUT_DIR" \
    --shell-b "$SHELL_B" --n-level "$N_LEVEL" --ensemble-m "$ENSEMBLE_M" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --patch-size 10 \
    --lr "$LR" --num-workers "$NUM_WORKERS" --max-cached-subjects "$MAX_CACHED_SUBJECTS" --patience 999 \
    --val-num-workers "$VAL_NUM_WORKERS" --val-max-cached-subjects "$VAL_MAX_CACHED_SUBJECTS" \
    --warmup-steps "$WARMUP_STEPS" \
    --no-resume \
    "${FREEZE_FLAG[@]}" "${INIT_FLAG[@]}" "${FREEZE_FLOW_FLAG[@]}" "${ZERO_INIT_FLAG[@]}" \
    --job-id "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"

echo "[ok] mini-teste concluido -- confira as linhas 'epoca N resumo' acima (ou no .out deste job) para comparar wait%."