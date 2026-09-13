#!/bin/bash
#SBATCH --job-name=dmri_scheme
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-05:00:00
#SBATCH --account=tibrahim
#SBATCH --error=logs/scheme.%A_%a.err
#SBATCH --output=logs/scheme.%A_%a.out
#
# So a geracao do esquema de subamostragem (etapa 2) -- gera um .npz leve
# (indices, poucos KB) por sujeito, cobrindo TODOS os niveis de uma vez.
# Isso e barato e roda uma vez so; nao precisa ser por combo.
#
# A reconstrucao do baseline SH (o que gera os recon_target.nii.gz pesados)
# NAO esta mais aqui -- ver 02b_baseline_reconstruct.sh, que roda por
# combo (array), pra nao acumular todos os 30 combos em disco de uma vez.
#
# Uso (canonico, sem overrides, 1 job so cobrindo o manifesto inteiro):
#   sbatch 02_baseline_sh.sh <work_dir>
#
# Uso (esquema ALTERNATIVO -- ex.: mais niveis, so pra uma investigacao
# pontual tipo a curva erro-vs-N-direcoes -- sem sobrescrever o esquema
# canonico usado por RCAE/RRIN):
#   sbatch 02_baseline_sh.sh <work_dir> <out_dir_ou_-> [level1 level2 ...]
# O 2o argumento e o diretorio de saida alternativo (caminho ABSOLUTO); passe
# "-" nesse lugar se quiser manter o out-dir canonico mas so mudar os
# niveis. Se nenhum nivel for passado depois dele, usa o default (canonico).
# Ex. (curva mais fina, pasta separada, so shell 1000):
#   sbatch 02_baseline_sh.sh <work_dir> $WORK_DIR/subsampling_basecurve \
#       6 10 12 16 20 24 28 32 36 40 44 48 52 56 60
#
# Uso (paralelizando por SUJEITO via --array -- cada task processa so uma
# fatia do manifesto; nao precisa de merge depois, cada sujeito grava seu
# proprio .npz): combina com qualquer uma das formas acima, so acrescente
# --array. Recomendado com METHOD=electrostatic (bem mais lento que fps por
# sujeito, ver abaixo) -- com METHOD=fps (default) o job inteiro geralmente
# ja e rapido o bastante sem precisar de array.
#   sbatch --array=1-20 02_baseline_sh.sh <work_dir>
#   METHOD=electrostatic sbatch --array=1-20 02_baseline_sh.sh <work_dir>
#
# ARGUMENTOS POSICIONAIS EM VEZ DE VARIAVEIS DE AMBIENTE: em alguns clusters
# SLURM, "VAR=valor sbatch ..." nao propaga a variavel pro ambiente do job de
# verdade (depende de configuracao do site) -- o script cairia no default
# canonico SILENCIOSAMENTE nesse caso. Argumento posicional nunca tem esse
# problema. OUT_DIR_OVERRIDE/LEVELS_OVERRIDE (variaveis de ambiente) ainda
# sao aceitas como fallback, mas os argumentos posicionais acima, se
# passados, tem prioridade. Confirme no log a linha "METHOD=..."/"out_dir
# (arg 2)=..." pra ter certeza de que a variavel realmente chegou no job
# (ver secao 12.1 do protocolo sobre esse problema neste cluster
# especifico) -- se a linha nao aparecer, a variavel nao propagou.
#
# METHOD=fps|electrostatic (variavel de ambiente, default "fps"): metodo de
# selecao de direcoes -- ver docstring de scripts/02_subsample_directions.py
# e utils/gradients.py:electrostatic_repulsion_sampling. Recomendado usar
# METHOD=electrostatic a partir de 2026-09 (mesmo metodo do pre-processamento
# clinico via --sub_qspace, evita domain gap geometrico entre treino e
# producao -- ver notas do projeto). N_STARTS/MAX_LOCAL_ITER/SAMPLING_SEED
# (variaveis de ambiente, defaults 20/150/42) so tem efeito com
# METHOD=electrostatic -- 20 sementes calibrado em 2026-09 via
# scripts/02e_calibrate_electrostatic_nstarts.py pra shells 500-2000 x N
# 6-54 (ja 10 sementes ficava a <1% da energia de 160 sementes em todas as
# combinacoes testadas; 20 mantem margem 2x). electrostatic e BEM mais
# lento que fps por sujeito (o --time acima pode precisar subir se voce
# mudar pra electrostatic com muitos
# sujeitos/niveis -- confira o tempo de 1-2 sujeitos antes de rodar o
# manifesto inteiro).

set -euo pipefail
mkdir -p logs
WORK_DIR="${1:?uso: sbatch 02_baseline_sh.sh <work_dir> [out_dir_ou_-] [levels...]}"

POS_OUT_DIR="${2:-}"
if [[ $# -ge 2 ]]; then
    shift 2
else
    shift 1
fi
POS_LEVELS=("$@")

source "./00_env_common.sh"

if [[ -n "$POS_OUT_DIR" && "$POS_OUT_DIR" != "-" ]]; then
    OUT_DIR="$POS_OUT_DIR"
    echo "out_dir (arg 2) = $OUT_DIR -- gravando esquema nesse caminho (nao mexe no canonico)"
elif [[ -n "${OUT_DIR_OVERRIDE:-}" ]]; then
    OUT_DIR="$OUT_DIR_OVERRIDE"
    echo "OUT_DIR_OVERRIDE=$OUT_DIR_OVERRIDE -- gravando esquema nesse caminho (nao mexe no canonico)"
else
    OUT_DIR="$WORK_DIR/subsampling"
fi

if [[ ${#POS_LEVELS[@]} -gt 0 ]]; then
    LEVELS=("${POS_LEVELS[@]}")
    echo "niveis (argumentos posicionais) = ${LEVELS[*]}"
elif [[ -n "${LEVELS_OVERRIDE:-}" ]]; then
    read -ra LEVELS <<< "$LEVELS_OVERRIDE"
    echo "LEVELS_OVERRIDE='$LEVELS_OVERRIDE' -- usando essa lista em vez da canonica"
else
    LEVELS=(6 10 16 20 24 32 48 54)
fi

METHOD="${METHOD:-fps}"
N_STARTS="${N_STARTS:-20}"
MAX_LOCAL_ITER="${MAX_LOCAL_ITER:-150}"
SAMPLING_SEED="${SAMPLING_SEED:-42}"
echo "METHOD=$METHOD (n_starts=$N_STARTS, max_local_iter=$MAX_LOCAL_ITER, sampling_seed=$SAMPLING_SEED -- so usados se METHOD=electrostatic)"

# Sharding por sujeito via --array (opcional -- ver comentario de uso acima).
# Mesma logica de SHARD_INDEX/SHARD_COUNT ja usada em
# slurm/02b_baseline_reconstruct.sh: SHARD_INDEX sempre base 1 do
# SLURM_ARRAY_TASK_ID (nao do minimo desta submissao), pra sobreviver a
# resubmissoes parciais de tasks que falharam (--array=3,7,12 preserva os
# IDs originais). Numa resubmissao parcial, passe SHARD_COUNT_OVERRIDE=<total
# original> explicitamente, senao o fallback MAX desta submissao fica errado.
SHARD_INDEX=0
SHARD_COUNT=1
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    SHARD_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
    if [[ -n "${SHARD_COUNT_OVERRIDE:-}" ]]; then
        SHARD_COUNT="$SHARD_COUNT_OVERRIDE"
    elif [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" && "${SLURM_ARRAY_TASK_MIN:-1}" == "1" ]]; then
        SHARD_COUNT="$SLURM_ARRAY_TASK_COUNT"
    else
        SHARD_COUNT=$((${SLURM_ARRAY_TASK_MAX:-1}))
    fi
    echo "[shard] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID MIN=${SLURM_ARRAY_TASK_MIN:-?} MAX=${SLURM_ARRAY_TASK_MAX:-?} -> SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT (numa resubmissao parcial, confira que SHARD_COUNT bate com o total original -- use SHARD_COUNT_OVERRIDE=<N> se nao bater)"
fi

python scripts/02_subsample_directions.py \
    --manifest "$WORK_DIR/manifest.csv" \
    --out-dir "$OUT_DIR" \
    --levels "${LEVELS[@]}" \
    --method "$METHOD" \
    --n-starts "$N_STARTS" \
    --max-local-iter "$MAX_LOCAL_ITER" \
    --sampling-seed "$SAMPLING_SEED" \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT"
# uniao de todos os niveis usados em qualquer experimento (ver configs/experiments.tsv);
# niveis maiores que as direcoes disponiveis numa shell/sujeito sao pulados
# automaticamente (aviso no log), sem problema.