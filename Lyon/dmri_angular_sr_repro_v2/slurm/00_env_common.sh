#!/bin/bash
# Trecho comum de setup de ambiente, no mesmo padrao do seu script de treino
# original -- incluido (via `source`) em todos os jobs SLURM deste diretorio,
# para nao repetir o boilerplate em cada arquivo.
#
# Ajuste aqui uma unica vez se o modulo do anaconda ou o nome do env mudar.

module purge
module load anaconda3/2025.7.0-2-python_3.11
eval "$(conda shell.bash hook)"
conda activate dmri_dl

export PYTHONNOUSERSITE=1
# acompanha o que o SLURM realmente alocou (--cpus-per-task) em vez de um
# valor fixo -- com 8 fixo e --cpus-per-task=16 (script de producao), so
# metade dos cores alocados eram usados pelas operacoes de algebra linear/
# convolucao do PyTorch em CPU, desperdicando a outra metade.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# sem isso, o stdout redirecionado pro .out do SLURM fica bufferizado em
# blocos -- os prints (inclusive os de progresso por epoca/batch) so
# aparecem no arquivo quando o buffer enche ou o processo termina, dando a
# falsa impressao de que o job travou enquanto na verdade esta rodando.
export PYTHONUNBUFFERED=1

# IMPORTANTE: o SLURM copia o .sh submetido para /var/spool/slurmd/... e
# executa a copia de la -- por isso NAO da pra usar BASH_SOURCE pra achar
# os arquivos vizinhos (ele apontaria pro spool, nao pro repo). Em vez
# disso contamos com o cwd do job, que o SLURM inicializa como o diretorio
# de onde voce rodou `sbatch` ($SLURM_SUBMIT_DIR) -- por isso todo script
# desta pasta pede pra voce rodar `cd slurm` antes de `sbatch ...`.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
fi
if [[ ! -f "./00_env_common.sh" ]]; then
    echo "Erro: nao estou na pasta slurm/ (cwd=$(pwd))."
    echo "Rode 'cd slurm' e submeta o job de la (ex.: cd slurm && sbatch 01_prepare_data.sh ...)."
    exit 1
fi
REPO_ROOT="$(cd .. && pwd)"
cd "$REPO_ROOT" || exit 1
echo "REPO_ROOT=$REPO_ROOT"
echo "python: $(which python)"
python -c "import torch; print('torch', torch.__version__, '| CUDA disponivel:', torch.cuda.is_available())"