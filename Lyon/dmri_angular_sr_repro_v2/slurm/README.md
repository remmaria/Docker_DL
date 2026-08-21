# Jobs SLURM

Baseados no seu script de treino original (mesmo cluster `gpu`, partition
`l40s`, conta `tibrahim`, módulo `anaconda3/2025.7.0-2-python_3.11`, env
conda `dmri_dl`). `00_env_common.sh` concentra esse boilerplate — os outros
scripts só fazem `source` nele, então se algo mudar (versão do módulo, nome
do env), ajuste em um lugar só.

Seu ambiente `dmri_dl` provavelmente já tem PyTorch; confira se também tem
`nibabel`, `dipy`, `pandas`, `scikit-image` (ver `requirements.txt` na raiz
do repo) — se não tiver, `conda activate dmri_dl && pip install nibabel dipy
pandas scikit-image scikit-learn` antes de submeter.

## Ordem

```bash
cd slurm

# 1. manifesto + relatório de shells disponíveis (rápido, roda uma vez)
sbatch 01_prepare_data.sh /ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias /caminho/work_dir
# -> olhe work_dir/shell_availability_summary.csv e decida quais
#    (shell_b, n_level) valem a pena; edite configs/experiments.tsv

# 2. so gera os esquemas de subamostragem (.npz leves, cobre todos os niveis
#    de uma vez -- barato, roda uma vez so)
sbatch 02_baseline_sh.sh /caminho/work_dir

N=$(grep -vc '^#' configs/experiments.tsv)

# 2b. reconstrucao do baseline SH -- por combo (array), nao gera os 30 de
#     uma vez. Pode rodar em paralelo com o passo 3 (treino), sao independentes
sbatch --array=1-$N 02b_baseline_reconstruct.sh /caminho/work_dir

# 3. treino do RCAE -- um job por linha de configs/experiments.tsv
sbatch --array=1-$N 03_train_rcae.sh /caminho/work_dir

# 4. reconstrução com o modelo treinado (depois que os checkpoints existirem)
sbatch --array=1-$N 04_reconstruct_rcae.sh /caminho/work_dir

# 5. métricas de sinal + downstream DTI (e NODDI, com RUN_NODDI=1).
#    CLEANUP_AFTER=1 apaga os recon_target.nii.gz (a parte pesada) assim
#    que as metricas desse combo estao salvas -- ver "Espaço em disco" abaixo
CLEANUP_AFTER=1 sbatch --array=1-$N 05_evaluate_and_downstream.sh /caminho/work_dir

# 6. (opcional) tratografia via MRtrix3 -- ajuste o module load dentro do arquivo
# IMPORTANTE: se for rodar tratografia, NÃO use CLEANUP_AFTER=1 no passo 5 --
# rode a tratografia primeiro e só depois limpe (08b_cleanup.sh)
sbatch --array=1-$N 06_tractography.sh /caminho/work_dir
sbatch --array=1-$N 08b_cleanup.sh /caminho/work_dir   # se pulou o CLEANUP_AFTER acima

# 7. agrega tudo em tabelas/figuras (rápido, sem array)
sbatch 07_aggregate_and_plot.sh /caminho/work_dir
```

## Espaço em disco

Os arquivos pesados do pipeline são os `recon_target.nii.gz` (um volume 3D
por direção reconstruída, por sujeito, por combo) gerados nas etapas 3
(baseline) e 5 (RCAE) — dependendo da resolução dos seus volumes, isso pode
somar centenas de GB somando todos os combos de `experiments.tsv`. **Nem o
treino (etapa 4) nem a reconstrução em si (etapa 5) precisam ler de volta
esses arquivos** — cada etapa sempre parte do dwi original + do esquema de
subamostragem (`.npz`, leve). Os `recon_target.nii.gz` só existem para as
etapas 6/7/8 calcularem métricas uma vez; depois disso, o volume em si não
serve mais pra nada, só os CSVs de métricas importam.

Por isso o baseline SH virou dois passos: `02_baseline_sh.sh` só gera os
esquemas (índices, leve) e `02b_baseline_reconstruct.sh` reconstrói UM
combo por vez, em array — assim como o RCAE. Sem isso, um `sbatch` só do
baseline já geraria os 30 combos de uma vez, o pico de disco de tudo junto.
Combine com `CLEANUP_AFTER=1` no passo 5 (como no exemplo acima) para
apagar automaticamente o `recon_target.nii.gz` de cada combo (baseline e
RCAE) assim que as métricas dele forem calculadas — o pico de disco fica
limitado a poucos combos em voo por vez, nunca aos 30 acumulados. Se for
usar a tratografia (etapa 8), rode ela antes de limpar (ou use
`08b_cleanup.sh` manualmente depois). `target_idx.npy`, `mask.npy` e todos
os CSVs nunca são apagados.

## Encadeando com dependências

Em vez de esperar cada etapa terminar manualmente antes de submeter a
próxima, use `--dependency=afterok:<jobid>` (para um job único) ou
`afterok:<jobid_array>` (SLURM já trata o array inteiro). Exemplo
encadeando treino -> reconstrução -> avaliação para o mesmo array:

```bash
jid_base=$(sbatch --array=1-$N --parsable 02b_baseline_reconstruct.sh /caminho/work_dir)
jid1=$(sbatch --array=1-$N --parsable 03_train_rcae.sh /caminho/work_dir)
jid2=$(sbatch --array=1-$N --dependency=afterok:$jid1 --parsable 04_reconstruct_rcae.sh /caminho/work_dir)
# 05 le tanto o baseline (2b) quanto o RCAE (4), entao depende dos dois
CLEANUP_AFTER=1 sbatch --array=1-$N --dependency=afterok:$jid_base:$jid2 --parsable \
    05_evaluate_and_downstream.sh /caminho/work_dir
```

## Rodando só uma combinação (sem array)

Todo script aceita `shell_b` e `n_level` como argumentos extras, para testar
uma combinação isolada antes de disparar o array inteiro (recomendado: rode
uma vez assim, com poucas épocas, antes do `--array=1-N` grande):

```bash
sbatch 02b_baseline_reconstruct.sh /caminho/work_dir 1000 10
sbatch 03_train_rcae.sh /caminho/work_dir 1000 10
```

## Logs

Tudo cai em `slurm/logs/` (criado automaticamente), nomeado
`<etapa>.<jobid>_<array_task_id>.out/.err`.
