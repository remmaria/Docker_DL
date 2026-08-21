# Reprodução: aumento de resolução angular em dMRI (dados próprios)

Pipeline para reproduzir, com seus dados (single-shell e multi-shell, layout
próprio em `studies/<estudo>/<sessão>/`, não precisa ser BIDS),
o experimento de super-resolução angular em dMRI adaptado de Lyon et al.
2022 (RCAE, arXiv:2203.15598), com um baseline clássico de interpolação por
harmônicos esféricos para comparação. Ver `docs/protocolo_metodologia.md`
para o desenho experimental completo (métricas, splits, comparações,
ablação) — esse documento é o que deve ir para a seção de métodos da tese.
Para rodar no cluster SLURM (baseado no seu script de treino), ver `slurm/README.md`.

## Status deste código

Escrito e revisado neste ambiente, mas **não executado ponta a ponta**: o
sandbox de desenvolvimento não tem acesso à PyPI para instalar
`nibabel`/`dipy`/`torch` (só `numpy`/`scipy`/`pandas`/`matplotlib`, que já
vinham instalados). Cada peça foi validada dentro do possível:

- `utils/gradients.py` (farthest-point sampling, split de shells) — testado com dados sintéticos.
- `utils/metrics.py` (PSNR/SSIM/NMSE/ACC/Wilcoxon) — testado com dados sintéticos.
- `utils/sh_basis.py` (ajuste/predição SH) — testado com sinal sintético suave, reconstrução em direções held-out com correlação ~0.96.
- `utils/manifest.py` (descoberta recursiva por sufixo de nome, assinatura de shells por sujeito, split treino/val/teste global) — testado com árvore sintética no mesmo layout do seu cluster (`studies/<estudo>/<sessão>/bgpdwis_PA_geomcorr.{nii,bval,bvec}`, incluindo `.nii` sem compressão, arquivos de máscara/FA/MD "espúrios" no meio e nomes de sessão repetidos entre estudos) — encontrou só os 4 sujeitos reais, ignorou o lixo, sem colisão de ID.
- `scripts/01b_shell_availability_report.py` (QC de disponibilidade de shells) — testado ponta a ponta com manifesto sintético heterogêneo (~38 sujeitos, protocolos variados).
- `model/rcae.py` — revisado manualmente (não executável aqui por falta de `torch`); inclui `_smoke_test()` — rode `python -m model.rcae` no cluster antes de treinar de verdade, para validar shapes.
- `scripts/09_aggregate_and_plot.py` — testado ponta a ponta com CSVs sintéticos, gera as figuras e tabelas esperadas.
- Os scripts que dependem de `nibabel`/`dipy`/`torch`/MRtrix3 (01, 03, 04, 05, 07, 08 nas partes de I/O real) tiveram só a sintaxe checada (`py_compile`), não a execução real. Rode um teste pequeno (2-3 sujeitos, poucas épocas) no cluster antes de rodar a base toda.

## Instalação (no cluster)

```bash
pip install -r requirements.txt
# NODDI (opcional): pip install dmri-amico
# Tratografia (opcional): instalar MRtrix3 separadamente (não é pacote pip)
```

## Ordem de execução

```bash
# 1. Descobre sujeitos (varredura recursiva por sufixo de nome, ex.: _geomcorr),
#    registra a assinatura de shells de cada um (b-values, n_direcoes, n_b0
#    -- sem exigir protocolo uniforme), faz o split global
python scripts/01_prepare_data.py --data-root /ix1/tibrahim/rmm270/DATA/DWIs/studies \
    --name-suffix _geomcorr --out-dir work_dir

# 1b. QC: quantos sujeitos tem cada b-value candidato (nativo vs. extraido de
#     multi-shell) -- rode isso antes de decidir quais experimentos valem a pena
python scripts/01b_shell_availability_report.py --manifest work_dir/manifest.csv \
    --candidate-bvalues 500 700 750 1000 1500 2000 \
    --out-csv work_dir/shell_availability.csv

# 2. Gera esquema de subamostragem angular (por shell, por sujeito)
python scripts/02_subsample_directions.py --manifest work_dir/manifest.csv \
    --out-dir work_dir/subsampling --levels 6 10 15 20 30

# 3. Baseline SH (sem treino, referência rápida)
python scripts/03_baseline_sh_interpolation.py --manifest work_dir/manifest.csv \
    --scheme-dir work_dir/subsampling --out-dir work_dir/baseline_recon --split test

# 4. Treina RCAE (repita para cada shell/nivel que quiser cobrir)
python scripts/04_train_rcae.py --manifest work_dir/manifest.csv \
    --scheme-dir work_dir/subsampling --shell-b 1000 --n-level 10 \
    --out-dir work_dir/rcae_checkpoints --epochs 100

# 5. Reconstrói o conjunto de teste com o modelo treinado
python scripts/05_reconstruct_rcae.py --manifest work_dir/manifest.csv \
    --scheme-dir work_dir/subsampling \
    --checkpoint work_dir/rcae_checkpoints/shell1000_n10/best.pt \
    --shell-b 1000 --n-level 10 --out-dir work_dir/rcae_recon

# 6. Métricas de sinal (baseline vs. RCAE vs. ground truth)
python scripts/06_evaluate_reconstruction.py --manifest work_dir/manifest.csv \
    --baseline-dir work_dir/baseline_recon --rcae-dir work_dir/rcae_recon \
    --shell-b 1000 --n-level 10 --out-csv work_dir/metrics/signal_metrics_shell1000_n10.csv

# 7. Downstream: DTI (+ NODDI opcional com --run-noddi)
python scripts/07_downstream_dti_noddi.py --manifest work_dir/manifest.csv \
    --baseline-dir work_dir/baseline_recon --rcae-dir work_dir/rcae_recon \
    --shell-b 1000 --n-level 10 --out-dir work_dir/downstream

# 8. (Opcional) Tratografia via MRtrix3
python scripts/08_downstream_tractography.py --manifest work_dir/manifest.csv \
    --baseline-dir work_dir/baseline_recon --rcae-dir work_dir/rcae_recon \
    --shell-b 1000 --n-level 10 --out-dir work_dir/tractography

# 9. Agrega tudo em tabelas e figuras finais
python scripts/09_aggregate_and_plot.py --metrics-dir work_dir/metrics \
    --downstream-dir work_dir/downstream --tractography-dir work_dir/tractography \
    --out-dir work_dir/figures
```

Repita os passos 2 (uma vez só, já cobre todos os níveis) a 8 para cada
`--shell-b`/`--n-level` que quiser reportar (ex.: shell 1000 com 6, 10, 15,
20, 30 direções); o passo 9 concatena tudo automaticamente ao final.

## Estrutura esperada dos dados

Não precisa ser BIDS — a descoberta (`utils/manifest.discover_dwi_files`) varre
`--data-root` recursivamente procurando qualquer arquivo terminando em
`<--name-suffix>.bval` que tenha `.bvec` e `.nii`/`.nii.gz` companheiros
(mesmo nome, mesma pasta); outros arquivos na mesma pasta (máscaras, mapas
de FA/MD, dados brutos sem bval/bvec) são ignorados automaticamente. No seu
caso:

```
DATA/DWIs/studies/
  all_bias/                                    # seus ~1000 sujeitos moram aqui
    20160914203805_160914-volunteer/
      bgpdwis_PA_geomcorr.nii
      bgpdwis_PA_geomcorr.bval
      bgpdwis_PA_geomcorr.bvec
      bgpdwis_PA_geomcorr_mask3d.nii.gz         # máscara de cérebro (ver abaixo)
      (outros arquivos na pasta, ex. mapas de FA/MD, são ignorados)
    <outra-pasta-de-sessão>/
      ...
```

Como `all_bias` é a pasta com todos os 1000 sujeitos (não múltiplos
subestudos), aponte `--data-root` direto para ela:
`--data-root /ix1/tibrahim/rmm270/DATA/DWIs/studies/all_bias`. Nesse caso o
identificador do sujeito é só o nome da pasta de sessão (ex.:
`20160914203805_160914-volunteer`), sem prefixo de estudo. Se um dia você
apontar `--data-root` para `studies/` (o nível acima, com múltiplas pastas
tipo `all_bias`), o identificador vira `<pasta>__<pasta_sessão>`
automaticamente, para não colidir entre pastas — mas para o `all_bias`
sozinho isso não é necessário. `--name-suffix` default é `_geomcorr`.

**Máscara de cérebro**: seu arquivo `bgpdwis_PA_geomcorr_mask3d.nii.gz`
já bate com o padrão que os scripts procuram por padrão
(`--mask-suffix _mask3d.nii.gz`, aplicado como
`<dwi_sem_extensão><mask_suffix>`) — não precisa passar nada extra, é
usado automaticamente em vez do threshold simples no b0.

## Protocolos heterogêneos (single/multi-shell com b-value, n_direções, n_b0 variáveis)

O pipeline foi ajustado para tratar cada **shell** (b-value) como a unidade
do experimento, não um "protocolo" fixo. Um sujeito multi-shell com
700/1000/2000 participa dos três experimentos possíveis (um por b-value);
um sujeito single-shell b=1000 participa só do experimento de b=1000. O
split treino/val/teste é global por sujeito (`utils/manifest.assign_splits`)
e reusado em todos os experimentos por b-value que esse sujeito integra.
Rode `scripts/01b_shell_availability_report.py` logo depois do passo 1 para
ver, por b-value candidato, quantos sujeitos há e se vale a pena montar um
experimento (baseline + RCAE) para ele. As métricas de sinal (etapa 6) e de
DTI/NODDI (etapa 7) já gravam uma coluna `acquisition_context`
(`native_single_shell` vs. `from_multishell`) para você checar se há viés
sistemático entre sujeitos que tinham aquele b-value nativamente e sujeitos
onde ele veio de dentro de uma aquisição multi-shell maior. NODDI (etapa 7,
`--run-noddi`) só roda para sujeitos com `n_shells >= 2` — é
matematicamente mal-posto em dados single-shell, então não faz sentido
tentar nos outros. Ver seção 1.1 de `docs/protocolo_metodologia.md` para o
raciocínio completo, incluindo por que TE e bobina (ainda não consolidados
no seu caso) provavelmente não exigem grupos separados, já que o pipeline
normaliza tudo por S/S0.

## Pontos de atenção antes de rodar

- Pré-processe antes (denoising, `topup`+`eddy`, correção de bias) — os
  scripts aqui assumem dados já corrigidos. Ver seção 2 do protocolo.
- Máscara de cérebro: os scripts procuram, por padrão,
  `<dwi_sem_extensão><--mask-suffix>` na mesma pasta — o default já é
  `--mask-suffix _mask3d.nii.gz`, batendo com o seu
  `bgpdwis_PA_geomcorr_mask3d.nii.gz` automaticamente, sem precisar passar
  nada. Se algum subestudo usar outro padrão de nome de máscara, passe
  `--mask-suffix` correspondente nesse script (03, 05, 07, 08). Sem
  encontrar a máscara, cai num threshold simples no b0 médio
  (`utils/masking.simple_brain_mask`) — funcional para não travar o
  pipeline, mas a sua máscara de verdade dá métricas mais confiáveis.
- Ordem SH mínima: níveis de subamostragem muito baixos (ex. 6 direções)
  limitam a ordem SH ajustável no baseline — o script já calcula isso
  automaticamente (`utils/sh_basis.max_order_for_n_directions`), mas
  significa que o baseline fica fraco nesses níveis por construção (o que,
  aliás, é esperado e reforça o argumento a favor do RCAE).
- `model/rcae.py` agora é uma reprodução estrutural fiel da arquitetura do
  paper (portada manualmente do código oficial TensorFlow/Keras,
  github.com/m-lyon/dMRI-RCNN): blocos multi-ramo (kernels 1/2/3 em
  paralelo, concatenados) com as mesmas contagens de canal (104–668),
  InstanceNorm3d no primeiro estágio + BatchNorm3d nos seguintes, ativação
  swish, reinjeção do bvec em cada estágio (encoder e decoder), e uma
  `ConvLSTM3D` (não mais uma ConvGRU customizada) agregando a sequência de
  direções de entrada. Os canais são fixos (não são mais hiperparâmetros
  `base_ch`/`encoder_out_ch`/`gru_hidden_ch` ajustáveis) — só
  `--lstm-size` (default 48, igual ao paper) é configurável. Diferenças de
  fidelidade que ainda restam (documentadas no topo de `model/rcae.py`):
  padding assimétrico "SAME" do TF já é replicado; a ativação recorrente
  da ConvLSTM (`hard_sigmoid` no Keras) foi aproximada por sigmoid comum;
  a inicialização de pesos usa `kaiming_uniform_` (próximo de
  `he_uniform`, não bit-a-bit idêntico); os modelos alternativos do paper
  (`get_1d_*`, mais leves) e o carregamento de pesos pré-treinados não
  foram portados.
- Com os canais bem maiores que antes, `--batch-size`/`--patch-size`
  ficam mais sensíveis à memória de GPU — os defaults atuais
  (`--patch-size 10 --batch-size 4`, iguais ao paper) já refletem isso;
  se estourar VRAM, reduza `--batch-size` antes de mexer na arquitetura
  (mudar canais quebraria a fidelidade).
- **Checkpoints antigos (treinados antes desta reprodução fiel da
  arquitetura) não são compatíveis** — o `state_dict` mudou de forma
  completa (chaves e formas diferentes). É preciso retreinar do zero com
  o pipeline atual antes de rodar `05_reconstruct_rcae.py`.