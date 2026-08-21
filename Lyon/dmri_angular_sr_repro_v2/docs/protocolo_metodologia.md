# Protocolo de reprodução: aumento de resolução angular em dMRI

Adaptado de Lyon, Cheng et al., *"Angular Super-Resolution in Diffusion MRI with a 3D Recurrent Convolutional Autoencoder"* (arXiv:2203.15598), com um baseline clássico de interpolação por harmônicos esféricos (SH) para comparação justa. Desenhado para rodar sobre dados próprios (single-shell e multi-shell, layout de diretórios próprio — `studies/<estudo>/<sessão>/<nome>_geomcorr.{nii,bval,bvec}`, não BIDS —, ~1000 sujeitos), não sobre o HCP.

## 1. Pergunta e hipótese

**Pergunta:** um modelo treinado para reconstruir um protocolo de dMRI com resolução angular reduzida (poucas direções) consegue recuperar sinal, métricas de microestrutura (DTI/NODDI) e, opcionalmente, tratografia equivalentes aos obtidos com o protocolo completo?

**Hipótese:** a reconstrução aprendida (RCAE) supera a interpolação clássica por SH em todas as métricas, principalmente em regimes de subamostragem mais agressiva (poucas direções).

## 1.1 Desenho para protocolos heterogêneos (o seu caso: 1000 sujeitos, protocolos variados)

Diferente do HCP (um protocolo único para todo mundo), aqui single-shell e
multi-shell variam entre sujeitos em b-value (500/700/750/1000/1500/2000,
não todo sujeito tem todos), número de direções e número de b0 — mesmo
dentro do "single-shell" já há variação de b-value e direções. TE e bobina
também variam, mas ainda não estão consolidados por sujeito; como o
pipeline trabalha com sinal normalizado pelo b0 (S/S0), a atenuação por
difusão é, em primeira aproximação, insensível a TE e a bobina (ambos
afetam sobretudo a intensidade absoluta do b0, que é dividida fora) — por
isso não é necessário esperar essa consolidação para começar. Quando TE e
bobina estiverem disponíveis, vale usá-los como uma checagem de robustez
(reavaliar se pooling por b-value ainda segura, estratificando os
resultados por TE/bobina), não como pré-requisito.

Dado isso, o desenho passa a ser **por shell**, não por "protocolo" fixo:

- **Unidade do experimento = um b-value alvo** (ex.: 1000), não um
  protocolo inteiro. Um sujeito entra no experimento de b=1000 se ele tiver
  essa shell — não importa se ele é "nativamente single-shell b=1000" ou
  se b=1000 é uma das shells de uma aquisição multi-shell dele. Isso
  aumenta bastante o N por b-value (ex.: um sujeito multi-shell com
  700/1000/2000 contribui para os três experimentos, não só para um).
- **QC antes de escolher os experimentos**: rode
  `scripts/01b_shell_availability_report.py` depois do manifesto para ver,
  por b-value candidato, quantos sujeitos disponíveis (nativos vs.
  extraídos de multi-shell), e a distribuição de nº de direções/b0. Não
  vale a pena montar um experimento (baseline + treino RCAE) para um
  b-value com poucos sujeitos (ex.: <20) — trate como nota exploratória, não
  como resultado principal da tese.
- **Split é global por sujeito**, feito uma única vez (`assign_splits`), e
  reusado em todos os experimentos por b-value — um sujeito multi-shell que
  entra em três experimentos (700/1000/2000) fica no mesmo split (train,
  val ou test) nos três, para não complicar a leitura dos resultados.
- **Contexto de aquisição como covariável de checagem**: como um mesmo
  b-value pode vir de aquisição nativamente single-shell ou de dentro de
  multi-shell, os scripts 06 e 07 já gravam uma coluna
  `acquisition_context` (`native_single_shell` vs. `from_multishell`) nas
  métricas. Reporte as métricas também estratificadas por esse contexto —
  se não houver diferença sistemática, reforça que o pooling é seguro; se
  houver, é um achado relevante por si (ex.: pode indicar diferença de
  TR/tempo de aquisição entre os protocolos que a normalização por b0 não
  cobre completamente).
- **DTI**: usa só b0 + a shell alvo, então funciona igual para sujeitos
  nativamente single-shell ou multi-shell — sem restrição adicional.
- **NODDI**: exige de verdade múltiplas shells (é mal-posto em
  single-shell) — por isso só roda para sujeitos com `n_shells >= 2`
  (`scripts/07_downstream_dti_noddi.py` já filtra isso automaticamente
  quando `--run-noddi` é passado). Ou seja, a validação de NODDI usa só o
  subconjunto genuinamente multi-shell do seu dataset, não os 1000
  sujeitos inteiros.
- **Ordem de prioridade sugerida**: comece pelos 2-3 b-values com mais
  sujeitos (provavelmente 1000 e/ou os mais comuns no multi-shell) — são o
  resultado principal, com N alto e baixo risco. b-values raros viram
  seção de generalização/discussão, não claim central.

## 2. Dados

- Fonte: seus próprios dados, em `studies/<estudo>/<sessão>/`, protocolos single-shell e multi-shell (não precisa reorganizar para BIDS — a descoberta varre recursivamente por sufixo de nome, ver `utils/manifest.discover_dwi_files`).
- Requisito mínimo por sujeito: `dwi.nii.gz`, `dwi.bval`, `dwi.bvec`, `dwi.json`, mais `b0`/`fmap` se disponíveis para correção de distorção.
- Pré-processamento (fora do escopo dos scripts de SR, mas necessário antes): denoising (`dwidenoise`), correção de Gibbs, `topup`+`eddy` (ou equivalente), correção de bias de campo (`N4`). Deixe isso resolvido antes de entrar no pipeline de super-resolução — caso contrário o modelo aprende a compensar artefato, não a interpolar q-space.
- Split por sujeito (não por volume, para não vazar informação): treino / validação / teste. Sugestão para 50+ sujeitos: 70% / 15% / 15%, estratificado por protocolo (single vs multi-shell) se os dois grupos não forem os mesmos indivíduos.

## 3. Desenho do experimento (ground truth por subamostragem)

Como no HCP, aqui a "verdade" é o próprio protocolo completo que você já adquiriu. Cada shell disponível (ex.: b=1000 no protocolo single-shell; b=1000/2000/3000 no multi-shell) é tratado separadamente:

1. Para cada shell, selecione um subconjunto de *N* direções já adquiridas (não se inventam direções nem se interpola no domínio de aquisição — trabalha-se sempre com o subconjunto real de gradientes medidos), usando amostragem por máxima dispersão angular (farthest-point sampling sobre a esfera), para simular uma aquisição de baixa resolução angular realista.
2. Testar múltiplos níveis de subamostragem (ex.: 6, 10, 15, 20, 30 direções, dependendo de quantas direções você tem no protocolo completo).
3. O modelo (ou o baseline SH) recebe o subconjunto e tenta reconstruir o sinal nas direções remanescentes (as que foram removidas), que servem de ground truth.
4. Repita em validação cruzada por nível de subamostragem — isso gera a curva "qualidade de reconstrução vs. número de direções de entrada", que é o gráfico central desse tipo de paper.

Para multi-shell, duas variantes valem a pena reportar: (a) subamostragem angular dentro de cada shell mantendo todas as shells (mede só resolução angular); (b) subamostragem angular combinada com uso de menos shells, se você quiser também discutir redução de tempo de aquisição total.

## 4. Métodos comparados

- **Baseline não-DL:** ajuste de harmônicos esféricos reais e simétricos (ordem par, escolhida conforme o número de direções de entrada) com regularização de Laplace-Beltrami, seguido de predição do sinal nas direções removidas. Zero treinamento, serve de piso de referência.
- **RCAE:** autoencoder convolucional 3D com módulo recorrente (GRU) que processa a sequência de direções de entrada e gera a sequência de direções alvo, condicionado à direção/b-value alvo. Treinado por shell (ou conjuntamente, se decidir compartilhar pesos entre shells).

## 5. Métricas

**Nível de sinal (reconstrução do DWI):**
- PSNR, SSIM (por volume 3D, direção a direção)
- NMSE / RMSE
- Coeficiente de correlação angular (ACC) entre ODFs/SH reconstruídas e ground truth
- Correlação de Pearson voxel a voxel (dispersão)

**Nível de microestrutura (downstream, o que realmente importa para a tese):**
- DTI (via DIPY, usando a shell b≈1000 reconstruída): FA, MD, AD, RD — comparação voxel a voxel + mapas de erro espacial, focando substância branca (usar máscara).
- NODDI (via AMICO, usando as shells reconstruídas combinadas, se multi-shell): NDI (ICVF), ODI, ISOVF.
- Testes estatísticos pareados (Wilcoxon signed-rank ou t-pareado, dependendo de normalidade) comparando baseline vs. RCAE vs. ground truth, por região de interesse (atlas) ou globalmente em substância branca.

**Nível de tratografia (opcional, se quiser um resultado mais forte para a tese):**
- CSD (MRtrix3) + `tckgen` nos dados reconstruídos e no ground truth.
- Comparação: Dice de densidade de streamlines por voxel, número de streamlines válidas/inválidas (se usar critério anatômico), tractometria ao longo de feixes de interesse (ex.: corpo caloso, fascículo longitudinal superior) com `pyBundleSeg`/`dipy.segment` ou `scilpy`.

## 6. Comparações e ablação

- Baseline SH vs. RCAE, em cada nível de subamostragem, em cada shell/protocolo.
- Ablação do RCAE (se o tempo permitir): remover o módulo recorrente (deixar só CNN 3D "por direção", sem contexto entre direções) para quantificar o ganho específico de modelar a dependência angular.
- Generalização: treinar no protocolo multi-shell e testar no single-shell (ou vice-versa) para checar se o modelo generaliza entre protocolos — isso é opcional, mas fortalece bastante a tese se dois protocolos estiverem disponíveis.

## 7. Ordem de execução

1. `01_prepare_data.py` — descobre sujeitos (varredura recursiva por sufixo de nome), valida shells, faz o split treino/val/teste.
2. `02_subsample_directions.py` — gera os subconjuntos de direções por nível de subamostragem.
3. `03_baseline_sh_interpolation.py` — reconstrução clássica (referência).
4. `04_train_rcae.py` — treina o modelo por nível de subamostragem (ou multi-nível, se preferir um único modelo condicionado ao número de entradas).
5. `05_reconstruct_rcae.py` — aplica o modelo treinado ao conjunto de teste.
6. `06_evaluate_reconstruction.py` — métricas de sinal (baseline e RCAE vs. ground truth).
7. `07_downstream_dti_noddi.py` — métricas de microestrutura.
8. `08_downstream_tractography.py` — opcional, tratografia.
9. `09_aggregate_and_plot.py` — tabelas e figuras finais.

## 8. Riscos e cuidados

- Não subamostrar de forma que a shell de entrada fique com menos direções do que o mínimo necessário para o ajuste SH de referência (baseline) funcionar (ordem par mínima costuma exigir pelo menos 6 direções para SH ordem 2, 15 para ordem 4 etc.).
- Garantir que o split treino/val/teste seja por sujeito, nunca por volume — do contrário há vazamento de dados entre treino e teste.
- Se os protocolos single e multi-shell forem de scanners/sequências diferentes, tratar como dois experimentos separados (não misturar no mesmo treino sem normalizar b-values/unidades).
- Documentar quantas direções cada shell realmente tem — isso limita quais níveis de subamostragem fazem sentido testar.
