#!/usr/bin/env python3
"""
Etapa 4i ("ensemble em estrela" da linha PairFlow -- ver model/pairflow_star.py
e pedido explicito da usuaria em 2026-09-03, "quero o pairflow ensemble"):
treina a PairFlowStar para um (shell, n_level) especifico, usando os feixes
de M pares diversos ja construidos por scripts/02b_build_rrin_triplets.py
--ensemble-m M -- MESMO esquema de trincas/feixe usado por
scripts/04e_train_rrin_star.py (RRIN3DStar), so' trocando o modelo.

Espelha DELIBERADAMENTE scripts/04e_train_rrin_star.py (mesmo dataset --
utils/rrin_dataset.py:RRINTripletDataset --, mesmo manifesto/split, mesmo
layout de checkpoint out_dir/<run_tag>/{best,last}.pt, mesmo resume
automatico, mesmo formato de batch_log.csv) e incorpora de
scripts/04h_train_pairflow_finetune.py o `--init-checkpoint` (carrega so' os
pesos de `flow_net` de um checkpoint da Etapa 1,
scripts/04g_train_pairflow_ssl.py) e o `--freeze-flow` (congela `flow_net`
durante este treino, so' refine_net/weight_head sao treinados).

DIFERENCAS em relacao a 04e_train_rrin_star.py: nao ha `--use-quality-cond`
(a linha PairFlow nao condiciona o FlowNet -- ver model/pairflow_ssl.py,
PairFlowNet3D nao tem esse parametro), so' `--weight-quality-cond` (condiciona
a PairFlowWeightHead3D, ver model/pairflow_star.py).

Start do treino (ADITIVO, ambos default DESLIGADOS, trazidos de
scripts/04f_train_implicit.py a pedido da usuaria em 2026-09-04, ver addendum
secao 33.19): `--warmup-steps N` sobe a LR linearmente de 0.1*--lr ate --lr ao
longo dos N primeiros passos de otimizador (mesmo mecanismo do implicit,
ignorado com aviso se --resume-checkpoint apontar pra um checkpoint
existente). `--zero-init-refine-output` zera a ultima camada de
model.refine_net (RefineNet3D.net[-1]) na inicializacao -- e' o analogo
arquiteturalmente apropriado do `--init-output-bias-from-data` do implicit:
la' o decoder produz o alvo do zero (bias-init no valor medio do alvo ajuda);
aqui pred = blend + residual, onde blend ja e' um sinal real (interpolacao
via fluxo optico dos volumes medidos), entao zerar o residual so' faz a
predicao inicial coincidir exatamente com o blend, em vez de fazer o modelo
"aprender" um deslocamento constante irrelevante nas primeiras iteracoes.

Uso (com pre-treino da Etapa 1):
    python scripts/04i_train_pairflow_star.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --ensemble-m 3 \
        --init-checkpoint work_dir/pairflow_ssl_checkpoints/shell1000/best.pt \
        --out-dir work_dir/pairflow_star_checkpoints \
        --epochs 100 --batch-size 8 --patch-size 10 --lr 1e-4

Uso (controle, sem pre-treino -- treina do zero nas trincas curadas):
    python scripts/04i_train_pairflow_star.py \
        --manifest work_dir/manifest.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 --ensemble-m 3 \
        --out-dir work_dir/pairflow_star_checkpoints \
        --epochs 100 --batch-size 8 --patch-size 10 --lr 1e-4

Requer PyTorch + GPU. Nao executado neste ambiente de desenvolvimento --
revisado manualmente, testado por compilacao de sintaxe. Modelo verificado
por smoke test proprio (python -m model.pairflow_star).

Iteracao rapida (pedido explicito da usuaria em 2026-09-03, "quero acelerar
o treino... sem ter que esperar a epoca inteira pra ver se deu certo"):
--max-train-batches/--max-val-batches (DEBUG, default None = sem efeito)
limitam quantos batches cada epoca processa, permitindo rodar uma "epoca"
inteira (treino+val) em segundos/minutos pra checar se o codigo funciona e
se a loss esta descendo, mesmo com o manifesto/dataset completo. Combine
com scripts/99_make_mini_manifest.py (reduz o NUMERO DE SUJEITOS do
manifesto, o que tambem acelera o carregamento/cache de dados) para o ciclo
de debug mais rapido possivel -- as duas tecnicas sao independentes e
compoem. NENHUMA das duas serve para avaliar convergencia real (curva de
varias epocas completas) -- so' para correcao de bugs e um sinal grosseiro
de "a loss esta caindo em poucos batches".

Uso rapido de debug (poucos batches, poucas epocas, sem ocupar a GPU por
muito tempo):
    python scripts/04i_train_pairflow_star.py \
        --manifest work_dir/manifest_mini.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 --ensemble-m 3 \
        --out-dir work_dir/pairflow_star_checkpoints_debug \
        --epochs 3 --batch-size 4 --patch-size 10 --lr 1e-4 \
        --num-workers 0 --val-num-workers 0 \
        --max-train-batches 5 --max-val-batches 2 --no-resume

Mini-teste de gargalo de I/O (--freeze-subject-order, ver addendum secao
30/33 -- hipotese ainda nao avaliada: particionamento por worker JA' ativo
+ ordem congelada pode deixar o cache de pagina do SO esquentar a partir
da 2a epoca, baixando o "wait" que hoje fica em ~93-99%). Rode o MESMO
comando duas vezes (uma com a flag, uma sem) num manifesto mini com pelo
menos ~2x --num-workers sujeitos por split (pra ter mais de 1 sujeito por
worker e a chance de reler o mesmo arquivo importar), varias epocas
completas (nao so' --max-train-batches -- o efeito de cache so' aparece
DE EPOCA PRA EPOCA), e compare a linha "[train] epoca N resumo: ... wait
total ... (X%)" entre as duas rodadas e entre epocas 1 e 4-6 da MESMA
rodada:
    python scripts/04i_train_pairflow_star.py \
        --manifest work_dir/manifest_mini.csv \
        --triplets-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 --ensemble-m 3 \
        --out-dir work_dir/pairflow_star_checkpoints_debug_freeze \
        --epochs 6 --batch-size 8 --patch-size 10 --lr 1e-4 \
        --num-workers 8 --val-num-workers 0 --no-resume \
        --freeze-subject-order
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.rrin_dataset import RRINTripletDataset
from utils.dataset import SubjectGroupedSampler, worker_init_fn
from utils.sh_basis import max_order_for_n_directions
from utils.sh_angular_loss import n_coeffs_even, compute_sh_angular_loss
from model.pairflow_star import build_pairflow_star_model


def _sh_bundle_forward_star(model, batch, device, weight_quality_cond: bool):
    """Analogo a scripts/04b_train_rrin.py:_sh_bundle_forward, adaptado para
    PairFlowStar (porte pedido pela usuaria em 2026-09-09, 'como colocar
    angular loss neles?'). Roda o modelo sobre o feixe `*_sh` do batch (K
    trincas de UMA direcao-alvo cada, ver utils/rrin_dataset.py:
    RRINTripletDataset.sh_q_out -- MESMO mecanismo ja usado pelo RRIN,
    PairFlowStar ja importa a MESMA RRINTripletDataset).

    CORRECAO (2026-09-09, ver addendum secao 33.34 -- a versao anterior
    desta funcao rodava cada uma das K direcoes do feixe SH como um
    ensemble DEGENERADO de M=1, por nao haver ainda infraestrutura de
    dataset pra um "feixe de ensembles"): confirmado que essa
    infraestrutura ja existia, so' nao estava sendo lida -- os campos
    `ens_pair_a`/`ens_valid`/etc gravados por
    scripts/02b_build_rrin_triplets.py ja tem shape (n_alvos, M), um
    ensemble de M candidatos por TRINCA (nao so' pela trinca escolhida
    como item principal), entao utils/rrin_dataset.py:RRINTripletDataset
    agora le esse ensemble tambem pras `sh_q_out` trincas do feixe SH
    (campos "*_sh_ens", ver docstring de sh_q_out la'). Esta funcao usa
    esse ensemble COMPLETO -- cada uma das K direcoes do feixe roda pelo
    modelo com ate' `ensemble_m` pares candidatos reais (mesmo M usado
    pelo item principal), nao mais uma aproximacao M=1. Motivacao: a
    comparacao empirica feita pela usuaria (mesmo INIT_CHECKPOINT/
    ZERO_INIT_REFINE_OUTPUT/seed nos dois runs, ver addendum 33.34) mostrou
    a aproximacao M=1 antiga custando ~22% de loss_signal ja na epoca 1 --
    hipotese principal: as previsoes 'M=1' do feixe SH sao bem mais
    ruidosas/fracas que as previsoes 'M=--ensemble-m' do item principal, e
    empurrar o modelo pra ficar 'consistente angularmente' contra alvos
    ruidosos puxava os pesos pra longe do otimo da tarefa principal."""
    vol_a_sh_ens = batch["vol_a_sh_ens"].to(device)     # (B, K, M, 1, ps, ps, ps)
    vol_b_sh_ens = batch["vol_b_sh_ens"].to(device)
    target_sh = batch["target_sh"].to(device)           # (B, K, 1, ps, ps, ps)
    bvec_a_sh_ens = batch["bvec_a_sh_ens"].to(device)    # (B, K, M, 3)
    bvec_b_sh_ens = batch["bvec_b_sh_ens"].to(device)
    bvec_t_sh_ens = batch["bvec_t_sh_ens"].to(device)    # (B, K, M, 3) -- ignorado pelo modelo (ver forward)
    bvec_t_sh = batch["bvec_t_sh"].to(device)            # (B, K, 3) -- direcao-alvo REAL, usada so' pela loss angular
    t_frac_sh_ens = batch["t_frac_sh_ens"].to(device)    # (B, K, M)
    ensemble_mask_sh = batch["ensemble_mask_sh"].to(device)  # (B, K, M) bool
    sh_mask = batch["sh_mask"].to(device)                # (B, K) bool -- direcoes REAIS do feixe (nao padding)

    B, K, M = vol_a_sh_ens.shape[0], vol_a_sh_ens.shape[1], vol_a_sh_ens.shape[2]
    # achata (B,K) -> B*K, mantendo o eixo M do ensemble -- MESMO truque de
    # achatamento de run_epoch()/model(vol_a, ...) pro item principal, so'
    # que aqui B*K faz o papel do "batch" e M continua sendo o eixo do
    # ensemble em estrela de verdade (nao mais um M=1 degenerado).
    vol_a_flat = vol_a_sh_ens.reshape(B * K, M, *vol_a_sh_ens.shape[3:])
    vol_b_flat = vol_b_sh_ens.reshape(B * K, M, *vol_b_sh_ens.shape[3:])
    bvec_a_flat = bvec_a_sh_ens.reshape(B * K, M, 3)
    bvec_b_flat = bvec_b_sh_ens.reshape(B * K, M, 3)
    bvec_t_flat = bvec_t_sh_ens.reshape(B * K, M, 3)
    t_frac_flat = t_frac_sh_ens.reshape(B * K, M)
    ensemble_mask_flat = ensemble_mask_sh.reshape(B * K, M)
    quality_flat = None
    if weight_quality_cond:
        quality_flat = batch["quality_sh_ens"].to(device).reshape(B * K, M, 2)

    pred_flat = model(vol_a_flat, vol_b_flat, bvec_a_flat, bvec_b_flat, bvec_t_flat,
                       t_frac_flat, ensemble_mask_flat, quality=quality_flat)
    pred_sh = pred_flat.reshape(B, K, *pred_flat.shape[1:])
    return pred_sh, target_sh, bvec_t_sh, sh_mask


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int,
              need_quality: bool = False, batch_log_f=None, max_batches: int = None,
              warmup_state: dict = None, angular_loss_weight: float = 0.0,
              sh_loss_high_order_min: int = 4, sh_loss_lmax_cap: int = 8):
    model.train(mode=train)
    total_loss = 0.0
    total_loss_signal = 0.0
    total_loss_angular = 0.0
    n_batches = 0
    n_samples = 0
    total_wait_s = 0.0
    total_compute_s = 0.0
    split = "train" if train else "val"
    prev_end = time.time()
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        t_received = time.time()
        wait_s = t_received - prev_end

        vol_a = batch["vol_a_ens"].to(device)          # (B,M,1,ps,ps,ps)
        vol_b = batch["vol_b_ens"].to(device)
        target = batch["target"].to(device)            # (B,1,ps,ps,ps) -- MESMO alvo p/ todo o feixe
        bvec_a = batch["bvec_a_ens"].to(device)         # (B,M,3)
        bvec_b = batch["bvec_b_ens"].to(device)
        bvec_t = batch["bvec_t_ens"].to(device)         # (B,M,3) -- ignorado pelo modelo, ver forward
        t_frac = batch["t_frac_ens"].to(device)         # (B,M)
        ensemble_mask = batch["ensemble_mask"].to(device)  # (B,M) bool
        quality = batch["quality_ens"].to(device) if need_quality else None

        with torch.set_grad_enabled(train):
            pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, ensemble_mask,
                         quality=quality)
            # MAE, mesma escolha de 04e_train_rrin_star.py/04h_train_pairflow_finetune.py.
            loss_signal = (pred - target).abs().mean()
            # termo angular/SH opcional (porte pedido pela usuaria em
            # 2026-09-09, ver _sh_bundle_forward_star acima) -- desativado
            # por padrao (angular_loss_weight=0.0), loss identica a antes.
            if angular_loss_weight > 0:
                pred_sh, target_sh, bvec_t_sh, sh_mask = _sh_bundle_forward_star(
                    model, batch, device, need_quality)
                loss_angular = compute_sh_angular_loss(
                    pred_sh, target_sh, bvec_t_sh, sh_mask,
                    l_max_cap=sh_loss_lmax_cap, high_order_min=sh_loss_high_order_min)
                loss = loss_signal + angular_loss_weight * loss_angular
            else:
                loss_angular = None
                loss = loss_signal
            if train:
                if warmup_state is not None and warmup_state["step"] < warmup_state["warmup_steps"]:
                    warmup_state["step"] += 1
                    frac = warmup_state["step"] / warmup_state["warmup_steps"]
                    start_factor = warmup_state["start_factor"]
                    lr_now = warmup_state["base_lr"] * (start_factor + frac * (1.0 - start_factor))
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr_now
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_compute_end = time.time()
        compute_s = t_compute_end - t_received

        total_loss += loss.item()
        total_loss_signal += loss_signal.item()
        if loss_angular is not None:
            total_loss_angular += loss_angular.item()
        n_batches += 1
        n_samples += vol_a.shape[0]
        total_wait_s += wait_s
        total_compute_s += compute_s

        if batch_log_f is not None:
            tags_str = ";".join(batch["subject_tag"])
            n_real_mean = ensemble_mask.float().sum(dim=1).mean().item()
            loss_angular_str = f"{loss_angular.item():.6f}" if loss_angular is not None else ""
            batch_log_f.write(f"{epoch},{split},{n_batches},{loss.item():.6f},"
                               f"{wait_s:.3f},{compute_s:.3f},{tags_str},{n_real_mean:.2f},"
                               f"{loss_signal.item():.6f},{loss_angular_str}\n")
            batch_log_f.flush()

        prev_end = time.time()

    if n_batches > 0:
        total_s = total_wait_s + total_compute_s
        throughput = n_samples / total_s if total_s > 0 else float("nan")
        pct_wait = 100 * total_wait_s / total_s if total_s > 0 else float("nan")
        print(f"[{split}] epoca {epoch} resumo: {n_batches} batches, {n_samples} patches | "
              f"wait total {total_wait_s:.1f}s ({pct_wait:.0f}%) | compute total "
              f"{total_compute_s:.1f}s | {throughput:.2f} patches/s", flush=True)

    avg_loss_signal = total_loss_signal / max(1, n_batches)
    avg_loss_angular = (total_loss_angular / max(1, n_batches)) if angular_loss_weight > 0 else None
    return total_loss / max(1, n_batches), avg_loss_signal, avg_loss_angular


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--triplets-dir", required=True,
                     help="pasta com os <tag>_rrin_triplets.npz da etapa 2b -- PRECISA ter "
                          "sido gerada com --ensemble-m >= --ensemble-m deste script (MESMO "
                          "esquema de feixe usado por scripts/04e_train_rrin_star.py).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--ensemble-m", type=int, default=3,
                     help="M do ensemble em estrela (mesma semantica de --ensemble-m em "
                          "scripts/04e_train_rrin_star.py). Default 3.")
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base-ch", type=int, default=16)
    ap.add_argument("--max-disp", type=float, default=0.5)
    ap.add_argument("--weight-quality-cond", action="store_true",
                     help="alimenta residual_deg/gap_deg de cada par diretamente na "
                          "PairFlowWeightHead3D (a cabeca que decide o peso de fusao entre os "
                          "M pares) -- ver docstring de model/pairflow_star.py. NAO ha "
                          "--use-quality-cond equivalente aqui: PairFlowNet3D nao condiciona o "
                          "fluxo por nenhum sinal de qualidade (ver model/pairflow_ssl.py).")
    ap.add_argument("--norm-type", choices=["instance", "batch"], default="instance",
                     help="mesmas restricoes de scripts/04e_train_rrin_star.py (batch exige "
                          "treino do zero).")
    ap.add_argument("--no-only-valid", action="store_true",
                     help="mesmo espirito de --no-only-valid em 04e_train_rrin_star.py.")
    ap.add_argument("--init-checkpoint", default=None,
                     help="checkpoint da Etapa 1 (scripts/04g_train_pairflow_ssl.py) -- carrega "
                          "SO os pesos de `flow_net` pra inicializar (ver model/pairflow_star.py "
                          "e scripts/04h_train_pairflow_finetune.py, mesmo mecanismo). Default "
                          "None = treina PairFlowStar do zero (controle).")
    ap.add_argument("--freeze-flow", action="store_true",
                     help="congela flow_net durante o treino do ensemble (so' refine_net/"
                          "weight_head sao treinados) -- ver model.pairflow_star.PairFlowStar. "
                          "Requer --init-checkpoint (nao faz sentido congelar um fluxo do zero).")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2)
    ap.add_argument("--val-num-workers", type=int, default=None)
    ap.add_argument("--val-max-cached-subjects", type=int, default=1)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-id", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--resume-checkpoint", default=None)
    ap.add_argument("--freeze-subject-order", action="store_true",
                     help="EXPERIMENTAL (mesmo flag de scripts/04g_train_pairflow_ssl.py, "
                          "trazido aqui em 2026-09-03 especificamente para o mini-teste de "
                          "gargalo de I/O pedido pela usuaria), default DESLIGADO -- "
                          "comportamento default continua sendo SubjectGroupedSampler("
                          "freeze_order=False), ordem dos sujeitos reembaralhada a cada epoca. "
                          "Passar esta flag ativa SubjectGroupedSampler(freeze_order=True): a "
                          "ORDEM dos sujeitos fica fixa entre epocas -- COMBINADO com o "
                          "particionamento por worker (ja ativo aqui via num_workers/batch_size, "
                          "ver utils.dataset.SubjectGroupedSampler.__init__ e addendum secao 30), "
                          "isso faz cada worker reler SEMPRE os MESMOS arquivos a cada epoca, "
                          "dando chance real do cache de pagina do sistema de arquivos de rede "
                          "esquentar a partir da 2a epoca -- ainda NAO testado (freeze_order "
                          "sozinho, SEM particionamento, foi tentado e revertido em 04g/"
                          "pairflow_ssl em 2026-09-02 por nao ter ajudado e ter coincidido com "
                          "uma rodada travada; a combinacao com particionamento e' uma hipotese "
                          "nova, ainda nao avaliada em producao -- prefira testar primeiro num "
                          "job curto/mini-manifesto antes de usar num treino longo).")
    ap.add_argument("--max-train-batches", type=int, default=None,
                     help="DEBUG: limita quantos batches de treino sao processados por epoca, "
                          "independente do tamanho real do dataset/DataLoader -- serve pra "
                          "iterar rapido em mudancas de codigo (ex.: --init-checkpoint/"
                          "--freeze-flow) sem esperar uma epoca inteira no dataset completo. "
                          "Default None = sem limite (comportamento identico ao anterior). "
                          "Ver scripts/99_make_mini_manifest.py para tambem reduzir o numero "
                          "de sujeitos.")
    ap.add_argument("--max-val-batches", type=int, default=None,
                     help="DEBUG: mesmo espirito de --max-train-batches, mas para a epoca de "
                          "validacao. Default None = sem limite.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                     help="ADITIVO, default 0 (desligado). Numero de passos de OTIMIZADOR "
                          "(nao epocas) durante os quais a LR sobe linearmente de 0.1*--lr ate "
                          "--lr, antes do ReduceLROnPlateau assumir -- mesmo mecanismo trazido "
                          "de scripts/04f_train_implicit.py (ver addendum secao 33.18/33.19). "
                          "Ignorado (com aviso impresso) se --resume-checkpoint apontar para um "
                          "checkpoint existente, pois warmup so' faz sentido comecando do zero.")
    ap.add_argument("--zero-init-refine-output", action="store_true",
                     help="ADITIVO, default DESLIGADO. Zera peso e bias da ultima camada de "
                          "RefineNet3D (model.refine_net.net[-1], ver model/rrin3d.py) logo apos "
                          "a criacao do modelo (e apos --init-checkpoint, que so' carrega "
                          "flow_net) -- forca residual_f=0 no inicio do treino, entao a predicao "
                          "inicial (pred = blend + residual) comeca EXATAMENTE igual ao blend "
                          "(interpolacao via fluxo optico), que ja e' um sinal real plausivel. "
                          "Este e' o analogo arquiteturalmente apropriado de "
                          "--init-output-bias-from-data (de 04f_train_implicit.py) para "
                          "PairFlowStar: la' o decoder produz o alvo do zero (bias-init no valor "
                          "medio do alvo faz sentido); aqui a predicao ja parte de um blend com "
                          "magnitude de sinal real, entao a tecnica analoga e' a classica "
                          "'zero-init da ultima camada do ramo residual' (ResNet-style), nao "
                          "bias-init -- ver addendum secao 33.19 para a discussao completa.")
    ap.add_argument("--angular-loss-weight", type=float, default=0.0,
                     help="lambda do termo de loss opcional no dominio angular/SH (porte de "
                          "scripts/04_train_rcae.py/04b_train_rrin.py, ver protocolo secao "
                          "9/14.5 -- pedido explicito da usuaria em 2026-09-09). Default 0.0 = "
                          "DESATIVADO, comportamento identico ao treino sem esse termo. Com "
                          "peso > 0, o dataset TAMBEM monta um feixe de --sh-loss-q-out trincas "
                          "do mesmo sujeito/patch por item (ver utils/rrin_dataset.py, ja usado "
                          "pelo RRIN) -- cada uma das trincas do feixe roda pelo modelo com um "
                          "ensemble COMPLETO de ate --ensemble-m pares candidatos (ver "
                          "_sh_bundle_forward_star acima; corrigido em 2026-09-09, addendum "
                          "secao 33.34 -- versao anterior usava so' M=1 por direcao do feixe, "
                          "aproximacao que se mostrou custar loss_signal mensuravel). Grava em "
                          "run_tag com sufixo _sh.")
    ap.add_argument("--sh-loss-high-order-min", type=int, default=4,
                     help="ordem SH minima (par) considerada 'alta' pelo termo angular -- "
                          "mesma semantica de scripts/04_train_rcae.py/04b_train_rrin.py. So "
                          "tem efeito se --angular-loss-weight > 0.")
    ap.add_argument("--sh-loss-lmax-cap", type=int, default=8,
                     help="teto de ordem SH usado no ajuste -- mesma semantica de "
                          "scripts/04_train_rcae.py/04b_train_rrin.py. So tem efeito se "
                          "--angular-loss-weight > 0.")
    ap.add_argument("--sh-loss-q-out", type=int, default=16,
                     help="tamanho do feixe de trincas extra por item usado SO para o termo de "
                          "loss angular/SH (ver utils/rrin_dataset.py, RRINTripletDataset."
                          "sh_q_out -- mesmo flag/default de scripts/04b_train_rrin.py). Default "
                          "16 sustenta ate l_max=4. So tem efeito se --angular-loss-weight > 0.")
    args = ap.parse_args()

    if args.ensemble_m < 1:
        raise ValueError(f"--ensemble-m deve ser >= 1 (recebido {args.ensemble_m})")
    if args.freeze_flow and not args.init_checkpoint:
        sys.exit("--freeze-flow requer --init-checkpoint (nao faz sentido congelar fluxo do "
                  "zero, nao-treinado)")
    if args.max_train_batches is not None or args.max_val_batches is not None:
        print(f"[debug] limitando batches/epoca -- max_train_batches={args.max_train_batches}, "
              f"max_val_batches={args.max_val_batches} (uso pretendido: iteracao rapida de "
              f"codigo, NAO para avaliar convergencia real do modelo)", flush=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, "| job_id:", args.job_id or "(nao informado)")

    entries = load_manifest(args.manifest)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    only_valid = not args.no_only_valid
    print(f"[resumo] only_valid={only_valid}, ensemble_m={args.ensemble_m}")
    sh_q_out = args.sh_loss_q_out if args.angular_loss_weight > 0 else 0
    if args.angular_loss_weight > 0:
        max_l = min(max_order_for_n_directions(sh_q_out), args.sh_loss_lmax_cap)
        print(f"[angular-loss] ATIVO: lambda={args.angular_loss_weight}, "
              f"high_order_min={args.sh_loss_high_order_min}, lmax_cap={args.sh_loss_lmax_cap}, "
              f"sh_q_out={sh_q_out} -> ordem maxima alcancavel = l={max_l} (cada direcao do "
              f"feixe roda com ensemble COMPLETO de ate ensemble_m={args.ensemble_m} pares "
              f"candidatos, ver _sh_bundle_forward_star -- corrigido 2026-09-09, addendum "
              f"33.34, antes era M=1 degenerado)", flush=True)
        if max_l < args.sh_loss_high_order_min:
            print(f"[angular-loss][aviso] --sh-loss-q-out {sh_q_out} so sustenta ate l={max_l} "
                  f"(< --sh-loss-high-order-min {args.sh_loss_high_order_min}) -- este termo vai "
                  f"ser pulado em praticamente todo item. Aumente --sh-loss-q-out, ou reduza "
                  f"--sh-loss-high-order-min.", flush=True)
    else:
        print("[angular-loss] desativado (--angular-loss-weight 0.0, default) -- "
              "loss identica a antes (so MAE de sinal).", flush=True)
    train_ds = RRINTripletDataset(train_entries, args.triplets_dir, args.shell_b, args.n_level,
                                   patch_size=args.patch_size, training=True,
                                   mask_suffix=args.mask_suffix, only_valid=only_valid,
                                   min_tile_coverage=args.min_tile_coverage,
                                   seed=args.seed, max_cached_subjects=args.max_cached_subjects,
                                   ensemble_m=args.ensemble_m, sh_q_out=sh_q_out)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = RRINTripletDataset(val_entries, args.triplets_dir, args.shell_b, args.n_level,
                                 patch_size=args.patch_size, training=False,
                                 mask_suffix=args.mask_suffix, only_valid=only_valid,
                                 min_tile_coverage=args.min_tile_coverage,
                                 seed=args.seed + 1, max_cached_subjects=args.val_max_cached_subjects,
                                 ensemble_m=args.ensemble_m, sh_q_out=sh_q_out)

    persistent_train = args.num_workers > 0
    persistent_val = val_num_workers > 0
    train_sampler = SubjectGroupedSampler(train_ds, seed=args.seed,
                                           freeze_order=args.freeze_subject_order,
                                           num_workers=args.num_workers, batch_size=args.batch_size)
    if args.freeze_subject_order:
        print("[dataloader] ordem dos sujeitos CONGELADA entre epocas (--freeze-subject-order "
              "explicito -- ver utils.dataset.SubjectGroupedSampler.__init__)", flush=True)
    if args.num_workers > 1:
        print(f"[dataloader] SubjectGroupedSampler particionado por worker "
              f"(num_workers={args.num_workers}, batch_size={args.batch_size}) -- "
              f"elimina releitura redundante de sujeito entre workers na mesma epoca "
              f"(ver utils.dataset.SubjectGroupedSampler.__init__, addendum 2026-09-03)", flush=True)
    winit = worker_init_fn if args.num_workers > 0 else None
    winit_val = worker_init_fn if val_num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               num_workers=args.num_workers, drop_last=True,
                               persistent_workers=persistent_train, worker_init_fn=winit)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=val_num_workers,
                             persistent_workers=persistent_val, worker_init_fn=winit_val)

    print(f"[resumo] treino: {len(train_ds.usable)} sujeitos utilizaveis "
          f"({len(train_ds)} patches, {len(train_loader)} batches/epoca)")
    print(f"[resumo] val:    {len(val_ds.usable)} sujeitos utilizaveis "
          f"({len(val_ds)} patches, {len(val_loader)} batches/epoca)", flush=True)

    need_quality = args.weight_quality_cond
    model = build_pairflow_star_model(base_ch=args.base_ch, max_disp=args.max_disp,
                                       norm_type=args.norm_type, freeze_flow=args.freeze_flow,
                                       weight_quality_cond=args.weight_quality_cond).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[resumo] PairFlowStar: {n_params} parametros ({n_trainable} treinaveis, "
          f"freeze_flow={args.freeze_flow}), base_ch={args.base_ch}, ensemble_m={args.ensemble_m}, "
          f"weight_quality_cond={args.weight_quality_cond}, norm_type={args.norm_type})")

    if args.init_checkpoint:
        print(f"[init] carregando flow_net do checkpoint da Etapa 1: {args.init_checkpoint}",
              flush=True)
        ssl_ckpt = torch.load(args.init_checkpoint, map_location=device)
        ssl_args = ssl_ckpt.get("args", {})
        if ssl_args.get("base_ch") is not None and ssl_args["base_ch"] != args.base_ch:
            print(f"[init][aviso] --base-ch ({args.base_ch}) difere do checkpoint da Etapa 1 "
                  f"({ssl_args['base_ch']}) -- load_state_dict provavelmente vai falhar por "
                  f"shape incompativel.", flush=True)
        model.flow_net.load_state_dict(ssl_ckpt["model_state"])
        print(f"[init] flow_net inicializado (checkpoint da Etapa 1: epoca "
              f"{ssl_ckpt.get('epoch')}, val_loss {ssl_ckpt.get('val_loss')})", flush=True)
    else:
        print("[init] --init-checkpoint nao passado -- treinando PairFlowStar do ZERO "
              "(controle, ver docstring do modulo)", flush=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=5)

    print("[sanity] testando 1 batch de treino + 1 de validacao antes do loop de epocas...",
          flush=True)

    def _sanity_step(loader, split_name, do_backward):
        t0 = time.time()
        batch = next(iter(loader))
        vol_a = batch["vol_a_ens"].to(device)
        vol_b = batch["vol_b_ens"].to(device)
        target = batch["target"].to(device)
        bvec_a = batch["bvec_a_ens"].to(device)
        bvec_b = batch["bvec_b_ens"].to(device)
        bvec_t = batch["bvec_t_ens"].to(device)
        t_frac = batch["t_frac_ens"].to(device)
        ensemble_mask = batch["ensemble_mask"].to(device)
        quality = batch["quality_ens"].to(device) if need_quality else None
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            pred = model(vol_a, vol_b, bvec_a, bvec_b, bvec_t, t_frac, ensemble_mask,
                         quality=quality)
            loss = (pred - target).abs().mean()
            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        n_real_mean = ensemble_mask.float().sum(dim=1).mean().item()
        print(f"[sanity] {split_name} OK ({time.time() - t0:.1f}s, loss={loss.item():.6f}, "
              f"pares_reais_media={n_real_mean:.2f}/{args.ensemble_m}, "
              f"sujeitos={sorted(set(batch['subject_tag']))})", flush=True)
        if args.angular_loss_weight > 0:
            t1 = time.time()
            with torch.set_grad_enabled(do_backward):
                pred_sh, target_sh, bvec_t_sh, sh_mask = _sh_bundle_forward_star(
                    model, batch, device, need_quality)
                loss_ang = compute_sh_angular_loss(
                    pred_sh, target_sh, bvec_t_sh, sh_mask,
                    l_max_cap=args.sh_loss_lmax_cap, high_order_min=args.sh_loss_high_order_min)
            print(f"[sanity] {split_name} (feixe SH, ensemble ate M={args.ensemble_m}/direcao) "
                  f"OK ({time.time() - t1:.1f}s, loss_angular={loss_ang.item():.6f}, "
                  f"K={sh_mask.shape[1]}, "
                  f"n_validos_medio={sh_mask.float().sum(dim=1).mean().item():.1f})", flush=True)

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- comecando o loop de epocas de verdade", flush=True)

    # run_tag: mesma disciplina de scripts/04e_train_rrin_star.py -- toda
    # variante que muda o comportamento treinavel precisa de um sufixo
    # proprio.
    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}_pfstar{args.ensemble_m}"
    if args.weight_quality_cond:
        run_tag += "_wqc"
    if not only_valid:
        run_tag += "_inclinv"
    if args.norm_type == "batch":
        run_tag += "_bn"
    if args.angular_loss_weight > 0:
        run_tag += "_sh"
    if args.init_checkpoint:
        run_tag += "_pretrained"
    if args.freeze_flow:
        run_tag += "_frozen"
    if abs(args.lr - 1e-4) > 1e-12:
        run_tag += f"_lr{args.lr:g}"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt -- caminho fixo, "
          f"usado pela etapa 5k)")
    print(f"[resumo] logs deste run em: {run_dir}")

    start_epoch = 1
    best_val = float("inf")
    epochs_no_improve = 0
    resume_ckpt_path = None
    if not args.no_resume:
        if args.resume_checkpoint:
            resume_ckpt_path = Path(args.resume_checkpoint)
        elif (out_dir / "last.pt").exists():
            resume_ckpt_path = out_dir / "last.pt"

    if resume_ckpt_path is not None:
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"--resume-checkpoint {resume_ckpt_path} nao existe")
        print(f"[resume] carregando checkpoint existente: {resume_ckpt_path}", flush=True)
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        old_args = ckpt.get("args", {})
        for key in ("shell_b", "n_level", "patch_size", "base_ch", "max_disp",
                    "weight_quality_cond", "ensemble_m", "norm_type", "freeze_flow", "lr",
                    "angular_loss_weight", "sh_loss_high_order_min", "sh_loss_lmax_cap",
                    "sh_loss_q_out"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional.",
                      flush=True)
        old_norm_type = old_args.get("norm_type", "instance")
        if old_norm_type != args.norm_type:
            raise ValueError(
                f"--norm-type={args.norm_type} nao bate com o checkpoint ({old_norm_type}) -- "
                f"norm_type nao e retomavel entre variantes. Use --no-resume ou um --out-dir/"
                f"--norm-type novos para treinar a variante '{args.norm_type}' do zero.")
        # ATENCAO (mesmo comentario de 04h_train_pairflow_finetune.py): retomar via last.pt
        # SOBRESCREVE o que --init-checkpoint teria carregado -- comportamento correto, o run
        # ja em andamento ja incorporou o pre-treino na primeira epoca.
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(f"[resume] retomando da epoca {start_epoch} (best_val={best_val:.6f}, "
              f"epochs_no_improve={epochs_no_improve})", flush=True)
        if start_epoch > args.epochs:
            print(f"[resume] epoca de retomada ({start_epoch}) ja passa de --epochs "
                  f"({args.epochs}) -- nada a fazer.", flush=True)
    else:
        print("[resume] nenhum checkpoint anterior encontrado (ou --no-resume) -- "
              "comecando do zero (ou do --init-checkpoint, se passado).", flush=True)

    warmup_state = None
    if args.warmup_steps > 0:
        if resume_ckpt_path is not None:
            print(f"[warmup] --warmup-steps={args.warmup_steps} ignorado -- retomando de "
                  f"checkpoint existente (warmup so' faz sentido comecando do zero).", flush=True)
        else:
            warmup_state = {"step": 0, "warmup_steps": args.warmup_steps,
                             "base_lr": args.lr, "start_factor": 0.1}
            print(f"[warmup] ativado: LR sobe linearmente de {0.1 * args.lr:.2e} ate "
                  f"{args.lr:.2e} ao longo dos primeiros {args.warmup_steps} passos de "
                  f"otimizador, antes do ReduceLROnPlateau assumir.", flush=True)

    # NOTA (bug corrigido em 2026-09-04, ver addendum secao 33.21): o zero-init
    # PRECISA ser aplicado (ou pulado, com aviso) so' DEPOIS de sabermos se
    # havera' resume -- se aplicado antes (junto da criacao do modelo) e'
    # silenciosamente sobrescrito pelo model.load_state_dict(ckpt["model_state"])
    # do bloco de resume acima, sem nenhum aviso, dando a falsa impressao (via
    # a mensagem "[init] ... ativado") de que o zero-init teve efeito quando na
    # verdade os pesos resumidos (nao-zerados) prevaleceram.
    if args.zero_init_refine_output:
        if resume_ckpt_path is not None:
            print("[init] --zero-init-refine-output ignorado -- retomando de checkpoint "
                  "existente (os pesos resumidos do refine_net prevalecem; zero-init so' "
                  "faz sentido comecando do zero).", flush=True)
        else:
            with torch.no_grad():
                model.refine_net.net[-1].weight.zero_()
                model.refine_net.net[-1].bias.zero_()
            print("[init] --zero-init-refine-output ativado: ultima camada de RefineNet3D "
                  "(model.refine_net.net[-1]) zerada -- residual_f=0 no inicio, entao a "
                  "predicao inicial (pred = blend + residual) comeca EXATAMENTE igual ao "
                  "blend da interpolacao por fluxo optico.", flush=True)

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w") as f:
        # train_loss_signal/train_loss_angular/val_loss_signal/val_loss_angular:
        # colunas novas no FIM (mesma convencao de 04_train_rcae.py/
        # 04f_train_implicit.py, addendum secao 33.29) -- vazias quando
        # --angular-loss-weight=0.0.
        f.write("epoch,train_loss,val_loss,lr,"
                "train_loss_signal,train_loss_angular,"
                "val_loss_signal,val_loss_angular\n")
    batch_log_path = run_dir / "batch_log.csv"
    batch_log_f = open(batch_log_path, "w")
    batch_log_f.write("epoch,split,batch,loss,wait_s,compute_s,subject_tags,n_pares_reais_media,"
                       "loss_signal,loss_angular\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss, train_loss_signal, train_loss_angular = run_epoch(
                model, train_loader, optimizer, device, train=True,
                epoch=epoch, need_quality=need_quality,
                batch_log_f=batch_log_f,
                max_batches=args.max_train_batches,
                warmup_state=warmup_state,
                angular_loss_weight=args.angular_loss_weight,
                sh_loss_high_order_min=args.sh_loss_high_order_min,
                sh_loss_lmax_cap=args.sh_loss_lmax_cap)
            val_loss, val_loss_signal, val_loss_angular = run_epoch(
                model, val_loader, optimizer, device, train=False,
                epoch=epoch, need_quality=need_quality,
                batch_log_f=batch_log_f,
                max_batches=args.max_val_batches,
                angular_loss_weight=args.angular_loss_weight,
                sh_loss_high_order_min=args.sh_loss_high_order_min,
                sh_loss_lmax_cap=args.sh_loss_lmax_cap)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss_angular_str = f"{train_loss_angular:.6f}" if train_loss_angular is not None else ""
            val_loss_angular_str = f"{val_loss_angular:.6f}" if val_loss_angular is not None else ""
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{current_lr:.2e},"
                        f"{train_loss_signal:.6f},{train_loss_angular_str},"
                        f"{val_loss_signal:.6f},{val_loss_angular_str}\n")
            if args.angular_loss_weight > 0:
                print(f"epoch {epoch:03d} | train {train_loss:.6f} "
                      f"(mae {train_loss_signal:.6f}, ang {train_loss_angular:.6f}) | "
                      f"val {val_loss:.6f} (mae {val_loss_signal:.6f}, ang {val_loss_angular:.6f}) | "
                      f"lr {current_lr:.2e}")
            else:
                print(f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | "
                      f"lr {current_lr:.2e}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "args": vars(args),
                    "epoch": epoch, "val_loss": val_loss, "best_val": best_val,
                    "epochs_no_improve": epochs_no_improve,
                }, out_dir / "best.pt")
                shutil.copy2(out_dir / "best.pt", run_dir / "best.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping na epoca {epoch} (sem melhora ha {args.patience} epocas)")
                    break

            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "args": vars(args), "epoch": epoch,
                        "val_loss": val_loss, "best_val": best_val,
                        "epochs_no_improve": epochs_no_improve}, out_dir / "last.pt")
            shutil.copy2(out_dir / "last.pt", run_dir / "last.pt")
    finally:
        batch_log_f.close()

    print("Treino concluido. Melhor val_loss:", best_val, "-> checkpoint em", out_dir / "best.pt")
    print(f"Copia permanente deste run em: {run_dir / 'best.pt'} (job_id={run_id})")


if __name__ == "__main__":
    main()