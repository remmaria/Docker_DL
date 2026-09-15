#!/usr/bin/env python3
"""
Etapa 4f: treina o modelo de representacao angular IMPLICITA (NeRF/LIIF-
style, ver model/implicit_angular.py e addendum secao 20.11) para um
(shell, nivel de subamostragem) especifico. Repita a chamada para cada
combinacao que quiser cobrir.

Linha NOVA e INDEPENDENTE do RCAE (scripts/04_train_rcae.py) e da familia
RRIN/AMT/HFD/estrela (scripts/04b/04c/04d/04e) -- nao ha correspondencia
par-a-par nenhuma aqui: o dataset (utils/dataset.py:DWIPatchDataset, o MESMO
usado pelo RCAE -- e uma utilidade GENERICA de carregamento de patch, nao
codigo especifico do RCAE, ver docstring do modulo) entrega TODAS as
n_level direcoes medidas de uma vez (nao pares), e o modelo consulta
qualquer direcao-alvo continua (ver model/implicit_angular.py).

Uso:
    python scripts/04f_train_implicit.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --shell-b 1000 --n-level 16 \
        --out-dir work_dir/implicit_checkpoints \
        --epochs 100 --batch-size 4 --patch-size 10 --lr 1e-3

Requer PyTorch + GPU. Nao executado neste ambiente de desenvolvimento (sem
torch instalado); revisar/ajustar hiperparametros no cluster.

Iteracao rapida (pedido explicito da usuaria em 2026-09-03, "dá pra fazer
este mini treino pro implicit tb? testando mean e attention?" -- ver
addendum secao 33 em diante): combine um manifesto mini
(scripts/99_make_mini_manifest.py --filter-scheme --scheme-dir ...) com
--freeze-subject-order (ADITIVO, default desligado) e opcionalmente
--max-train-batches/--max-val-batches (DEBUG, default None) pra comparar
--aggregation mean vs attention em poucos minutos, mesmo espirito do
mini-teste ja usado na linha PairFlowStar (scripts/04i_train_pairflow_
star.py) -- ver slurm/04f_minitest_agg.sh.

Start do treino (pedido explicito da usuaria em 2026-09-03, "o implicit
começa com val loss bem alta... se pudesse melhorar alguma coisa"):
--init-output-bias-from-data (ADITIVO, default desligado) inicializa o
bias da ultima camada do decoder com a media do sinal-alvo (o sinal e
normalizado por percentil, nao centrado em zero -- boa parte do loss
inicial alto e so esse descompasso de nivel constante). --warmup-steps N
(ADITIVO, default 0) aquece a LR linearmente nos primeiros N passos de
otimizador antes do ReduceLROnPlateau assumir -- ver addendum
2026-09-03.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.manifest import load_manifest
from utils.dataset import (
    DWIPatchDataset, collate_variable_targets, SubjectGroupedSampler, worker_init_fn,
)
from utils.viz import save_patch_debug_png
from utils.sh_basis import max_order_for_n_directions
from utils.sh_angular_loss import n_coeffs_even, compute_sh_angular_loss
from model.implicit_angular import build_implicit_model


def _tensor_stats(x: torch.Tensor, outlier_threshold: float = None) -> tuple:
    """Identico a scripts/04_train_rcae.py:_tensor_stats -- pequeno o
    bastante para duplicar aqui em vez de importar entre scripts numerados
    (convencao do projeto: scripts/0X_*.py nao se importam entre si, so
    utils/ e compartilhado livremente)."""
    if x.numel() == 0:
        stats = (float("nan"), float("nan"), float("nan"), float("nan"))
        return stats + (0,) if outlier_threshold is not None else stats
    stats = (x.mean().item(), x.std().item(), x.min().item(), x.max().item())
    if outlier_threshold is not None:
        n_outliers = int((x.abs() > outlier_threshold).sum().item())
        return stats + (n_outliers,)
    return stats


def run_epoch(model, loader, optimizer, device, train: bool, epoch: int, batch_log_f=None,
              debug_state=None, outlier_threshold: float = 3.0, batch_log_every: int = 5,
              max_batches: int = None, warmup_state: dict = None,
              angular_loss_weight: float = 0.0, sh_loss_high_order_min: int = 4,
              sh_loss_lmax_cap: int = 8):
    """MESMO espirito de scripts/04_train_rcae.py:run_epoch (mask MAE sobre
    target_mask, log por batch amostrado, snapshot de debug opcional) --
    adaptado para model.encode()/model.decode() (metodos, nao submodulos
    .encoder/.decoder como no RCAE, ver model/implicit_angular.py).

    angular_loss_weight/sh_loss_high_order_min/sh_loss_lmax_cap: termo de
    loss opcional no dominio angular/SH (porte de scripts/04_train_rcae.py,
    ver protocolo secao 9 -- pedido explicito da usuaria em 2026-09-09,
    "como colocar angular loss neles?"). O `implicit` e' a linha MAIS
    naturalmente adequada das tres (RCAE/RRIN/implicit) pra esse termo:
    target_vols/target_bvecs/target_mask aqui ja tem EXATAMENTE o mesmo
    formato (B, N_out, ..., D,H,W)/(B, N_out, 3)/(B, N_out) que
    compute_sh_angular_loss espera, sem precisar de nenhum mecanismo de
    "feixe"/bundle adicional (ao contrario do RRIN, que precisa amostrar um
    feixe extra de trincas -- ver scripts/04b_train_rrin.py:
    _sh_bundle_forward).

    CORRECAO (2026-09-09, ver addendum secao 33.33 -- a versao anterior
    deste docstring/do aviso de startup dizia erroneamente que N_out em
    TREINO nao era limitado por --q-out; estava ERRADA): confirmado em
    utils/dataset.py:DWIPatchDataset._dynamic_split, `n_take = min(
    self.n_level + self.q_out, n_avail)` -- ou seja, --q-out e' o teto de
    N_out tanto em TREINO quanto em VALIDACAO, sempre (so' a ESCOLHA de
    quais direcoes formam o alvo e' re-amostrada a cada item em treino, a
    CONTAGEM maxima e' sempre --q-out). Com o default --q-out=10, so' ha'
    direcoes-alvo suficientes pra sustentar ate' l_max=2
    (n_coeffs_even(2)=6<=10; l=4 precisaria de 15) -- com
    --sh-loss-high-order-min=4 (default) o termo fica ZERADO em TODO
    batch, treino e validacao (mesma limitacao ja documentada pro RCAE,
    ver protocolo secao 9). Pra ter o termo com efeito real, ou baixe
    --sh-loss-high-order-min pra 2, ou suba --q-out (>=15 pra l=4)."""
    model.train(mode=train)
    total_loss = 0.0
    # totais separados de loss_signal (MAE de sinal) e loss_angular --
    # mesmo padrao adotado em scripts/04_train_rcae.py (addendum secao
    # 33.29) pra poder ver se o termo angular esta baixando de verdade.
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
        subject_tags = batch["subject_tags"]
        input_vols = batch["input_vols"].to(device)
        input_bvecs = batch["input_bvecs"].to(device)
        target_vols = batch["target_vols"].to(device)
        target_bvecs = batch["target_bvecs"].to(device)
        target_mask = batch["target_mask"].to(device)
        mask = target_mask[:, :, None, None, None, None].expand_as(target_vols).float()

        with torch.set_grad_enabled(train):
            # encode/decode chamados separado (em vez de model(...) direto)
            # -- MESMO calculo, ImplicitAngularModel3D.forward faz
            # exatamente isso por dentro, sem custo extra -- so pra ficar
            # com "state" a mao pro snapshot de debug (mesmo padrao de
            # scripts/04_train_rcae.py).
            state = model.encode(input_vols, input_bvecs)
            pred = model.decode(state, target_bvecs)
            err = (pred - target_vols).abs()  # MAE, mesmo motivo do RCAE (ver 04_train_rcae.py)
            loss_signal = (err * mask).sum() / mask.sum().clamp(min=1.0)
            # termo angular/SH opcional -- desativado por padrao
            # (angular_loss_weight=0.0), loss identica a antes. target_vols/
            # target_bvecs/target_mask ja estao no formato (B,N_out,...)/
            # (B,N_out,3)/(B,N_out) que compute_sh_angular_loss espera, sem
            # nenhum bundle extra (ver docstring de run_epoch acima).
            if angular_loss_weight > 0:
                loss_angular = compute_sh_angular_loss(
                    pred, target_vols, target_bvecs, target_mask,
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

        t_compute_end = time.time()
        compute_s = t_compute_end - t_received

        total_loss += loss.item()
        total_loss_signal += loss_signal.item()
        if loss_angular is not None:
            total_loss_angular += loss_angular.item()
        n_batches += 1
        n_samples += input_vols.shape[0]
        total_wait_s += wait_s
        total_compute_s += compute_s

        if batch_log_f is not None and (n_batches == 1 or n_batches % max(1, batch_log_every) == 0):
            in_mean, in_std, in_min, in_max, in_n_out = _tensor_stats(
                input_vols, outlier_threshold=outlier_threshold)
            valid_target = target_vols[mask.bool()]
            tg_mean, tg_std, tg_min, tg_max, tg_n_out = _tensor_stats(
                valid_target, outlier_threshold=outlier_threshold)
            tags_str = ";".join(subject_tags)
            # loss_signal/loss_angular no fim (nao mudam a posicao das
            # colunas antigas) -- loss_angular fica vazia quando o termo
            # esta desativado, mesma convencao de 04_train_rcae.py.
            loss_angular_str = f"{loss_angular.item():.6f}" if loss_angular is not None else ""
            batch_log_f.write(
                f"{epoch},{split},{n_batches},{loss.item():.6f},"
                f"{in_mean:.4f},{in_std:.4f},{in_min:.4f},{in_max:.4f},{in_n_out},"
                f"{tg_mean:.4f},{tg_std:.4f},{tg_min:.4f},{tg_max:.4f},{tg_n_out},"
                f"{wait_s:.3f},{compute_s:.3f},{tags_str},"
                f"{loss_signal.item():.6f},{loss_angular_str}\n"
            )
            batch_log_f.flush()

        if debug_state is not None and train:
            debug_state["step"] += 1
            step = debug_state["step"]
            every = debug_state["every"]
            if every > 0 and (step == 1 or step % every == 0):
                png_path = debug_state["dir"] / f"step_{step:06d}_epoch{epoch:04d}_batch{n_batches:04d}.png"
                subj0 = subject_tags[0] if subject_tags else "?"
                save_patch_debug_png(
                    png_path, input_vols[0], target_vols[0], pred_vols=pred[0].detach(),
                    context=state[0].detach(), max_dirs=debug_state["max_dirs"],
                    title=f"step {step} | epoca {epoch} batch {n_batches} | {subj0} | loss {loss.item():.6f}",
                )
                print(f"[debug] snapshot (batch) salvo em {png_path}", flush=True)

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


def plot_fixed_debug_patch(model, fixed_batch, device, plot_dir, epoch, val_loss=None,
                            shell_b=None, n_level=None, max_dirs=6):
    """MESMO papel de scripts/04_train_rcae.py:plot_fixed_debug_patch --
    roda o patch fixo de validacao (eval, no_grad) e salva o snapshot."""
    model.eval()
    with torch.no_grad():
        state = model.encode(fixed_batch["input_vols"].to(device),
                              fixed_batch["input_bvecs"].to(device))
        target_bvecs = fixed_batch["target_bvecs"].to(device)
        pred = model.decode(state, target_bvecs)
    loss_str = f" | val_loss {val_loss:.6f}" if val_loss is not None else " | baseline (sem treino)"
    png_path = plot_dir / f"epoch_{epoch:04d}.png"
    save_patch_debug_png(
        png_path, fixed_batch["input_vols"][0], fixed_batch["target_vols"][0],
        pred_vols=pred[0], context=state[0], max_dirs=max_dirs,
        title=f"shell={shell_b} n={n_level} | epoca {epoch}{loss_str}",
    )
    print(f"[debug] snapshot da predicao (patch fixo) salvo em {png_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scheme-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shell-b", type=float, required=True)
    ap.add_argument("--n-level", type=int, required=True)
    ap.add_argument("--patch-size", type=int, default=10,
                     help="tamanho do patch cubico (default 10, mesmo default do RCAE/RRIN).")
    ap.add_argument("--q-out", type=int, default=10,
                     help="numero fixo de direcoes-alvo por exemplo em VALIDACAO (default 10 -- "
                          "ver utils/dataset.py:DWIPatchDataset. Em TREINO o split e "
                          "re-amostrado a cada exemplo, ver --training/_dynamic_split; este "
                          "flag so limita o N_out MAXIMO por exemplo). Ao contrario do RCAE "
                          "(paper fixa N_out=10 por ser o valor usado no artigo), aqui N_out "
                          "e livre (o decoder implicito aceita qualquer N_out, ver "
                          "model/implicit_angular.py) -- 10 e so um default razoavel de custo.")
    ap.add_argument("--l-max", type=int, default=None,
                     help="ordem par maxima da base SH usada para codificar direcoes de "
                          "entrada e alvo (ver model/implicit_angular.py:sh_positional_encoding). "
                          "Default None = automatico, utils.sh_basis.max_order_for_n_directions"
                          "(n_level) -- amarra a resolucao angular da representacao a quantas "
                          "direcoes sao realmente medidas, mesma convencao do baseline_sh. FIXO "
                          "para o checkpoint inteiro (muda o numero de canais da rede -- nao da "
                          "pra mudar em --resume-checkpoint, ver checagem abaixo).")
    ap.add_argument("--base-ch", type=int, default=16,
                     help="largura (numero de canais) dos blocos conv de PerDirectionEncoder3D/"
                          "SpatialTrunk3D/ImplicitDecoderHead3D (default 16, mesmo default de "
                          "FlowNet3D em model/rrin3d.py).")
    ap.add_argument("--norm-type", choices=["instance", "batch"], default="instance",
                     help="'instance' (default, ver model/rrin3d.py:_norm3d para a discussao "
                          "completa do artefato de costura entre patches na reconstrucao por "
                          "sliding-window) ou 'batch' (resolve a costura, exige treinar do "
                          "ZERO). Mesma semantica de --norm-type em scripts/04b_train_rrin.py.")
    ap.add_argument("--aggregation", choices=["mean", "attention"], default="mean",
                     help="'mean' (default, comportamento ORIGINAL desta linha -- media simples "
                          "entre as n_level direcoes de entrada) ou 'attention' (ADITIVO, ver "
                          "model/implicit_angular.py:AttentionAggregator3D e addendum "
                          "2026-09-03 secao 29 -- media PONDERADA, com pesos aprendidos POR "
                          "VOXEL, motivada pelo diagnostico de "
                          "scripts/14_diagnose_implicit_pooling.py). Muda o shape dos pesos "
                          "(adiciona a cabeca de atencao) -- BLOQUEANTE para --resume-checkpoint "
                          "entre valores diferentes, mesma logica de --base-ch/--norm-type "
                          "(ver checagem abaixo); ganha seu proprio run_tag automaticamente, "
                          "entao um treino novo com --aggregation attention nunca colide com "
                          "checkpoints antigos de --aggregation mean no mesmo --out-dir.")
    ap.add_argument("--cross-direction-attention", action="store_true",
                     help="ADITIVO, default DESLIGADO (item 2 do addendum 2026-09-13, ver "
                          "model/implicit_angular.py:CrossDirectionAttention3D) -- insere um "
                          "bloco de self-attention ENTRE as n_level direcoes de entrada logo "
                          "apos PerDirectionEncoder3D e ANTES de qualquer agregacao (mean OU "
                          "attention -- as duas etapas sao independentes, esta flag compoe com "
                          "--aggregation qualquer). Motivado por producao real saturando em "
                          "val_loss~0,041-0,042 ja na epoca 3 mesmo com --aggregation attention "
                          "(o teto parece ser falta de interacao ENTRE direcoes, nao 'que tipo "
                          "de pooling'). Adiciona parametros novos -- BLOQUEANTE para "
                          "--resume-checkpoint, mesma logica de --aggregation; ganha seu proprio "
                          "run_tag (_xattn) automaticamente, nunca colide com checkpoints sem "
                          "esta flag.")
    ap.add_argument("--cross-attn-heads", type=int, default=4,
                     help="numero de cabecas de CrossDirectionAttention3D (default 4). So tem "
                          "efeito com --cross-direction-attention. PRECISA dividir --base-ch "
                          "sem resto (senao a construcao do modelo falha cedo com um erro "
                          "claro).")
    ap.add_argument("--decoder-base-ch", type=int, default=None,
                     help="NOVO (2026-09-14, ver addendum de capacidade): largura do "
                          "ImplicitDecoderHead3D DESACOPLADA do resto. Default None = segue "
                          "--base-ch (comportamento historico). Motivacao: o decoder e' a rede "
                          "que representa a funcao continua sobre a esfera -- o PROPOSITO do "
                          "modelo implicito -- e com base_ch=16 ele tem 13.873 parametros e 2 "
                          "convolucoes, contra 3.723.497 e 10 convolucoes do decoder do RCAE "
                          "(268x). MUDA O SHAPE DOS PESOS -- bloqueante para resume. Sufixo "
                          "_dbc<N> no run_tag.")
    ap.add_argument("--decoder-depth", type=int, default=1,
                     help="NOVO: numero de camadas ocultas do ImplicitDecoderHead3D (default 1 = "
                          "topologia historica, byte-a-byte). MUDA O SHAPE DOS PESOS -- "
                          "bloqueante para resume. Sufixo _dd<N> no run_tag.")
    ap.add_argument("--decoder-reinject-code", action="store_true",
                     help="NOVO: reinjeta o codigo SH da direcao-alvo na entrada de CADA camada "
                          "oculta do decoder, em vez de so' na primeira. Hoje a direcao-alvo (a "
                          "'consulta' do modelo implicito) entra uma unica vez e precisa "
                          "sobreviver a rede inteira a partir dali. O RCAE reinjeta o bvec em "
                          "TODOS os estagios do decoder (RepeatBVector), e NeRF/LIIF reinjetam a "
                          "coordenada ao longo do MLP -- e' a pratica padrao para representacoes "
                          "implicitas. So' tem efeito com --decoder-depth > 1. MUDA O SHAPE DOS "
                          "PESOS -- bloqueante para resume. Sufixo _dreinj no run_tag.")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2)
    ap.add_argument("--val-num-workers", type=int, default=None)
    ap.add_argument("--val-max-cached-subjects", type=int, default=1)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-id", default="")
    ap.add_argument("--debug-plot-every", type=int, default=0)
    ap.add_argument("--debug-plot-every-batches", type=int, default=0)
    ap.add_argument("--debug-max-dirs", type=int, default=0)
    ap.add_argument("--outlier-threshold", type=float, default=3.0)
    ap.add_argument("--batch-log-every", type=int, default=5)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--resume-checkpoint", default=None)
    ap.add_argument("--freeze-subject-order", action="store_true",
                     help="EXPERIMENTAL (mesmo flag de scripts/04g_train_pairflow_ssl.py e "
                          "scripts/04i_train_pairflow_star.py), default DESLIGADO -- ativa "
                          "SubjectGroupedSampler(freeze_order=True): ordem dos sujeitos fixa "
                          "entre epocas, combinado com o particionamento por worker ja ativo "
                          "aqui via num_workers/batch_size -- ver addendum secoes 30/33.1/33.4 "
                          "(medido no PairFlowStar; ainda nao medido nesta linha `implicit`, que "
                          "foi onde o gargalo de 93-99% de wait foi originalmente diagnosticado, "
                          "seção 30).")
    ap.add_argument("--max-train-batches", type=int, default=None,
                     help="DEBUG (mesmo espirito de scripts/04i_train_pairflow_star.py): limita "
                          "quantos batches de treino sao processados por epoca, independente do "
                          "tamanho real do dataset -- pra iterar rapido em mudancas de codigo "
                          "(ex.: --aggregation attention) sem esperar uma epoca inteira. Default "
                          "None = sem limite.")
    ap.add_argument("--max-val-batches", type=int, default=None,
                     help="DEBUG: mesmo espirito de --max-train-batches, para a epoca de "
                          "validacao. Default None = sem limite.")
    ap.add_argument("--init-output-bias-from-data", action="store_true",
                     help="ADITIVO (default DESLIGADO, pedido explicito da usuaria em "
                          "2026-09-03 -- 'implicit começa com val loss bem alta'). Antes do "
                          "loop de epocas, estima a MEDIA do sinal-alvo (mascarado por "
                          "target_mask) sobre alguns batches de treino e inicializa o bias da "
                          "ULTIMA camada de ImplicitDecoderHead3D (model.decoder_head.net[-1]) "
                          "com esse valor, em vez do bias default do PyTorch (~0). Como o sinal "
                          "e normalizado por percentil (nao e centrado em zero, ver "
                          "utils/dataset.py), a rede comeca 'chutando' perto de zero enquanto o "
                          "alvo real tem media bem diferente -- boa parte do loss inicial alto e "
                          "so esse descompasso de nivel constante, nao erro estrutural. So muda "
                          "a INICIALIZACAO (nao ha bloqueio de --resume-checkpoint -- um "
                          "checkpoint existente sobrescreve esse bias normalmente via "
                          "load_state_dict).")
    ap.add_argument("--warmup-steps", type=int, default=0,
                     help="ADITIVO (default 0 = DESLIGADO, mesmo pedido de 2026-09-03). Numero "
                          "de passos de OTIMIZADOR (nao epocas) de aquecimento linear de LR no "
                          "inicio do treino: a LR sobe linearmente de "
                          "0.1*--lr ate --lr ao longo desses passos, antes do "
                          "ReduceLROnPlateau assumir o controle normalmente. Nao muda o valor "
                          "do loss NA PRIMEIRA iteracao, mas costuma suavizar a descida inicial "
                          "(evita passos grandes demais antes da rede ter uma nocao inicial do "
                          "problema) -- mesma motivacao da 'bolha' de instabilidade vista em LR "
                          "alta nos mini-testes de BATCH_SIZE (ver addendum secao 33.14/33.17). "
                          "So' ativado quando comecando do ZERO -- ignorado (com aviso) ao "
                          "retomar de um checkpoint existente, ja que o treino retomado nao "
                          "deveria reaquecer a LR de um modelo ja parcialmente treinado.")
    ap.add_argument("--angular-loss-weight", type=float, default=0.0,
                     help="lambda do termo de loss opcional no dominio angular/SH (porte de "
                          "scripts/04_train_rcae.py, ver protocolo secao 9 -- pedido explicito "
                          "da usuaria em 2026-09-09). soma lambda*erro_SH_ordem_alta a MAE do "
                          "sinal bruto (loss = loss_signal + lambda*loss_angular). Default 0.0 "
                          "= DESATIVADO, comportamento identico ao treino sem esse termo. "
                          "--q-out e' o teto de N_out TANTO em TREINO quanto em VALIDACAO (ver "
                          "utils/dataset.py:_dynamic_split, n_take limitado por --q-out sempre; "
                          "a re-amostragem em treino so' muda QUAIS direcoes formam o alvo, nao "
                          "a CONTAGEM maxima) -- com o default --q-out=10, so' sustenta ate "
                          "l_max=2 (n_coeffs_even(2)=6<=10), entao com "
                          "--sh-loss-high-order-min=4 (default) este termo fica ZERADO em TODO "
                          "batch, treino e validacao. Baixe --sh-loss-high-order-min pra 2, ou "
                          "suba --q-out (>=15 pra l=4), pra ter o termo com efeito real; ver "
                          "aviso no [angular-loss] de startup.")
    ap.add_argument("--sh-loss-high-order-min", type=int, default=4,
                     help="grau l MINIMO (par) considerado 'ordem alta' no termo angular -- "
                          "mesma semantica de scripts/04_train_rcae.py. Default 4. So tem "
                          "efeito se --angular-loss-weight > 0.")
    ap.add_argument("--sh-loss-lmax-cap", type=int, default=8,
                     help="teto de l_max tentado por item no termo angular -- mesma semantica "
                          "de scripts/04_train_rcae.py. Default 8. So tem efeito se "
                          "--angular-loss-weight > 0.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, "| job_id:", args.job_id or "(nao informado)")
    debug_max_dirs = args.debug_max_dirs if args.debug_max_dirs > 0 else max(args.n_level, args.q_out)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    if args.angular_loss_weight > 0:
        max_l_val = min(max_order_for_n_directions(args.q_out), args.sh_loss_lmax_cap)
        print(f"[angular-loss] ATIVO: lambda={args.angular_loss_weight}, "
              f"high_order_min={args.sh_loss_high_order_min}, lmax_cap={args.sh_loss_lmax_cap}", flush=True)
        if max_l_val < args.sh_loss_high_order_min:
            print(f"[angular-loss][aviso] --q-out {args.q_out} so sustenta ate l={max_l_val} "
                  f"(precisaria de {n_coeffs_even(args.sh_loss_high_order_min)} direcoes-alvo "
                  f"pra chegar em l={args.sh_loss_high_order_min}) -- --q-out e' o teto de N_out "
                  f"TANTO em treino QUANTO em validacao (ver utils/dataset.py:_dynamic_split), "
                  f"entao este termo vai ficar ZERADO em TODO batch (train_loss_angular/"
                  f"val_loss_angular=0.000000 no train_log.csv), nao so' na validacao. Aumente "
                  f"--q-out (>=15 pra l=4) se quiser o termo com efeito real, ou reduza "
                  f"--sh-loss-high-order-min (2 e' o maximo sustentavel com --q-out 10).", flush=True)
    else:
        print("[angular-loss] desativado (--angular-loss-weight 0.0, default) -- "
              "treino identico ao MAE puro sobre o sinal.", flush=True)

    entries = load_manifest(args.manifest)
    train_entries = [e for e in entries if e.split == "train"]
    val_entries = [e for e in entries if e.split == "val"]

    train_ds = DWIPatchDataset(train_entries, args.scheme_dir, args.shell_b, args.n_level,
                                patch_size=args.patch_size, q_out=args.q_out, training=True,
                                mask_suffix=args.mask_suffix,
                                min_tile_coverage=args.min_tile_coverage,
                                seed=args.seed, max_cached_subjects=args.max_cached_subjects)
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = DWIPatchDataset(val_entries, args.scheme_dir, args.shell_b, args.n_level,
                              patch_size=args.patch_size, q_out=args.q_out, training=False,
                              mask_suffix=args.mask_suffix,
                              min_tile_coverage=args.min_tile_coverage,
                              seed=args.seed + 1, max_cached_subjects=args.val_max_cached_subjects)

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
                               collate_fn=collate_variable_targets,
                               persistent_workers=persistent_train, worker_init_fn=winit)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=val_num_workers,
                             collate_fn=collate_variable_targets,
                             persistent_workers=persistent_val, worker_init_fn=winit_val)

    print(f"[resumo] treino: {len(train_ds.usable)} sujeitos utilizaveis "
          f"({len(train_ds)} patches, {len(train_loader)} batches/epoca)")
    print(f"[resumo] val:    {len(val_ds.usable)} sujeitos utilizaveis "
          f"({len(val_ds)} patches, {len(val_loader)} batches/epoca)")

    debug_fixed_batch = None
    if args.debug_plot_every > 0:
        best_idx = int(np.argmax(val_ds.tile_coverage))
        best_si, best_origin = val_ds.tile_index[best_idx]
        best_entry, best_tag = val_ds.usable[best_si]
        print(f"[resumo] patch fixo de debug: sujeito={best_tag} origem={best_origin} "
              f"cobertura_mascara={val_ds.tile_coverage[best_idx]:.3f}", flush=True)
        debug_fixed_batch = collate_variable_targets([val_ds[best_idx]])

    model = build_implicit_model(n_level=args.n_level, l_max=args.l_max, base_ch=args.base_ch,
                                  norm_type=args.norm_type, aggregation=args.aggregation,
                                  cross_direction_attention=args.cross_direction_attention,
                                  cross_attn_heads=args.cross_attn_heads,
                                  decoder_base_ch=args.decoder_base_ch,
                                  decoder_depth=args.decoder_depth,
                                  decoder_reinject_code=args.decoder_reinject_code).to(device)
    print(f"[modelo] n_level={args.n_level} l_max={model.l_max} (sh_dim={model.sh_dim}) "
          f"base_ch={args.base_ch} norm_type={args.norm_type} aggregation={args.aggregation} "
          f"cross_direction_attention={args.cross_direction_attention} "
          f"(cross_attn_heads={args.cross_attn_heads}) "
          f"decoder_base_ch={model.decoder_base_ch} decoder_depth={args.decoder_depth} "
          f"decoder_reinject_code={args.decoder_reinject_code} -- "
          f"{sum(p.numel() for p in model.parameters())} parametros")
    # quebra por submodulo + alerta do gargalo de informacao do `state` (ver
    # addendum de capacidade 2026-09-14): o `state` que sai do encoder tem
    # base_ch canais e precisa carregar o perfil angular INTEIRO do voxel,
    # porque o decoder consulta esse mesmo state para QUALQUER direcao-alvo.
    # Se base_ch < sh_dim, esse perfil nao cabe no estado nem em principio.
    print(f"[modelo] parametros por submodulo: "
          f"per_dir_encoder={sum(p.numel() for p in model.per_dir_encoder.parameters())}, "
          f"trunk={sum(p.numel() for p in model.trunk.parameters())}, "
          f"decoder_head={sum(p.numel() for p in model.decoder_head.parameters())}")
    if args.base_ch < model.sh_dim:
        print(f"[modelo][AVISO] base_ch={args.base_ch} e MENOR que sh_dim={model.sh_dim} "
              f"(l_max={model.l_max}): o 'state' de {args.base_ch} canais nao tem graus de "
              f"liberdade suficientes para representar um perfil angular de ordem "
              f"{model.l_max}, que precisa de {model.sh_dim} coeficientes. Isso e um gargalo "
              f"de INFORMACAO, nao so de capacidade -- considere --base-ch >= "
              f"{2 * model.sh_dim}.", flush=True)
    elif args.base_ch < 2 * model.sh_dim:
        print(f"[modelo][nota] base_ch={args.base_ch} esta no limite de sh_dim={model.sh_dim} "
              f"(l_max={model.l_max}) -- o 'state' mal comporta o perfil angular, sem folga "
              f"para tambem codificar contexto espacial. Ver addendum de capacidade "
              f"2026-09-14.", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=5)

    print("[sanity] testando 1 batch de treino + 1 de validacao antes do loop de epocas...",
          flush=True)

    def _sanity_step(loader, split_name, do_backward):
        t0 = time.time()
        batch = next(iter(loader))
        input_vols = batch["input_vols"].to(device)
        input_bvecs = batch["input_bvecs"].to(device)
        target_vols = batch["target_vols"].to(device)
        target_bvecs = batch["target_bvecs"].to(device)
        target_mask = batch["target_mask"].to(device)
        mask = target_mask[:, :, None, None, None, None].expand_as(target_vols).float()
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            pred = model(input_vols, input_bvecs, target_bvecs)
            err = (pred - target_vols).abs()
            loss = (err * mask).sum() / mask.sum().clamp(min=1.0)
            if args.angular_loss_weight > 0:
                loss_angular_sanity = compute_sh_angular_loss(
                    pred, target_vols, target_bvecs, target_mask,
                    l_max_cap=args.sh_loss_lmax_cap, high_order_min=args.sh_loss_high_order_min)
                loss = loss + args.angular_loss_weight * loss_angular_sanity
            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        print(f"[sanity] {split_name} OK ({time.time() - t0:.1f}s, loss={loss.item():.6f}, "
              f"sujeitos={sorted(set(batch['subject_tags']))})", flush=True)

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- comecando o loop de epocas de verdade", flush=True)

    # run_tag: sufixos analogos aos ja usados em scripts/04b_train_rrin.py/
    # scripts/04_train_rcae.py (_qc/_inclinv/_bn/_sh) -- aqui so l_max/
    # base_ch/norm_type mudam o SHAPE dos pesos (bloqueante para resume, ver
    # checagem abaixo), entao cada combinacao ganha seu proprio out_dir.
    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}"
    if args.l_max is not None:
        run_tag += f"_lmax{args.l_max}"
    if args.base_ch != 16:
        run_tag += f"_ch{args.base_ch}"
    if args.norm_type == "batch":
        run_tag += "_bn"
    if args.aggregation == "attention":
        run_tag += "_attn"
    if args.cross_direction_attention:
        run_tag += "_xattn"
        if args.cross_attn_heads != 4:
            run_tag += f"{args.cross_attn_heads}h"
    if args.decoder_base_ch is not None and args.decoder_base_ch != args.base_ch:
        run_tag += f"_dbc{args.decoder_base_ch}"
    if args.decoder_depth != 1:
        run_tag += f"_dd{args.decoder_depth}"
    if args.decoder_reinject_code:
        run_tag += "_dreinj"
    if args.angular_loss_weight > 0:
        # mesmo padrao de scripts/04_train_rcae.py -- evita colisao de
        # checkpoint entre a variante com/sem loss angular no MESMO combo
        # shell_b/n_level/l_max/base_ch/norm_type/aggregation.
        run_tag += "_sh"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_plot_dir = run_dir / "debug_patches"
    debug_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt -- caminho fixo, "
          f"usado pela etapa 5i)")
    print(f"[resumo] logs/debug deste run em: {run_dir}")

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
            raise FileNotFoundError(
                f"--resume-checkpoint {resume_ckpt_path} nao existe (confira o caminho, ou "
                f"use --no-resume pra comecar do zero sem retomar de checkpoint nenhum)")
        print(f"[resume] carregando checkpoint existente: {resume_ckpt_path}", flush=True)
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        old_args = ckpt.get("args", {})
        # l_max/base_ch/norm_type SAO bloqueantes (mudam shape dos pesos,
        # nao so a loss) -- mesma logica de bloqueio de decoder_type em
        # scripts/04_train_rcae.py / norm_type em scripts/04b_train_rrin.py.
        old_l_max = old_args.get("l_max")
        effective_old_l_max = old_l_max  # None so se o checkpoint tambem usou automatico
        if effective_old_l_max is None:
            print("[resume][aviso] checkpoint antigo nao registrou l_max explicito (ou usou "
                  "automatico) -- confie no proprio load_state_dict para pegar qualquer "
                  "incompatibilidade de shape.", flush=True)
        elif effective_old_l_max != args.l_max:
            raise ValueError(
                f"--l-max mudou entre o checkpoint ({old_l_max}) e esta chamada ({args.l_max}) "
                f"-- muda o numero de canais de PerDirectionEncoder3D/ImplicitDecoderHead3D "
                f"(sh_dim_for_lmax), incompativel para resume. Use --no-resume para treinar "
                f"a variante nova do zero.")
        old_base_ch = old_args.get("base_ch", 16)
        if old_base_ch != args.base_ch:
            raise ValueError(
                f"--base-ch mudou entre o checkpoint ({old_base_ch}) e esta chamada "
                f"({args.base_ch}) -- muda o shape de TODOS os pesos. Use --no-resume.")
        old_norm_type = old_args.get("norm_type", "instance")
        if old_norm_type != args.norm_type:
            raise ValueError(
                f"--norm-type mudou entre o checkpoint ({old_norm_type}) e esta chamada "
                f"({args.norm_type}) -- InstanceNorm3d e BatchNorm3d tem parametros "
                f"incompativeis (mesmo motivo de model/rrin3d.py). Use --no-resume.")
        # decoder_* (2026-09-14) -- todos mudam o shape dos pesos do decoder.
        old_dec_bc = old_args.get("decoder_base_ch", None)
        old_dec_bc_eff = old_dec_bc if old_dec_bc is not None else old_args.get("base_ch", 16)
        new_dec_bc_eff = args.decoder_base_ch if args.decoder_base_ch is not None else args.base_ch
        if old_dec_bc_eff != new_dec_bc_eff:
            raise ValueError(
                f"--decoder-base-ch efetivo ({new_dec_bc_eff}) nao bate com o checkpoint "
                f"({old_dec_bc_eff}) -- muda o shape dos pesos do ImplicitDecoderHead3D. "
                f"Use --no-resume.")
        old_dec_depth = old_args.get("decoder_depth", 1)
        if old_dec_depth != args.decoder_depth:
            raise ValueError(
                f"--decoder-depth mudou entre o checkpoint ({old_dec_depth}) e esta chamada "
                f"({args.decoder_depth}) -- muda o numero de camadas do decoder. Use --no-resume.")
        old_dec_reinj = old_args.get("decoder_reinject_code", False)
        if old_dec_reinj != args.decoder_reinject_code:
            raise ValueError(
                f"--decoder-reinject-code mudou entre o checkpoint ({old_dec_reinj}) e esta "
                f"chamada ({args.decoder_reinject_code}) -- muda o numero de canais de entrada "
                f"das camadas ocultas do decoder. Use --no-resume.")
        old_aggregation = old_args.get("aggregation", "mean")
        if old_aggregation != args.aggregation:
            raise ValueError(
                f"--aggregation mudou entre o checkpoint ({old_aggregation}) e esta chamada "
                f"({args.aggregation}) -- 'attention' adiciona a cabeca AttentionAggregator3D, "
                f"shape de pesos incompativel com 'mean'. Use --no-resume para treinar a "
                f"variante nova do zero (na pratica isso ja deveria acontecer sozinho: o "
                f"run_tag muda automaticamente com --aggregation attention, entao os dois "
                f"checkpoints vivem em out_dir/ diferentes e nunca deveriam colidir aqui).")
        old_cross_attn = old_args.get("cross_direction_attention", False)
        if old_cross_attn != args.cross_direction_attention:
            raise ValueError(
                f"--cross-direction-attention mudou entre o checkpoint ({old_cross_attn}) e "
                f"esta chamada ({args.cross_direction_attention}) -- adiciona/remove os pesos "
                f"de CrossDirectionAttention3D, incompativel para resume. Use --no-resume (na "
                f"pratica isso ja deveria acontecer sozinho: o run_tag muda automaticamente "
                f"com esta flag).")
        if args.cross_direction_attention:
            old_cross_attn_heads = old_args.get("cross_attn_heads", 4)
            if old_cross_attn_heads != args.cross_attn_heads:
                raise ValueError(
                    f"--cross-attn-heads mudou entre o checkpoint ({old_cross_attn_heads}) e "
                    f"esta chamada ({args.cross_attn_heads}) -- muda o shape interno de "
                    f"CrossDirectionAttention3D, incompativel para resume. Use --no-resume.")
        for key in ("shell_b", "n_level", "patch_size", "q_out", "lr",
                    "angular_loss_weight", "sh_loss_high_order_min", "sh_loss_lmax_cap"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional.",
                      flush=True)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("[resume][aviso] checkpoint antigo sem optimizer_state -- otimizador "
                  "reinicia do zero.", flush=True)
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", float("inf"))))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(f"[resume] retomando da epoca {start_epoch} (best_val={best_val:.6f}, "
              f"epochs_no_improve={epochs_no_improve}) -- treino ia ate a epoca "
              f"{args.epochs}", flush=True)
        if start_epoch > args.epochs:
            print(f"[resume] epoca de retomada ({start_epoch}) ja passa de --epochs "
                  f"({args.epochs}) -- nada a fazer, treino ja estava concluido.", flush=True)
    else:
        print("[resume] nenhum checkpoint anterior encontrado (ou --no-resume passado) -- "
              "comecando do zero.", flush=True)

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

    # NOTA (bug corrigido em 2026-09-08, mesmo padrao do fix em
    # scripts/04i_train_pairflow_star.py, ver addendum secao 33.24): o
    # bias-init PRECISA ser aplicado (ou pulado, com aviso) so' DEPOIS de
    # sabermos se havera' resume -- se aplicado antes (junto da criacao do
    # modelo, como era ate' aqui) e' silenciosamente sobrescrito pelo
    # model.load_state_dict(ckpt["model_state"]) do bloco de resume acima,
    # sem nenhum aviso, dando a falsa impressao (via a mensagem "[init] ...
    # inicializado") de que o bias-init teve efeito quando na verdade os
    # pesos resumidos (nao-inicializados por dado) prevaleceram.
    if args.init_output_bias_from_data:
        if resume_ckpt_path is not None:
            print("[init] --init-output-bias-from-data ignorado -- retomando de checkpoint "
                  "existente (os pesos resumidos do decoder prevalecem; bias-init so' faz "
                  "sentido comecando do zero).", flush=True)
        else:
            print("[init] estimando bias inicial da camada de saida a partir dos dados de "
                  "treino (--init-output-bias-from-data)...", flush=True)
            t0_bias = time.time()
            n_bias_batches = 4
            sum_target = 0.0
            n_target = 0.0
            bias_it = iter(train_loader)
            for _ in range(n_bias_batches):
                try:
                    bias_batch = next(bias_it)
                except StopIteration:
                    break
                tv = bias_batch["target_vols"]
                tm = bias_batch["target_mask"]
                m = tm[:, :, None, None, None, None].expand_as(tv).float()
                sum_target += (tv * m).sum().item()
                n_target += m.sum().item()
            del bias_it
            if n_target > 0:
                bias_val = sum_target / n_target
                with torch.no_grad():
                    model.decoder_head.net[-1].bias.fill_(bias_val)
                print(f"[init] bias de saida inicializado com a media do alvo "
                      f"({bias_val:.6f}, {n_bias_batches} batch(es) amostrados, "
                      f"{time.time() - t0_bias:.1f}s)", flush=True)
            else:
                print("[init][aviso] nao foi possivel estimar o bias (mascara vazia nos "
                      "batches amostrados?) -- mantendo inicializacao padrao do PyTorch.",
                      flush=True)

    if debug_fixed_batch is not None:
        snapshot_epoch = 0 if start_epoch == 1 else start_epoch - 1
        plot_fixed_debug_patch(model, debug_fixed_batch, device, debug_plot_dir,
                                epoch=snapshot_epoch, val_loss=(best_val if start_epoch > 1 else None),
                                shell_b=args.shell_b, n_level=args.n_level,
                                max_dirs=debug_max_dirs)

    debug_state = None
    if args.debug_plot_every_batches > 0:
        debug_state = {"dir": debug_plot_dir, "every": args.debug_plot_every_batches, "step": 0,
                        "max_dirs": debug_max_dirs}

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w") as f:
        # train_loss_signal/train_loss_angular/val_loss_signal/val_loss_angular:
        # colunas novas no FIM (mesma convencao de 04_train_rcae.py, addendum
        # secao 33.29) -- vazias quando --angular-loss-weight=0.0.
        f.write("epoch,train_loss,val_loss,lr,"
                "train_loss_signal,train_loss_angular,"
                "val_loss_signal,val_loss_angular\n")

    batch_log_path = run_dir / "batch_log.csv"
    batch_log_f = open(batch_log_path, "w")
    batch_log_f.write(
        "epoch,split,batch,loss,"
        "input_mean,input_std,input_min,input_max,input_n_outliers,"
        "target_mean,target_std,target_min,target_max,target_n_outliers,"
        "wait_s,compute_s,subject_tags,"
        "loss_signal,loss_angular\n"
    )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss, train_loss_signal, train_loss_angular = run_epoch(
                model, train_loader, optimizer, device, train=True,
                epoch=epoch, batch_log_f=batch_log_f,
                debug_state=debug_state, outlier_threshold=args.outlier_threshold,
                batch_log_every=args.batch_log_every,
                max_batches=args.max_train_batches,
                warmup_state=warmup_state,
                angular_loss_weight=args.angular_loss_weight,
                sh_loss_high_order_min=args.sh_loss_high_order_min,
                sh_loss_lmax_cap=args.sh_loss_lmax_cap)
            val_loss, val_loss_signal, val_loss_angular = run_epoch(
                model, val_loader, optimizer, device, train=False,
                epoch=epoch, batch_log_f=batch_log_f,
                outlier_threshold=args.outlier_threshold,
                batch_log_every=args.batch_log_every,
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
                print(f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

            if debug_fixed_batch is not None and (epoch % args.debug_plot_every == 0):
                plot_fixed_debug_patch(model, debug_fixed_batch, device, debug_plot_dir,
                                        epoch=epoch, val_loss=val_loss,
                                        shell_b=args.shell_b, n_level=args.n_level,
                                        max_dirs=debug_max_dirs)

            ckpt_common = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val": best_val,
                "epochs_no_improve": epochs_no_improve,
            }
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                ckpt_common["best_val"] = best_val
                ckpt_common["epochs_no_improve"] = epochs_no_improve
                torch.save(ckpt_common, out_dir / "best.pt")
                shutil.copy2(out_dir / "best.pt", run_dir / "best.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping na epoca {epoch} (sem melhora ha {args.patience} epocas)")
                    break

            ckpt_common["epochs_no_improve"] = epochs_no_improve
            torch.save(ckpt_common, out_dir / "last.pt")
            shutil.copy2(out_dir / "last.pt", run_dir / "last.pt")
    finally:
        batch_log_f.close()

    print("Treino concluido. Melhor val_loss:", best_val, "-> checkpoint em", out_dir / "best.pt")
    print(f"Copia permanente deste run em: {run_dir / 'best.pt'} (job_id={run_id})")
    print("Log por batch salvo em:", batch_log_path)


if __name__ == "__main__":
    main()