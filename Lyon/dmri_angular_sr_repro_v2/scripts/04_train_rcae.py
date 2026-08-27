#!/usr/bin/env python3
"""
Etapa 4: treina o RCAE para um (shell, nivel de subamostragem) especifico.
Repita a chamada para cada combinacao que quiser cobrir (ex.: shell 1000,
niveis 6/10/15/20/30).

Uso:
    python scripts/04_train_rcae.py \
        --manifest work_dir/manifest.csv \
        --scheme-dir work_dir/subsampling \
        --shell-b 1000 --n-level 10 \
        --out-dir work_dir/rcae_checkpoints \
        --epochs 100 --batch-size 2 --patch-size 24 --lr 1e-4

Requer PyTorch + GPU. Nao executado neste ambiente de desenvolvimento
(sem torch instalado); revisar/ajustar hiperparametros no cluster.
"""
import argparse
import os
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
from utils.sh_basis import real_sh_matrix, max_order_for_n_directions, cart2sphere
from model.rcae import RCAE


def n_coeffs_even(l_max: int) -> int:
    """Numero de coeficientes de uma base SH real so com ordens PARES ate
    `l_max` (inclusive): R = sum_{l=0,2,...,l_max} (2l+1) = (l_max+1)(l_max+2)/2.
    Mesma formula usada implicitamente em utils/sh_basis.py
    (max_order_for_n_directions/real_sh_matrix), so exposta aqui separada
    pra usar no aviso de startup sem precisar montar a matriz inteira."""
    return (l_max + 1) * (l_max + 2) // 2


def _sh_column_degrees(l_max: int) -> np.ndarray:
    """Grau `l` de cada coluna da matriz devolvida por
    utils.sh_basis.real_sh_matrix -- essa funcao NAO devolve os graus
    junto (ao contrario da versao alternativa em spherical_harmonics.py),
    entao replicamos aqui o MESMO laco de enumeracao de colunas usado
    dentro dela (`for l in range(0, l_max+1, 2): for m in range(-l, l+1)`)
    pra saber quais colunas sao "ordem alta" sem duplicar a matematica da
    base em si."""
    ls = []
    for l in range(0, l_max + 1, 2):
        for _m in range(-l, l + 1):
            ls.append(l)
    return np.array(ls, dtype=np.float64)


def compute_sh_angular_loss(pred, target_vols, target_bvecs, target_mask,
                             l_max_cap=8, high_order_min=4):
    """Termo de loss opcional no dominio angular/SH (ver protocolo, secao 9,
    prioridade 1): alem da MAE sobre o sinal bruto (que pesa TODO voxel
    igual, dominado pelos voxels de fibra unica que sao maioria do volume),
    penaliza tambem o erro nos coeficientes SH de ordem >= `high_order_min`
    (default l>=4) da FOD reconstruida -- exatamente os coeficientes que
    carregam a informacao de fibra cruzando que a POC de CSD
    (poc_csd_direction_count.py / sh_energy_by_order) mostrou serem onde o
    RCAE ja leva vantagem sobre o baseline SH.

    Projeta pred/target nas MESMAS direcoes-alvo (target_bvecs) usando a
    mesma base SH real do baseline classico (utils/sh_basis.py), por item
    do batch (cada item pode ter um subconjunto valido diferente de
    direcoes-alvo, por causa do padding de collate_variable_targets quando
    sujeitos do batch tem N_out diferente). A projecao (pseudo-inversa da
    matriz de base, calculada a partir dos bvecs, sem gradiente) e um
    mapeamento LINEAR nas direcoes -- o matmul com pred/target continua
    totalmente diferenciavel.

    Limitacao importante: com q_out=10 (default do treino, ver
    slurm/03_train_rcae.sh), so ha direcoes-alvo suficientes pra sustentar
    ate ordem l=2 (R=6 coeficientes; l=4 precisaria de 15). Ou seja, com o
    q_out atual esse termo captura a componente anisotropica de ordem 2
    (ja mais informativa que MAE puro, que mistura l=0 e l=2 com peso
    igual por voxel), mas NAO chega a l>=4 de verdade -- pra isso e
    preciso aumentar --q-out. Itens do batch sem direcoes-alvo validas
    suficientes pra sustentar `high_order_min` sao pulados (nao contribuem
    pra este termo; a loss de sinal continua normal pra eles)."""
    device = pred.device
    losses = []
    target_mask_cpu = target_mask.detach().cpu().numpy()
    for b in range(pred.shape[0]):
        valid = target_mask_cpu[b].astype(bool)
        n_valid = int(valid.sum())
        if n_valid == 0:  # item totalmente padding (collate_variable_targets) -- sem direcao valida
            continue
        bvecs_b = target_bvecs[b, valid].detach().cpu().numpy()
        l_max = min(max_order_for_n_directions(n_valid), l_max_cap)
        if l_max < high_order_min:
            # direcoes-alvo validas neste item nao sustentam a ordem pedida
            # -- pula (nao inventa coeficiente de ordem alta sem dado pra
            # sustentar), so a loss de sinal cobre este item.
            continue
        theta, phi = cart2sphere(bvecs_b)
        Bmat = real_sh_matrix(theta, phi, l_max)  # (n_valid, R)
        ls = _sh_column_degrees(l_max)            # (R,) -- mesma ordem de colunas
        high_idx = np.nonzero(ls >= high_order_min)[0]
        if high_idx.size == 0:
            continue
        Bmat_t = torch.as_tensor(Bmat, dtype=pred.dtype, device=device)
        pinv = torch.linalg.pinv(Bmat_t)  # (R, n_valid)
        valid_idx = torch.as_tensor(np.nonzero(valid)[0], device=device, dtype=torch.long)
        pred_b = pred[b, valid_idx].reshape(n_valid, -1)          # (n_valid, voxels)
        target_b = target_vols[b, valid_idx].reshape(n_valid, -1)
        coeffs_pred = pinv @ pred_b        # (R, voxels)
        coeffs_target = pinv @ target_b
        high_idx_t = torch.as_tensor(high_idx, device=device, dtype=torch.long)
        err = (coeffs_pred[high_idx_t] - coeffs_target[high_idx_t]).abs()
        losses.append(err.mean())
    if not losses:
        return torch.zeros((), device=device, dtype=pred.dtype)
    return torch.stack(losses).mean()


def _tensor_stats(x: torch.Tensor, outlier_threshold: float = None) -> tuple:
    """mean, std, min, max (+ n_outliers, se outlier_threshold for passado)
    -- usado no batch_log.csv pra sanity-check dos valores de patch (ex.:
    sinal normalizado pelo b0 deveria girar perto de 0-2; nan/inf ou
    tudo-zero aqui indicaria problema nos dados/mascara).

    n_outliers = quantidade de voxels com |valor| > outlier_threshold --
    depois da correcao do mascaramento de fundo, valores fora da faixa
    tipica (0-2) que ainda aparecem sao voxels DENTRO da mascara (ex.:
    LCR/vasos, onde a atenuacao pela difusao e menor e o sinal fica mais
    perto do b0) -- nao necessariamente erro, mas vale contar por batch pra
    nao precisar vasculhar o CSV inteiro manualmente atras de picos.
    """
    if x.numel() == 0:
        stats = (float("nan"), float("nan"), float("nan"), float("nan"))
        return stats + (0,) if outlier_threshold is not None else stats
    stats = (x.mean().item(), x.std().item(), x.min().item(), x.max().item())
    if outlier_threshold is not None:
        n_outliers = int((x.abs() > outlier_threshold).sum().item())
        return stats + (n_outliers,)
    return stats


def run_epoch(model, loader, optimizer, device, train: bool, b_ref: float,
              epoch: int, batch_log_f=None, debug_state=None, outlier_threshold: float = 3.0,
              batch_log_every: int = 5, angular_loss_weight: float = 0.0,
              sh_loss_high_order_min: int = 4, sh_loss_lmax_cap: int = 8):
    """debug_state (opcional): dict com "dir" (Path), "every" (int, a cada
    quantos batches de TREINO salvar um snapshot) e "step" (contador
    global, mutavel entre chamadas -- por isso e um dict e nao um int).
    Reaproveita o `pred` que o forward ja calculou pra loss (sem forward
    pass extra) e plota o primeiro item do batch atual.

    SEM print por batch -- so imprime o resumo por epoca/split (n_batches,
    throughput) no final. Deteccao de outlier (qual sujeito, quantos
    voxels) NAO gera aviso solto no stdout nem CSV separado -- fica
    embutida no proprio batch_log.csv (colunas input_n_outliers/
    target_n_outliers + subject_tags), que ja mostra o sujeito de cada
    batch sem precisar de filtro de "significancia" nenhum: quem quiser
    ver os outliers e so filtrar esse CSV (ex.: linhas onde
    input_n_outliers>0 ou target_n_outliers>0) depois.

    batch_log_every: so grava 1 a cada N batches no batch_log.csv (sempre
    grava o 1o batch da epoca) -- com treinos longos (muitas epocas x
    muitos batches por epoca), gravar TODO batch deixava o CSV enorme.
    Isso e amostragem, nao afeta o treino/loss em si, so o que fica
    registrado pra inspecao depois.
    """
    model.train(mode=train)
    total_loss = 0.0
    n_batches = 0
    n_samples = 0
    total_wait_s = 0.0
    total_compute_s = 0.0
    split = "train" if train else "val"
    prev_end = time.time()
    for batch in loader:
        # tempo entre o fim do processamento do batch anterior e o recebimento
        # deste = quanto tempo ficamos esperando o DataLoader (I/O de disco +
        # workers montando o patch) -- se isso dominar sobre "compute", a GPU
        # esta ociosa boa parte do tempo esperando dado, nao o contrario.
        t_received = time.time()
        wait_s = t_received - prev_end
        subject_tags = batch["subject_tags"]  # lista de str, um por item do batch
        input_vols = batch["input_vols"].to(device)
        input_bvecs = batch["input_bvecs"].to(device)
        input_bvals = batch["input_bvals"].to(device)
        target_vols = batch["target_vols"].to(device)
        target_bvecs = batch["target_bvecs"].to(device)
        target_bvals = batch["target_bvals"].to(device)
        # (B, N_out) -> (B, N_out, 1, 1, 1, 1) para broadcast contra
        # target_vols (B, N_out, 1, ps, ps, ps); mascara direcoes de
        # padding introduzidas pelo collate_variable_targets (ver
        # utils/dataset.py) quando sujeitos do batch tem N_out diferente.
        target_mask = batch["target_mask"].to(device)
        mask = target_mask[:, :, None, None, None, None].expand_as(target_vols).float()

        with torch.set_grad_enabled(train):
            # encoder/decoder chamados separado (em vez de model(...) direto)
            # -- MESMO calculo, RCAE.forward faz exatamente isso por dentro,
            # sem custo extra -- so pra ficar com "state" (o contexto, saida
            # do encoder, ANTES de condicionar em qualquer direcao-alvo) a
            # mao pro snapshot de debug (--debug-plot-every-batches), que
            # plota state ao lado de pred pra ajudar a ver se a rede esta
            # variando a predicao por direcao ou so devolvendo o contexto.
            state = model.encoder(input_vols, input_bvecs)
            pred = model.decoder(state, target_bvecs)
            # mask expandida pro shape completo (B, N_out, 1, ps, ps, ps) --
            # sem isso o denominador so conta pares (sujeito, direcao) e nao
            # os ps^3 voxels de cada um, o que deixaria a loss ~ps^3 vezes
            # maior que o mse_loss original (media sobre todos os elementos).
            # MAE (nao MSE) -- 'mae' e o loss default da implementacao
            # oficial (dmri_rcnn); trocamos aqui pra reproduzir mais fiel
            # (MSE penaliza outliers de voxel quadraticamente, o que pode
            # empurrar o modelo a "jogar seguro" e prever perto da media
            # entre direcoes justamente nos voxels mais variaveis).
            err = (pred - target_vols).abs()
            loss_signal = (err * mask).sum() / mask.sum().clamp(min=1.0)
            # termo angular/SH opcional (ver protocolo, secao 9, prioridade 1) --
            # desativado por padrao (angular_loss_weight=0.0), loss identica a
            # antes. Com peso > 0, soma-se a MAE dos coeficientes SH de ordem
            # alta (compute_sh_angular_loss acima) -- ver limitacao de q_out ali.
            if angular_loss_weight > 0:
                loss_angular = compute_sh_angular_loss(
                    pred, target_vols, target_bvecs, target_mask,
                    l_max_cap=sh_loss_lmax_cap, high_order_min=sh_loss_high_order_min)
                loss = loss_signal + angular_loss_weight * loss_angular
            else:
                loss_angular = None
                loss = loss_signal
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        t_compute_end = time.time()
        compute_s = t_compute_end - t_received

        total_loss += loss.item()
        n_batches += 1
        n_samples += input_vols.shape[0]
        total_wait_s += wait_s
        total_compute_s += compute_s
        # SEM print por batch aqui de proposito -- isso afogava o .out do
        # SLURM (uma linha por batch, milhares por epoca). O detalhe fica
        # todo no batch_log.csv (grava a mesma info por batch, sem perda de
        # granularidade); o stdout so recebe o resumo por epoca (mais
        # abaixo) e avisos que realmente importam (outlier, erro).

        # intercalado (ver batch_log_every) -- sempre grava o 1o batch da
        # epoca, senao so 1 a cada N. Sem isso, um treino de 150 epocas x
        # milhares de batches deixava o batch_log.csv gigante.
        if batch_log_f is not None and (n_batches == 1 or n_batches % max(1, batch_log_every) == 0):
            # estatisticas so sobre entradas "reais" (input e sempre valido;
            # target usa a mascara pra ignorar o padding do collate)
            in_mean, in_std, in_min, in_max, in_n_out = _tensor_stats(
                input_vols, outlier_threshold=outlier_threshold)
            # mask ja vem em (B, N_out, 1, ps, ps, ps) (calculada acima pra
            # loss) -- reaproveita em vez de reconstruir com Nones errados
            valid_target = target_vols[mask.bool()]
            tg_mean, tg_std, tg_min, tg_max, tg_n_out = _tensor_stats(
                valid_target, outlier_threshold=outlier_threshold)
            # subject_tags aqui ja mostra QUAIS sujeitos estao neste batch --
            # junto com input_n_outliers/target_n_outliers (colunas abaixo),
            # da pra filtrar o CSV depois (ex.: pandas, `n_outliers>0`) e ver
            # exatamente quais sujeitos tiveram outlier, sem filtro de
            # "significancia" nenhum aqui -- o dado bruto fica todo salvo,
            # e SEM aviso nenhum no stdout (isso ficava repetitivo demais).
            tags_str = ";".join(subject_tags)
            # loss_signal/loss_angular: colunas novas no FIM (nao mudam a posicao
            # das colunas antigas) -- loss_angular fica vazia quando o termo esta
            # desativado (angular_loss_weight=0.0), pra deixar claro no CSV que
            # aquele run nao usou o termo, em vez de escrever 0.0 (que poderia
            # ser confundido com "termo ativo mas convergiu pra zero").
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
                # o snapshot so plota o patch [0] do batch -- usa o tag do
                # sujeito correspondente (subject_tags[0]) no titulo, senao
                # fica impossivel saber qual sujeito gerou aquele patch so
                # olhando o PNG (o nome do arquivo so tem step/epoca/batch)
                subj0 = subject_tags[0] if subject_tags else "?"
                save_patch_debug_png(
                    png_path, input_vols[0], target_vols[0], pred_vols=pred[0].detach(),
                    context=state[0].detach(), max_dirs=debug_state["max_dirs"],
                    title=f"step {step} | epoca {epoch} batch {n_batches} | {subj0} | loss {loss.item():.6f}",
                )
                print(f"[debug] snapshot (batch) salvo em {png_path}", flush=True)

        # marca o fim do processamento deste batch DEPOIS do log/debug --
        # assim wait_s do PROXIMO batch inclui honestamente qualquer tempo
        # gasto em CSV/plot, nao so na chamada do modelo.
        prev_end = time.time()

    if n_batches > 0:
        total_s = total_wait_s + total_compute_s
        throughput = n_samples / total_s if total_s > 0 else float("nan")
        pct_wait = 100 * total_wait_s / total_s if total_s > 0 else float("nan")
        print(f"[{split}] epoca {epoch} resumo: {n_batches} batches, {n_samples} patches | "
              f"wait total {total_wait_s:.1f}s ({pct_wait:.0f}%) | compute total "
              f"{total_compute_s:.1f}s | {throughput:.2f} patches/s", flush=True)

    return total_loss / max(1, n_batches)


def plot_fixed_debug_patch(model, fixed_batch, device, b_ref, plot_dir, epoch, val_loss=None,
                            shell_b=None, n_level=None, max_dirs=6):
    """Roda o mesmo patch fixo de validacao pelo modelo (eval, no_grad) e
    salva o snapshot -- usado tanto pro baseline (epoca 0, antes de
    treinar) quanto pelos snapshots periodicos por epoca.

    `b_ref` nao e mais usado aqui dentro (a arquitetura atual do RCAE nao
    condiciona mais em bval, so em bvec -- ver model/rcae.py); mantido no
    parametro so pra nao mudar a assinatura/chamadas."""
    model.eval()
    with torch.no_grad():
        state = model.encoder(fixed_batch["input_vols"].to(device),
                               fixed_batch["input_bvecs"].to(device))
        pred = model.decoder(state, fixed_batch["target_bvecs"].to(device))
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
                     help="tamanho do patch cubico (default 10, igual ao paper: "
                          "patch_shape=(10,10,10)). Patches agora vem de uma grade "
                          "deterministica nao-sobreposta cobrindo o volume inteiro "
                          "(com zero-padding nas bordas), nao mais de crop aleatorio -- "
                          "ver utils/dataset.py.")
    ap.add_argument("--q-out", type=int, default=10,
                     help="numero fixo de direcoes-alvo por exemplo (default 10, igual "
                          "ao paper: N_out=10 fixo, nao 'todas as direcoes restantes da "
                          "shell').")
    ap.add_argument("--mask-suffix", default="_mask3d.nii.gz",
                     help="sufixo do arquivo de mascara real, procurado como "
                          "'<dwi_sem_extensao><mask_suffix>' na mesma pasta do dwi -- ver "
                          "utils/masking.py:find_mask_path. Default '_mask3d.nii.gz', igual "
                          "aos demais scripts do pipeline (03_baseline_sh_interpolation.py, "
                          "05_reconstruct_rcae.py, 07/08_downstream_*.py). ANTES este script "
                          "nao tinha esse flag e usava um default diferente ('_brainmask.nii.gz', "
                          "que nao bate com nenhum arquivo real) sem nenhum aviso -- caindo "
                          "SEMPRE no fallback simple_brain_mask (threshold simples do b0) em "
                          "vez da mascara de verdade. Corrigido: agora o default bate com o "
                          "resto do pipeline, e utils/masking.py avisa no log se cair no "
                          "fallback mesmo assim (ex.: sujeito sem mascara real).")
    ap.add_argument("--min-tile-coverage", type=float, default=0.1,
                     help="descarta da grade de patches os tiles com fracao de voxels de "
                          "mascara MENOR que isso (0 a 1). Default 0.1 (>=10% do tile "
                          "10x10x10 dentro do cerebro). O filtro antigo em "
                          "utils/dataset.py:_tile_origins so exigia >=1 voxel de mascara no "
                          "tile inteiro -- ou seja, tiles quase todo fundo (bem na borda do "
                          "cerebro) sempre entravam no pool de treino/validacao com o mesmo "
                          "peso que um tile cheio de sinal, o que explica ver bastante patch "
                          "'quase todo zero' no debug mesmo com a mascara certa (ver "
                          "utils/dataset.py:DWIPatchDataset.__init__ pra mais detalhe -- isso "
                          "NAO e o mesmo bug do --mask-suffix, e um filtro de amostragem "
                          "separado). Use 0.0 pra voltar ao comportamento antigo (nenhum tile "
                          "descartado por cobertura).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lstm-size", type=int, default=48,
                     help="unidades da ConvLSTM3D que agrega a sequencia de direcoes de "
                          "entrada (default 48, igual ao paper). Os demais canais do "
                          "modelo (104-668 por bloco multi-ramo) sao FIXOS -- replica "
                          "exata da arquitetura oficial, ver model/rcae.py -- nao da pra "
                          "ajustar por CLI de proposito, pra nao perder a fidelidade.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-cached-subjects", type=int, default=2,
                     help="quantos sujeitos (volume 4D completo) cada worker do DataLoader "
                          "mantem em cache (LRU) -- ver utils/dataset.py. Maior = menos "
                          "releitura de disco entre epocas/batches, mas mais RAM (memoria "
                          "usada aprox. num_workers x max_cached_subjects x tamanho_do_sujeito). "
                          "Com --num-workers 0 o cache e um so, compartilhado, sem multiplicar.")
    ap.add_argument("--val-num-workers", type=int, default=None,
                     help="workers do DataLoader de validacao (default: min(2, --num-workers)). "
                          "Com persistent_workers=True, os workers do train_loader E do "
                          "val_loader ficam residentes AO MESMO TEMPO a partir da 1a epoca "
                          "(cada DataLoader so sobe seus workers na 1a iteracao, mas depois "
                          "nunca morrem) -- ou seja, o pico de RAM e a SOMA dos dois, nao o "
                          "maximo. Validacao precisa de bem menos paralelismo que treino "
                          "(passa uma vez por epoca, sem shuffle), entao um valor menor aqui "
                          "evita OOM sem abrir mao do cache entre epocas.")
    ap.add_argument("--val-max-cached-subjects", type=int, default=1,
                     help="max_cached_subjects do dataset de validacao (default: 1, menor que "
                          "--max-cached-subjects). O DWIPatchDataset ja mantem os patches de "
                          "cada sujeito contiguos mesmo sem SubjectGroupedSampler (val usa "
                          "shuffle=False), entao 1 sujeito em cache por worker de validacao "
                          "ja evita releitura de disco sem inflar o pico de RAM combinado "
                          "com o cache do treino.")
    ap.add_argument("--torch-threads", type=int, default=0,
                     help="threads intra-op do PyTorch (torch.set_num_threads) pra CPU. "
                          "0 (default) usa o que o ambiente ja definiu (OMP_NUM_THREADS via "
                          "00_env_common.sh, que agora acompanha --cpus-per-task).")
    ap.add_argument("--patience", type=int, default=15,
                     help="early stopping: epocas sem melhora na val antes de parar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--job-id", default="", help="so para log/rastreabilidade (ex.: $SLURM_JOB_ID)")
    ap.add_argument("--debug-plot-every", type=int, default=0,
                     help="se > 0, salva um PNG (input/target/pred) do mesmo patch fixo de "
                          "validacao a cada N epocas (+ sempre um antes de comecar o treino, "
                          "'epoch_0000.png', pra ver o baseline sem treinar nada), em "
                          "out_dir/debug_patches/. 0 desativa (default).")
    ap.add_argument("--debug-plot-every-batches", type=int, default=0,
                     help="se > 0, alem do snapshot por epoca (fixo, comparavel), salva "
                          "tambem um PNG a cada N batches de TREINO (patch do batch atual, "
                          "sem custo extra -- reaproveita a predicao ja calculada pra loss). "
                          "Util pra ver a predicao mudando dentro de uma epoca longa, nao so "
                          "entre epocas. 0 desativa (default).")
    ap.add_argument("--outlier-threshold", type=float, default=3.0,
                     help="threshold de |valor| (sinal normalizado pelo b0) acima do qual um "
                          "voxel conta como outlier nas colunas input_n_outliers/"
                          "target_n_outliers do batch_log.csv (junto com subject_tags, da pra "
                          "filtrar dali quais sujeitos tiveram outlier -- sem aviso solto no "
                          "stdout). Default 3.0 -- o esperado e sinal ficar perto de 0-2; "
                          "valores bem acima podem ser fisiologia real (LCR/vasos) ou artefato, "
                          "vale conferir visualmente.")
    ap.add_argument("--batch-log-every", type=int, default=5,
                     help="grava 1 a cada N batches no batch_log.csv (sempre grava o 1o batch "
                          "da epoca). Default 5 -- gravar TODO batch deixava o CSV enorme em "
                          "treinos longos (muitas epocas x muitos batches/epoca). Use 1 para "
                          "voltar a granularidade total.")
    ap.add_argument("--angular-loss-weight", type=float, default=0.0,
                     help="lambda do termo de loss opcional no dominio angular/SH (ver "
                          "protocolo, secao 9, prioridade 1) -- soma lambda*erro_SH_ordem_alta "
                          "a MAE do sinal bruto (loss = loss_signal + lambda*loss_angular). "
                          "Default 0.0 = DESATIVADO, comportamento identico ao treino sem esse "
                          "termo (a checagem 'com ou sem' e so passar/nao passar essa flag). "
                          "Valores tipicos pra experimentar: 0.1-1.0 (a escala relativa dos "
                          "dois termos depende do numero de coeficientes SH de ordem alta, "
                          "ver --sh-loss-high-order-min). Ver compute_sh_angular_loss() acima "
                          "pra limitacao importante: com --q-out 10 (default), so ha direcoes-"
                          "alvo suficientes pra sustentar ate l=2 -- pra alcancar de fato "
                          "--sh-loss-high-order-min=4 e preciso tambem aumentar --q-out "
                          "(>=15).")
    ap.add_argument("--sh-loss-high-order-min", type=int, default=4,
                     help="grau l MINIMO (par) considerado 'ordem alta' no termo angular -- "
                          "coeficientes SH com l >= este valor entram na loss_angular. Default "
                          "4 (a partir da 1a ordem que carrega estrutura de fibra cruzando, "
                          "ver sh_energy_by_order em scripts/poc_csd_direction_count.py). So "
                          "tem efeito se --angular-loss-weight > 0.")
    ap.add_argument("--sh-loss-lmax-cap", type=int, default=8,
                     help="teto de l_max tentado por item de batch no termo angular (mesmo "
                          "papel do teto em max_order_for_n_directions no baseline SH). Default 8. "
                          "So tem efeito se --angular-loss-weight > 0.")
    ap.add_argument("--no-resume", action="store_true",
                     help="por padrao, se out_dir/<shell>_<n>/last.pt ja existir (de um "
                          "treino anterior do MESMO combo shell/n_level que morreu no meio -- "
                          "OOM, preempcao, timeout etc.), o treino RETOMA automaticamente "
                          "dali (pesos do modelo, estado do otimizador Adam, estado do "
                          "scheduler, epoca, best_val e contador de patience -- so a epoca "
                          "que estava em andamento quando o job morreu e perdida, nunca uma "
                          "epoca inteira ja concluida, ja que last.pt so e escrito no fim de "
                          "cada epoca). Passe --no-resume pra ignorar qualquer last.pt "
                          "existente e comecar do zero (equivalente ao comportamento antigo, "
                          "antes desta mudanca).")
    ap.add_argument("--resume-checkpoint", default=None,
                     help="caminho explicito de um checkpoint pra retomar (em vez do "
                          "out_dir/<shell>_<n>/last.pt 'canonico' -- por exemplo, a copia "
                          "permanente em .../runs/<job_id_antigo>/last.pt, se quiser retomar "
                          "de um run especifico em vez do mais recente). Ignorado se "
                          "--no-resume for passado.")
    ap.add_argument("--debug-max-dirs", type=int, default=0,
                     help="quantas direcoes (colunas) mostrar nos PNGs de debug. 0 (default) "
                          "= automatico, usa max(n_level, q_out) pra sempre mostrar TODAS as "
                          "direcoes de entrada e TODAS as q_out direcoes-alvo (antes ficava "
                          "fixo em 6, cortando direcoes se n_level/q_out fosse maior). Passe "
                          "um valor explicito pra limitar (ex.: 6) se os PNGs ficarem largos "
                          "demais com q_out grande.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device, "| job_id:", args.job_id or "(nao informado)")
    # automatico (0) = max(n_level, q_out) -- mostra TODAS as direcoes de
    # entrada e TODAS as direcoes-alvo nos PNGs de debug, em vez de cortar
    # nas primeiras 6 (o antigo default fixo de save_patch_debug_png).
    debug_max_dirs = args.debug_max_dirs if args.debug_max_dirs > 0 else max(args.n_level, args.q_out)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if device.type == "cpu":
        print(f"[cpu] torch.get_num_threads()={torch.get_num_threads()} "
              f"(cpus-per-task do SLURM: {os.environ.get('SLURM_CPUS_PER_TASK', '?')})")

    if args.angular_loss_weight > 0:
        max_l = min(max_order_for_n_directions(args.q_out), args.sh_loss_lmax_cap)
        print(f"[angular-loss] ATIVO: lambda={args.angular_loss_weight}, "
              f"high_order_min={args.sh_loss_high_order_min}, lmax_cap={args.sh_loss_lmax_cap}", flush=True)
        if max_l < args.sh_loss_high_order_min:
            print(f"[angular-loss][aviso] --q-out {args.q_out} so sustenta ate l={max_l} "
                  f"(precisaria de {n_coeffs_even(args.sh_loss_high_order_min)} direcoes-alvo "
                  f"pra chegar em l={args.sh_loss_high_order_min}) -- este termo vai ficar "
                  f"ZERADO em praticamente todos os batches (nenhuma direcao-alvo suficiente "
                  f"pra sustentar a ordem pedida). Aumente --q-out se quiser este termo com "
                  f"efeito real, ou reduza --sh-loss-high-order-min.", flush=True)
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
    # val_num_workers menor que num_workers (treino) de proposito -- ver
    # ajuda de --val-num-workers: com persistent_workers=True os workers do
    # train_loader E do val_loader ficam residentes ao mesmo tempo a partir
    # da 1a epoca (nenhum dos dois morre entre epocas), entao o pico de RAM
    # e a SOMA dos dois caches, nao o maximo. Sem essa reducao, um job com
    # --num-workers 4 (default) e --max-cached-subjects 2 (default) chega a
    # (4+4) x 2 = 16 sujeitos inteiros residentes em memoria ao mesmo
    # tempo -- foi isso que estourou o --mem=32G do sbatch no meio da 1a
    # validacao (job rodou a epoca de treino de boas, e so travou com OOM
    # quando os workers do val_loader subiram pela 1a vez).
    val_num_workers = args.val_num_workers if args.val_num_workers is not None \
        else min(2, args.num_workers)
    val_ds = DWIPatchDataset(val_entries, args.scheme_dir, args.shell_b, args.n_level,
                              patch_size=args.patch_size, q_out=args.q_out, training=False,
                              mask_suffix=args.mask_suffix,
                              min_tile_coverage=args.min_tile_coverage,
                              seed=args.seed + 1, max_cached_subjects=args.val_max_cached_subjects)

    # persistent_workers=True: sem isso, o DataLoader mata e recria os
    # workers a cada epoca -- e o cache de sujeitos de cada worker (ver
    # utils/dataset.py) e recriado do zero, entao TODA epoca comecava
    # recarregando os mesmos sujeitos do disco de novo, mesmo que a epoca
    # anterior ja tivesse carregado. So faz sentido com num_workers > 0.
    persistent_train = args.num_workers > 0
    persistent_val = val_num_workers > 0
    # SubjectGroupedSampler em vez de shuffle=True puro: embaralha por
    # SUJEITO (mantendo os patches de cada sujeito agrupados), pra reduzir
    # a troca de sujeito quase a cada batch que anulava o cache LRU do
    # DWIPatchDataset -- ver utils/dataset.py:SubjectGroupedSampler.
    train_sampler = SubjectGroupedSampler(train_ds, seed=args.seed)
    # worker_init_fn: sem isso, todo worker herda uma copia identica do RNG
    # de amostragem de patch (mesma seed) e fica em lockstep com os outros
    # -- ver comentario em utils/dataset.py:worker_init_fn e
    # DWIPatchDataset.__init__. So relevante com num_workers > 0.
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
    print(f"[resumo] batch_size={args.batch_size} patch_size={args.patch_size} "
          f"train: num_workers={args.num_workers} max_cached_subjects={args.max_cached_subjects} "
          f"persistent_workers={persistent_train} | "
          f"val: num_workers={val_num_workers} max_cached_subjects={args.val_max_cached_subjects} "
          f"persistent_workers={persistent_val}", flush=True)

    # amostra FIXA de validacao pra acompanhar visualmente a predicao
    # evoluindo epoca a epoca -- se sorteassemos um patch novo a cada vez,
    # a diferenca entre snapshots misturaria "mudou o patch" com "o modelo
    # aprendeu mais", inutil pra comparar. So ativa se --debug-plot-every >
    # 0 (custa uma leitura de subject + um forward pass extra por epoca
    # marcada, desprezivel).
    #
    # NAO usa mais val_ds[0] cego -- o indice 0 e sempre o primeiro tile
    # (em ordem crescente de eixo) do primeiro sujeito utilizavel, que
    # tende a cair bem no canto do volume (o cerebro raramente comeca
    # exatamente na origem) -- na pratica isso as vezes escolhia um patch
    # quase sem sinal nenhum (target_std~0), inutilizando essa serie toda
    # de snapshots. Em vez disso, escolhe o tile de MAIOR cobertura de
    # mascara (val_ds.tile_coverage, calculado de graca na hora de montar
    # a grade -- ver utils/dataset.py:_tile_origins), pra sempre cair num
    # patch de verdade representativo de tecido cerebral.
    debug_fixed_batch = None
    if args.debug_plot_every > 0:
        best_idx = int(np.argmax(val_ds.tile_coverage))
        best_si, best_origin = val_ds.tile_index[best_idx]
        best_entry, best_tag = val_ds.usable[best_si]
        print(f"[resumo] patch fixo de debug: sujeito={best_tag} origem={best_origin} "
              f"cobertura_mascara={val_ds.tile_coverage[best_idx]:.3f} (indice {best_idx} "
              f"de {len(val_ds)} tiles de validacao)", flush=True)
        debug_fixed_batch = collate_variable_targets([val_ds[best_idx]])

    model = RCAE(lstm_size=args.lstm_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                             factor=0.5, patience=5)

    # "smoke test" -- puxa 1 batch de treino e 1 de validacao e roda 1
    # forward(+backward no de treino) de cada, AQUI, antes do loop de
    # epocas. Motivo: com persistent_workers=True, os workers do
    # train_loader e do val_loader so sobem (e so passam a ocupar RAM) na
    # 1a vez que cada loader e iterado -- sem isso, um problema que so
    # aparece quando o val_loader sobe pela 1a vez (ex.: RAM combinada dos
    # dois conjuntos de workers estourando o --mem do job, que foi o que
    # gerou o OOM kill anterior) so seria descoberto depois de uma epoca de
    # TREINO inteira ja ter rodado (pode ser 20-30min) -- assim falha em
    # segundos/poucos minutos, com os dois loaders ja "quentes" quando o
    # loop principal comecar (o resultado deste batch nao conta pra loss
    # nem pro treino de verdade, e so warmup).
    print("[sanity] testando 1 batch de treino + 1 de validacao antes do loop de epocas "
          "(pra falhar rapido em vez de so depois de uma epoca inteira)...", flush=True)

    def _sanity_step(loader, split_name, do_backward):
        t0 = time.time()
        batch = next(iter(loader))
        input_vols = batch["input_vols"].to(device)
        input_bvecs = batch["input_bvecs"].to(device)
        input_bvals = batch["input_bvals"].to(device)
        target_vols = batch["target_vols"].to(device)
        target_bvecs = batch["target_bvecs"].to(device)
        target_bvals = batch["target_bvals"].to(device)
        target_mask = batch["target_mask"].to(device)
        mask = target_mask[:, :, None, None, None, None].expand_as(target_vols).float()
        model.train(mode=do_backward)
        with torch.set_grad_enabled(do_backward):
            pred = model(input_vols, input_bvecs, input_bvals, target_bvecs, target_bvals,
                         b_ref=args.shell_b)
            err = (pred - target_vols).abs()  # MAE, ver comentario em run_epoch
            loss = (err * mask).sum() / mask.sum().clamp(min=1.0)
            # exercita tambem o termo angular aqui (se ativado) -- assim um erro
            # nele (ex.: bvecs degenerados, singularidade no pinv) aparece no
            # sanity check em segundos, nao so depois de uma epoca inteira.
            if args.angular_loss_weight > 0:
                loss_angular = compute_sh_angular_loss(
                    pred, target_vols, target_bvecs, target_mask,
                    l_max_cap=args.sh_loss_lmax_cap, high_order_min=args.sh_loss_high_order_min)
                loss = loss + args.angular_loss_weight * loss_angular
            if do_backward:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        print(f"[sanity] {split_name} OK ({time.time() - t0:.1f}s, loss={loss.item():.6f}, "
              f"sujeitos={sorted(set(batch['subject_tags']))})", flush=True)

    _sanity_step(train_loader, "treino", do_backward=True)
    _sanity_step(val_loader, "validacao", do_backward=False)
    print("[sanity] ok -- workers de treino e validacao ja estao de pe, comecando o loop "
          "de epocas de verdade", flush=True)

    # ATENCAO (bug real ja encontrado em producao, ver protocolo): sem este
    # sufixo, treinar a variante COM loss angular/SH (--angular-loss-weight
    # > 0) no MESMO (shell_b, n_level) que a variante SEM ela colide no
    # MESMO out_dir/best.pt/last.pt canonico -- as duas competem pelo mesmo
    # arquivo (auto-resume inclusive pode fazer uma continuar da outra sem
    # avisar, alem da corrida entre os dois processos escrevendo o mesmo
    # caminho). O aviso de --angular-loss-weight mudou entre checkpoint e
    # chamada (algumas linhas abaixo) so pega isso NA HORA DE RETOMAR; o
    # sufixo aqui evita a colisao de raiz, mesmo sem nenhum resume envolvido
    # (dois `sbatch` para o mesmo combo, um com SH outro sem, rodando em
    # paralelo). Mesmo padrao ja usado em scripts/04b_train_rrin.py (_qc/_inclinv).
    run_tag = f"shell{int(args.shell_b)}_n{args.n_level}"
    if args.angular_loss_weight > 0:
        run_tag += "_sh"
    out_dir = Path(args.out_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # best.pt/last.pt ficam DIRETO em out_dir (caminho fixo e previsivel) --
    # 04_reconstruct_rcae.sh monta esse mesmo caminho na mao
    # ("$WORK_DIR/rcae_checkpoints/shell${SHELL_B}_n${N_LEVEL}/best.pt"),
    # entao nao da pra meter o job_id ai sem quebrar a etapa seguinte.
    #
    # Mas logs e snapshots de debug (que voce reroda toda hora testando)
    # vao numa subpasta com o job_id, pra cada submissao ficar isolada e
    # comparavel em vez de se sobrescreverem a cada `sbatch` novo.
    run_id = args.job_id.replace("/", "_") if args.job_id else "sem_job_id"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_plot_dir = run_dir / "debug_patches"
    debug_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"[resumo] checkpoints em: {out_dir} (best.pt/last.pt -- caminho fixo, "
          f"usado pela etapa 5)")
    print(f"[resumo] logs/debug deste run em: {run_dir}")

    # resume automatico (ver --no-resume/--resume-checkpoint): por padrao,
    # se ja existir um last.pt do MESMO combo (shell,n_level) -- de um
    # treino anterior morto no meio (OOM, preempcao, timeout, etc., ver
    # discussao no protocolo secao 9 prioridade 3) -- carrega dali em vez
    # de comecar do zero. So a epoca que estava em andamento na hora da
    # morte e perdida (last.pt so e escrito no FIM de cada epoca).
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
        # checagem de sanidade -- so um AVISO, nao impede o resume, porque
        # n_level/q_out/patch_size/shell_b nao mudam o SHAPE dos pesos (os
        # canais do modelo sao fixos, ver model/rcae.py), so mudariam o
        # SIGNIFICADO do que o modelo ja aprendeu. lstm_size SIM muda o
        # shape -- se estiver diferente, model.load_state_dict abaixo vai
        # falhar sozinho com um erro claro de mismatch de shape.
        #
        # angular_loss_weight/sh_loss_high_order_min/sh_loss_lmax_cap
        # tambem entram aqui: o caminho canonico do checkpoint
        # (out_dir/<shell>_<n>/last.pt) e o MESMO independente desses
        # parametros -- ou seja, um run "sem SH loss" e um "com SH loss" do
        # MESMO (shell_b, n_level) se sobrescrevem no mesmo arquivo. Sem
        # este aviso, retomar um treino "sem SH" depois de ter rodado "com
        # SH" nesse combo continuaria silenciosamente a partir do
        # checkpoint ERRADO (o mais recente, seja lá qual config gerou).
        old_args = ckpt.get("args", {})
        for key in ("shell_b", "n_level", "patch_size", "q_out", "lstm_size",
                    "angular_loss_weight", "sh_loss_high_order_min", "sh_loss_lmax_cap"):
            old_val, new_val = old_args.get(key), vars(args).get(key)
            if old_val is not None and old_val != new_val:
                print(f"[resume][aviso] --{key.replace('_','-')} mudou entre o checkpoint "
                      f"({old_val}) e esta chamada ({new_val}) -- confira se e intencional "
                      f"(ex.: voce pode estar prestes a continuar um treino 'sem SH loss' a "
                      f"partir de um checkpoint que foi treinado 'com SH loss', ou vice-versa, "
                      f"porque os dois usam o MESMO caminho canonico de checkpoint -- ver "
                      f"protocolo secao 9). Se nao for intencional, use --no-resume ou aponte "
                      f"--resume-checkpoint pro job certo.", flush=True)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("[resume][aviso] checkpoint antigo sem optimizer_state (salvo antes desta "
                  "mudanca) -- otimizador reinicia do zero (momentos do Adam perdidos, mas os "
                  "PESOS do modelo continuam retomados normalmente).", flush=True)
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

    # snapshot ANTES do loop de epocas -- com pesos aleatorios (baseline de
    # verdade) se NAO houve resume, ou com os pesos JA RETOMADOS se houve
    # (rotulado com a epoca de retomada em vez de 0, senao pareceria um
    # baseline aleatorio quando na verdade ja e o estado da epoca
    # start_epoch-1). Movido pra DEPOIS do bloco de resume acima --
    # antes desta mudanca este snapshot rodava sempre com o modelo recem-
    # criado, mesmo quando o resume ja tinha carregado os pesos retomados,
    # o que deixava o primeiro PNG enganoso (mostrava pesos aleatorios
    # rotulados como o estado atual do treino retomado).
    if debug_fixed_batch is not None:
        snapshot_epoch = 0 if start_epoch == 1 else start_epoch - 1
        plot_fixed_debug_patch(model, debug_fixed_batch, device, args.shell_b, debug_plot_dir,
                                epoch=snapshot_epoch, val_loss=(best_val if start_epoch > 1 else None),
                                shell_b=args.shell_b, n_level=args.n_level,
                                max_dirs=debug_max_dirs)

    # contador global de batches de treino (mutavel via dict, atravessa as
    # chamadas de run_epoch de todas as epocas) pros snapshots mais
    # frequentes (--debug-plot-every-batches), independente do snapshot por
    # epoca acima (que usa sempre o MESMO patch fixo de val).
    debug_state = None
    if args.debug_plot_every_batches > 0:
        debug_state = {"dir": debug_plot_dir, "every": args.debug_plot_every_batches, "step": 0,
                        "max_dirs": debug_max_dirs}

    log_path = run_dir / "train_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,lr\n")

    # log por batch (nao so por epoca) -- da pra ver progresso dentro de uma
    # epoca longa e conferir se os valores de patch (input/target) estao
    # numa faixa coerente (sinal normalizado pelo b0, deveria ficar perto de
    # 0-2; nan/inf ou tudo-zero aqui sinaliza problema nos dados/mascara
    # antes mesmo de esperar a epoca terminar).
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
            train_sampler.set_epoch(epoch)  # ordem de sujeitos diferente a cada epoca
            train_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                                    b_ref=args.shell_b, epoch=epoch, batch_log_f=batch_log_f,
                                    debug_state=debug_state, outlier_threshold=args.outlier_threshold,
                                    batch_log_every=args.batch_log_every,
                                    angular_loss_weight=args.angular_loss_weight,
                                    sh_loss_high_order_min=args.sh_loss_high_order_min,
                                    sh_loss_lmax_cap=args.sh_loss_lmax_cap)
            val_loss = run_epoch(model, val_loader, optimizer, device, train=False,
                                  b_ref=args.shell_b, epoch=epoch, batch_log_f=batch_log_f,
                                  outlier_threshold=args.outlier_threshold,
                                  batch_log_every=args.batch_log_every,
                                  angular_loss_weight=args.angular_loss_weight,
                                  sh_loss_high_order_min=args.sh_loss_high_order_min,
                                  sh_loss_lmax_cap=args.sh_loss_lmax_cap)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{current_lr:.2e}\n")
            print(f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

            if debug_fixed_batch is not None and (epoch % args.debug_plot_every == 0):
                plot_fixed_debug_patch(model, debug_fixed_batch, device, args.shell_b,
                                        debug_plot_dir, epoch=epoch, val_loss=val_loss,
                                        shell_b=args.shell_b, n_level=args.n_level,
                                        max_dirs=debug_max_dirs)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "best_val": best_val,
                    "epochs_no_improve": epochs_no_improve,
                }, out_dir / "best.pt")
                # copia (nao re-serializa) pra dentro de runs/<job_id>/ --
                # historico permanente desse run especifico, que o
                # out_dir/best.pt "canonico" (o mais recente) vai continuar
                # sobrescrevendo a cada novo treino do mesmo combo. Assim
                # da pra sempre voltar ao checkpoint de um run antigo pelo
                # job_id, mesmo que um run mais novo tenha rodado por cima.
                shutil.copy2(out_dir / "best.pt", run_dir / "best.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping na epoca {epoch} (sem melhora ha {args.patience} epocas)")
                    break

            # last.pt AGORA e salvo a cada epoca (nao so uma vez no final, fora
            # do try/finally) -- ANTES, se o job morresse no meio do treino
            # (OOM killer do SLURM, timeout, no cluster mata o processo com
            # SIGKILL -- nao da tempo nem do 'finally' rodar), o codigo que
            # salvava last.pt (depois do loop inteiro) nunca era alcancado, e
            # last.pt ficava simplesmente inexistente pra qualquer treino que
            # nao terminasse "limpo". Salvando a cada epoca, mesmo um job que
            # morre no meio deixa o last.pt da ULTIMA epoca que terminou
            # completa -- util pra continuar de onde parou ou so inspecionar
            # o estado mais recente, mesmo sem ter sido o melhor val_loss.
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
    print("Log por batch salvo em:", batch_log_path)


if __name__ == "__main__":
    main()