import os
import sys
import random
import time
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

import yaml
import wandb
from monai.inferers import sliding_window_inference

from model import QSpaceAttentionNetwork
from model_v2 import QSpaceAttentionNetwork_v2
from dataset import QSpaceDatasetCoord_KNearest_Shell
from losses import PhysicalCompoundLoss
from metrics import calculate_rmse_corr, calculate_region_metrics, calculate_res_sign_consistency
from utils import (
    save_comparison_png,
    save_debug_documentation_png,
    plot_q_space_polar,
    plot_q_space_selection_antipodal,
    backup_code,
    save_debug_image,
)

# ---------------------------------------------------------------------------
# VALIDAÇÃO POR PATCH
# ---------------------------------------------------------------------------

def validate(model, loader, criterion, device, max_steps=None):

    model.eval()

    # Acumuladores de Loss
    accum = {
        "loss": 0.0, "l1": 0.0, "ssim": 0.0, "res": 0.0, "grad": 0.0,
    }
    
    # Acumuladores de Métricas Estratificadas
    # Usamos listas para calcular a média real ao final
    metrics_accum = {
        "same_shell":  {"rmse": [], "pearson": [], "sign": []},
        "cross_shell": {"rmse": [], "pearson": [], "sign": []}
    }
    
    res_ratios = []
    steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="🧪 Validando"):
            # Envio para device
            neighbors = batch["source_neighbors"].to(device)
            target    = batch["target_real"].to(device)
            query     = batch["target_query"].to(device)
            n_coords  = batch["neighbors_coords"].to(device)
            mask      = batch["mask"].to(device)
            maskWM    = batch["maskWM"].to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Output completo do modelo
                output, res_pred, media_vizinhos = model(neighbors, query, n_coords)

            # --- CÁLCULO DAS LOSSES ---
            loss, loss_dict = criterion(
                output.float(), target.float(), mask.float(), maskWM.float(),
                pred=res_pred.float(), media_vizinhos=media_vizinhos.float()
            )

            # --- CÁLCULO DAS MÉTRICAS ---
            # 1. Identificar se é Cross-Shell (diferença entre b_origin e b_target)
            # Pegamos o b_origin do primeiro vizinho do batch
            b_origin = n_coords[0, 0, 0].cpu().item() 
            b_target = query[0, 0].cpu().item()
            is_cross = abs(b_origin - b_target) > 0.05 

            # 2. Métricas de Intensidade e Correlação (No volume predito final)
            rmse_val, pearson_val = calculate_rmse_corr(output, target, mask)
            
            # 3. Consistência de Sinal (No resíduo)
            sign_c = calculate_res_sign_consistency(res_pred, target, media_vizinhos, mask)
            
            # 4. Res Ratio (Magnitude do resíduo vs Target)
            # Mede quanto a rede está "atuando"
            # 1. Calcula a magnitude média do resíduo onde há cérebro
            res_mag = torch.mean(torch.abs(res_pred[mask.bool()]))

            # 2. Calcula a magnitude média do target onde há cérebro
            target_mag = torch.mean(torch.abs(target[mask.bool()]))

            # 3. Ratio global: quanto o resíduo representa do sinal médio do patch
            current_ratio = res_mag / (target_mag + 1e-8)

            res_ratios.append(current_ratio.item())

            # Organizar nos dicionários
            key = "cross_shell" if is_cross else "same_shell"
            metrics_accum[key]["rmse"].append(rmse_val)
            metrics_accum[key]["pearson"].append(pearson_val)
            metrics_accum[key]["sign"].append(sign_c)

            # Acumular Losses
            for k in accum:
                if k in loss_dict:
                    accum[k] += loss_dict[k].item()
            accum["loss"] += loss.item()

            steps += 1
            if (
                max_steps is not None
                and steps >= max_steps
            ):
                break



    # --- MÉDIAS FINAIS ---
    for k in accum:
        accum[k] /= max(steps, 1)

    # Prepara dicionário para o WandB
    # Usamos .get() e if para evitar erro de divisão por zero se uma categoria não aparecer no batch
    final_logs = {
        "val/loss": accum["loss"],
        "val/res_ratio": np.mean(res_ratios),
        "val/prediction_hist": wandb.Histogram(res_pred.cpu().float().numpy())
    }

    for task in ["same_shell", "cross_shell"]:
        if metrics_accum[task]["rmse"]:
            final_logs[f"val/rmse_{task}"] = np.mean(metrics_accum[task]["rmse"])
            final_logs[f"val/pearson_{task}"] = np.mean(metrics_accum[task]["pearson"])
            final_logs[f"val/sign_consistency_{task}"] = np.mean(metrics_accum[task]["sign"])

    # Log único no final da validação
    wandb.log(final_logs)
    model.train()
    # Retornamos o accum para o scheduler e res_pred para debug visual se necessário
    return final_logs, res_pred


# ---------------------------------------------------------------------------
# LOG DE IMAGENS NO W&B
# ---------------------------------------------------------------------------

def log_images_to_wb(output, target, step, alpha, prefix="train"):
    slice_idx    = output.shape[-1] // 2
    pred_slice   = output[0, 0, :, :, slice_idx].detach().cpu().float().numpy()
    target_slice = target[0, 0, :, :, slice_idx].detach().cpu().float().numpy()
    pred_slice   = np.clip(pred_slice * alpha, 0, 1)
    target_slice = np.clip(target_slice * alpha, 0, 1)

    wandb.log({
        f"{prefix}/predictions":  wandb.Image(pred_slice,   caption=f"Pred Step {step}"),
        f"{prefix}/ground_truth": wandb.Image(target_slice, caption=f"Target Step {step}"),
    }, step=step)


# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

def init_config(config_path="train_config.yaml"):
    with open(config_path, 'r') as stream:
        try:
            dic_config = yaml.safe_load(stream)
            print(f"📖 Configurações carregadas de: {config_path}", flush=True)
        except yaml.YAMLError as exc:
            print(f"❌ Erro ao ler o YAML: {exc}", flush=True)
            return None
    return dic_config


def setup_environment(dic_config):

    if torch.cuda.is_available():
        # Retorna o índice da GPU atual (geralmente 0)
        current_device = torch.cuda.current_device()
        
        # Retorna o nome da GPU
        print(f"GPU selecionada: {torch.cuda.get_device_name(current_device)}")
    else:
        print("CUDA não está disponível. O código está usando a CPU.")

    torch.backends.cudnn.deterministic = False

    seed = dic_config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    rs = RandomState(MT19937(SeedSequence(seed)))

    date_time_ref = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    name_wandb = '{}-job{}-{}'.format(
        date_time_ref, dic_config['job_id'], dic_config.get('tbcomment')
    )

    # CORRIGIDO: chave da API lida da variável de ambiente em vez de hardcoded.
    # Configure com: export WANDB_API_KEY="sua_chave"
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if wandb_key:
        wandb.login(key=wandb_key)
    # Se não houver a env var, o wandb tentará ler de ~/.netrc automaticamente.

    wandb.init(
        project=dic_config['name_project'],
        name=name_wandb,
        config=dic_config,
        reinit="return_previous",
    )

    return rs


# ---------------------------------------------------------------------------
# LOOP DE TREINO
# ---------------------------------------------------------------------------

def train(dic_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_checkpoint = f"checkpoints/{dic_config['job_id']}"
    os.makedirs(folder_checkpoint, exist_ok=True)
    backup_code(folder_checkpoint)

    model_source    = dic_config.get('model')

    patch_size  = dic_config.get('patch_size')

    batch_size  = dic_config.get('batch_size')
    epochs      = dic_config.get('epochs')
    lr          = dic_config.get('lr')
    alpha       = dic_config.get('alpha')
    k_neighbors = dic_config.get('k_neighbors')
    bval_max    = dic_config.get('bval_max')

    print(f"Usando alpha={alpha}, k_neighbors={k_neighbors}, bval_max={bval_max}", flush=True)

    BASE_DIR       = dic_config.get('csv_folder')
    TRAIN_CSV      = os.path.join(BASE_DIR, 'train.csv')
    VAL_CSV_PATCH  = os.path.join(BASE_DIR, 'val.csv')

    base_path = dic_config.get('base_path')

    # --- DATASETS ---
    train_ds = QSpaceDatasetCoord_KNearest_Shell(
        TRAIN_CSV, base_path, "bgpdwis_PA_geomcorr", alpha, bval_max,
        patch_size=patch_size, mode="train",
        k_neighbors=k_neighbors, mode_val="patch"
    )
    val_ds_patch = QSpaceDatasetCoord_KNearest_Shell(
        VAL_CSV_PATCH, base_path, "bgpdwis_PA_geomcorr", alpha, bval_max,
        patch_size=patch_size, mode="val",
        k_neighbors=k_neighbors, mode_val="patch"
    )


    # =====================================================
    # FAST VALIDATION SUBSET
    # =====================================================

    FAST_VAL_SIZE = min(
        400,
        len(val_ds_patch)
    )

    rng = np.random.RandomState(42)

    fast_indices = rng.choice(
        len(val_ds_patch),
        FAST_VAL_SIZE,
        replace=False
    )

    val_fast_ds = Subset(
        val_ds_patch,
        fast_indices
    )

    # --- DATALOADERS ---
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, prefetch_factor=1,
    )
    val_loader_patch = DataLoader(
        val_ds_patch, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, prefetch_factor=1,
    )

    val_loader_fast = DataLoader(
        val_fast_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=1,
    )


    print(f"📊 Treino: {len(train_ds)} patches | Val patch: {len(val_ds_patch)} | Val Fast patch: {len(val_fast_ds)}", flush=True)

    # --- MODELO, LOSS, OTIMIZADOR ---
    weights = dic_config.get('loss_weights')

    if model_source == "v1":
        model = QSpaceAttentionNetwork(k_neighbors=k_neighbors).to(device)
    elif model_source == "v2":
        model = QSpaceAttentionNetwork_v2(k_neighbors=k_neighbors).to(device)
    else:
        print(f"Model not available:{model_source}")

    criterion = PhysicalCompoundLoss(
        patch_size = patch_size,
        l1_weight=weights['l1'],
        ssim_weight=weights['ssim'],
        grad_weight=weights['grad'],
        res_weight=weights['res'],
        wm_multiplier=weights['wm'],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    scaler = torch.amp.GradScaler('cuda')

    # CORRIGIDO: checkpoint_path pode ser None se a chave não existir no YAML.
    # if checkpoint_path != "" falhava com None != "" == True -> crash.
    checkpoint_path = dic_config.get('load_model')
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Ao carregar para continuar:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Pesos carregados de {checkpoint_path}. LR={lr}", flush=True)

    # --- LOOP PRINCIPAL ---
    global_step   = 0
    best_val_score = float('inf')
    fase          = 1
    fase_manual   = None
    pesos_manuais = False
    for epoch in range(epochs):

        if fase_manual is None:
            cl = dic_config.get('curriculum')

            if epoch < cl.get('fase1'):
                fase = 1
            elif epoch < cl.get('fase2'):
                fase = 2
            else:
                fase = 3
        else:
            fase = fase_manual

        train_ds.set_fase(fase)
        val_ds_patch.set_fase(fase)
        
        if not pesos_manuais:
            criterion.set_phase_weights(fase)

        print(f"🔥 Epoch {epoch+1}/{epochs} | Fase {fase}", flush=True)
        model.train()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        print("🔄 Solicitando primeiro batch ao DataLoader...", flush=True)

        for batch in progress_bar:
            # --- CONTROLE EXTERNO ---
            # Dentro do loop de treinamento
            if global_step % 100 == 0:
                if os.path.exists("cmd_control.txt"):
                    try:
                        with open("cmd_control.txt", "r") as f:
                            partes = f.read().strip().split(',')
                            if len(partes) >= 7:
                                novo_lr   = float(partes[0].strip())
                                nova_fase = int(partes[1].strip())
                                w_l1      = float(partes[2].strip())
                                w_ssim    = float(partes[3].strip())
                                w_grad    = float(partes[4].strip())
                                w_res     = float(partes[5].strip())
                                w_wm      = float(partes[6].strip())

                                # 1. Trava a Fase Manualmente
                                if fase_manual != nova_fase:
                                    print(f"--- MUDANÇA DE FASE MANUAL: {fase} -> {nova_fase} ---")
                                    fase_manual = nova_fase
                                    fase = nova_fase # Atualiza a fase local também
                                    train_ds.set_fase(fase)
                                    val_ds_patch.set_fase(fase)

                                # 2. Aplica Pesos e Ativa a Flag de Proteção
                                criterion.w_l1 = w_l1
                                criterion.w_ssim = w_ssim
                                criterion.w_grad = w_grad
                                criterion.w_res = w_res
                                criterion.w_wm = w_wm
                                pesos_manuais = True # <-- BLOQUEIA o reset automático no início da epoch
                                
                                # 3. Atualiza LR
                                for param_group in optimizer.param_groups:
                                    if param_group['lr'] != novo_lr:
                                        print(f"--- AJUSTE LR MANUAL: {novo_lr} ---")
                                        param_group['lr'] = novo_lr

                    except Exception as e:
                        print(f"Erro no controle: {e}")

            optimizer.zero_grad()

            neighbors = batch["source_neighbors"].to(device)  # [B, K, 1, 32, 32, 32]
            target    = batch["target_real"].to(device)       # [B, 1, 32, 32, 32]
            query     = batch["target_query"].to(device)      # [B, 4]
            n_coords  = batch["neighbors_coords"].to(device)  # [B, K, 4]
            mask      = batch["mask"].to(device)              # [B, 1, 32, 32, 32]
            maskWM    = batch["maskWM"].to(device)

            t1 = time.time()

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output_final, res_predito, media_viz = model(neighbors, query, n_coords)
                t2 = time.time()

                loss, loss_dict = criterion(
                    output_final.float(), target.float(),
                    mask.float(), maskWM.float(),
                    pred=res_predito.float(),
                    media_vizinhos=media_viz.float(),
                )
                t3 = time.time()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t4 = time.time()

            origin_b = batch["origin_bval"]
            target_b = batch["target_bval"]

            wandb.log({
                "train/total_loss": loss.item(),
                "train/l1_loss":    loss_dict["l1"].item(),
                "train/ssim_loss":  loss_dict["ssim"].item(),
                "train/grad_loss":  loss_dict["grad"].item(),
                "train/res_loss":   loss_dict["res"].item(),
                "meta/origin_bval": origin_b[0].item(),
                "meta/target_bval": target_b[0].item(),
                "meta/fase":        fase,
                "global_step":      global_step,
                "epoch":            epoch + 1,
            })

            progress_bar.set_postfix({
                "L1":   f"{loss_dict['l1'].item():.4f}",
                "SSIM": f"{loss_dict['ssim'].item():.4f}",
            })

            # ---- VALIDAÇÃO POR PATCH ----
            if global_step % 50 == 0 and global_step != 0:
                print(f"VALIDAÇÃO {global_step}")

                # val_logs agora contém chaves como 'val/rmse_cross_shell', 'val/loss', etc.
                val_logs, res_pred = validate(model, val_loader_fast, criterion, device, max_steps=25)

                # Salva sempre o último estado para permitir Resume em caso de queda do servidor
                # Ao salvar o checkpoint:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), # Salve o scheduler também!
                    'step': global_step,
                }
                torch.save(checkpoint, f"{folder_checkpoint}/last_model.pt")
                torch.cuda.empty_cache()

            # ---- DEBUG VISUAL (a cada 50 steps) ----
            if global_step % 50 == 0:
                print(f"DEBUG VISUAL {global_step}")
                model.eval()
                with torch.no_grad():
                    subj_id         = batch["id"][0]
                    target_idx      = batch["target_idx"][0].item()
                    neighbor_indices = batch["neighbor_indices"][0].cpu().numpy()

                    bvals_plot  = np.loadtxt(f"{base_path}/{subj_id}/bgpdwis_PA_geomcorr.bval")
                    bvecs_plot  = np.loadtxt(f"{base_path}/{subj_id}/bgpdwis_PA_geomcorr.bvec")

                    if bvecs_plot.shape[0] == 3 and bvecs_plot.shape[1] != 3:
                        bvecs_plot = bvecs_plot.T

                    folder_debug = f"debug_images/{dic_config['job_id']}"
                    os.makedirs(folder_debug, exist_ok=True)

                    save_debug_documentation_png(
                        neighbors, target, output_final, res_predito,
                        global_step, folder_debug, origin_b, target_b,
                        alpha, query, n_coords,
                    )

                    q_path = os.path.join(folder_debug, f"qplot_step_{global_step}.png")
                    plot_q_space_selection_antipodal(
                        bvals_plot, bvecs_plot, target_idx, neighbor_indices, q_path
                    )

                    p_path = os.path.join(folder_debug, f"qplot_polar_step_{global_step}.png")
                    plot_q_space_polar(bvecs_plot, target_idx, neighbor_indices, p_path)

                    print(f"📸 Debug visual salvo — sujeito {subj_id} | step {global_step}", flush=True)

                model.train()

            global_step += 1

        # ---- FINAL DA EPOCH ----
        print(f"🧪 Finalizando Epoch {epoch+1}...", flush=True)
        # Rodamos uma validação completa de patch para o scheduler
        epoch_val_logs, _ = validate(model, val_loader_patch, criterion, device)

        # --- LÓGICA DE DECISÃO DO MELHOR MODELO ---
        # Na Fase 1, o modelo ainda não consegue fazer Cross-Shell.
        # Portanto, salvamos baseados na performance de Interpolação (Same-Shell).
        if fase == 1:
            current_score = epoch_val_logs.get("val/rmse_same_shell", epoch_val_logs["val/loss"])
            metric_name = "RMSE Same"
        else:
            # Nas Fases 2 e 3, o que define a "inteligência" do modelo é o Cross-Shell
            current_score = epoch_val_logs.get("val/rmse_cross_shell", epoch_val_logs["val/loss"])
            metric_name = "RMSE Cross"
            
        # Verificação de segurança: se a métrica falhou (NaN), não salva
        if not np.isnan(current_score):
            if current_score < best_val_score:
                best_val_score = current_score
                torch.save(model.state_dict(), f"{folder_checkpoint}/model_best.pt")
                print(f"🌟 [Step {global_step}] Novo melhor ({metric_name}): {current_score:.6f}", flush=True)

                wandb.log({
                    "val/best_score_track": best_val_score,
                    "meta/fase_atual": fase,
                    "global_step": global_step
                })


        # Escolha da métrica para o Scheduler
        if fase == 1:
            sched_metric = epoch_val_logs.get("val/rmse_same_shell", epoch_val_logs["val/loss"])
        else:
            sched_metric = epoch_val_logs.get("val/rmse_cross_shell", epoch_val_logs["val/loss"])

        # Atualiza o Learning Rate se houver platô na métrica física
        scheduler.step(sched_metric)
        
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"meta/lr": current_lr, "epoch": epoch + 1})

        # Checkpoint por época (opcional, para histórico)
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f"{folder_checkpoint}/epoch_{epoch+1}.pt")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Iniciando...", flush=True)

    dic_config = init_config("train_config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=str, default='local')
    args = parser.parse_args()

    dic_config['job_id'] = args.job_id
    print(f"Configurações finais: {dic_config}", flush=True)

    if dic_config:
        rs = setup_environment(dic_config)
        print("🚀 Iniciando o treinamento...", flush=True)
        train(dic_config)
        wandb.finish()