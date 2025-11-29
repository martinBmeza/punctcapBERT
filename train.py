import argparse
import math
import os
import yaml

from pathlib import Path
from datetime import datetime

import torch
import importlib
import numpy as np
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import BertTokenizerFast

from src.data.dataset import TokenChunkDataset, collate_fn_pad



def balanced_accuracy_binary(y_true, y_pred):
    """
    Compute balanced accuracy for binary classification
    """
    # Convert to numpy for easier computation
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Calculate true positive rate (sensitivity/recall) and true negative rate (specificity)
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        # If only one class present, return regular accuracy
        return (y_true == y_pred).mean()
    
    tpr = (y_pred[pos_mask] == 1).mean()  # True positive rate
    tnr = (y_pred[neg_mask] == 0).mean()  # True negative rate
    
    return (tpr + tnr) / 2.0


def balanced_accuracy_multiclass(y_true, y_pred, num_classes):
    """
    Compute balanced accuracy for multiclass classification
    """
    # Convert to numpy for easier computation
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    recalls = []
    for class_id in range(num_classes):
        class_mask = (y_true == class_id)
        if class_mask.sum() == 0:
            continue  # Skip classes not present in this batch
        
        # Recall for this class
        recall = (y_pred[class_mask] == class_id).mean()
        recalls.append(recall)
    
    if len(recalls) == 0:
        return 0.0
    
    return sum(recalls) / len(recalls)


def compute_class_weights(dataloader, device, ignore_index=-100):
    """
    Compute class weights based on inverse class frequencies
    
    Returns:
        pos_weight: Positive class weight for p_ini (scalar tensor)
        class_weights_p_fin: Class weights for p_fin (tensor of size 4)
        class_weights_cap: Class weights for cap (tensor of size 4)
    """
    print("Computing class weights from training data...")
    
    # Count classes
    count_p_ini = torch.zeros(2)  # [negative, positive]
    count_p_fin = torch.zeros(4)  # 4 classes
    count_cap = torch.zeros(4)    # 4 classes
    
    total_valid_tokens = 0
    
    for batch in tqdm(dataloader, desc="Computing class frequencies"):
        attention_mask = batch["attention_mask"]
        labels_p_ini = batch["labels_p_ini"]
        labels_p_fin = batch["labels_p_fin"]
        labels_cap = batch["labels_cap"]
        
        # Only count valid tokens
        mask = attention_mask.bool()
        
        # Count p_ini (binary)
        p_ini_labels = labels_p_ini[mask].long()
        count_p_ini[0] += (p_ini_labels == 0).sum().item()
        count_p_ini[1] += (p_ini_labels == 1).sum().item()
        
        # Count p_fin (multiclass)
        p_fin_mask = mask & (labels_p_fin != ignore_index)
        p_fin_labels = labels_p_fin[p_fin_mask]
        for i in range(4):
            count_p_fin[i] += (p_fin_labels == i).sum().item()
        
        # Count cap (multiclass)
        cap_mask = mask & (labels_cap != ignore_index)
        cap_labels = labels_cap[cap_mask]
        for i in range(4):
            count_cap[i] += (cap_labels == i).sum().item()
        
        total_valid_tokens += mask.sum().item()
    
    print(f"Total valid tokens: {total_valid_tokens}")
    print(f"P_ini class counts: {count_p_ini.tolist()}")
    print(f"P_fin class counts: {count_p_fin.tolist()}")
    print(f"Cap class counts: {count_cap.tolist()}")
    
    # Compute weights (inverse frequency)
    # For p_ini: pos_weight = negative_count / positive_count
    pos_weight = count_p_ini[0] / (count_p_ini[1] + 1e-8)
    pos_weight = pos_weight.to(device)
    
    # For multiclass: weight = total_samples / (n_classes * class_count)
    total_p_fin = count_p_fin.sum()
    class_weights_p_fin = total_p_fin / (4 * count_p_fin + 1e-8)
    class_weights_p_fin = class_weights_p_fin.to(device)
    
    total_cap = count_cap.sum()
    class_weights_cap = total_cap / (4 * count_cap + 1e-8)
    class_weights_cap = class_weights_cap.to(device)
    
    print(f"P_ini pos_weight: {pos_weight.item():.4f}")
    print(f"P_fin class weights: {class_weights_p_fin.tolist()}")
    print(f"Cap class weights: {class_weights_cap.tolist()}")
    
    return pos_weight, class_weights_p_fin, class_weights_cap


def compute_losses_and_metrics(outputs, batch, device, ignore_index=-100, pos_weight=None, class_weights_p_fin=None, class_weights_cap=None):
    """
    outputs: dict of logits from model
    batch: dict from collate_fn
    returns: (total_loss, dict_metrics)
    
    Args:
        pos_weight: Positive class weight for p_ini binary classification
        class_weights_p_fin: Class weights for p_fin multiclass classification (tensor of size 4)
        class_weights_cap: Class weights for cap multiclass classification (tensor of size 4)
    """
    logits_p_ini = outputs["logits_p_ini"].to(device)  # (B,T)
    logits_p_fin = outputs["logits_p_fin"].to(device)  # (B,T,4)
    logits_cap = outputs["logits_cap"].to(device)      # (B,T,4)

    labels_p_ini = batch["labels_p_ini"].to(device)    # (B,T) float
    labels_p_fin = batch["labels_p_fin"].to(device)    # (B,T) long, padded with ignore_index
    labels_cap = batch["labels_cap"].to(device)        # (B,T) long

    attention_mask = batch["attention_mask"].to(device).float()  # (B,T) 1/0
    valid_tokens = attention_mask.sum()

    # p_ini: BCEWithLogitsLoss reduction='none', mask and average
    bce_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss_p_ini_elem = bce_loss(logits_p_ini, labels_p_ini)  # (B,T)
    loss_p_ini = (loss_p_ini_elem * attention_mask).sum() / (valid_tokens + 1e-8)

    # p_fin and cap: CrossEntropyLoss with ignore_index and class weights
    ce_loss_p_fin = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum", weight=class_weights_p_fin)
    ce_loss_cap = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum", weight=class_weights_cap)
    
    B, T, C = logits_p_fin.shape
    loss_p_fin = ce_loss_p_fin(logits_p_fin.view(B * T, C), labels_p_fin.view(B * T))
    _, Cc = logits_cap.shape[2], logits_cap.shape[2]
    loss_cap = ce_loss_cap(logits_cap.view(B * T, Cc), labels_cap.view(B * T))

    # average by valid tokens
    loss_p_fin = loss_p_fin / (valid_tokens + 1e-8)
    loss_cap = loss_cap / (valid_tokens + 1e-8)

    # simple weighting: equal weight
    total_loss = loss_p_ini + loss_p_fin + loss_cap

    # metrics (balanced accuracy per task on valid tokens)
    with torch.no_grad():
        probs_ini = torch.sigmoid(logits_p_ini)
        preds_ini = (probs_ini >= 0.5).long()
        # flatten only valid tokens
        mask = attention_mask.bool()
        true_ini = labels_p_ini.long()

        # Balanced accuracy for p_ini (binary classification)
        if mask.sum() > 0:
            balanced_acc_ini = balanced_accuracy_binary(true_ini[mask], preds_ini[mask])
        else:
            balanced_acc_ini = 0.0

        # p_fin (multiclass)
        preds_p_fin = logits_p_fin.argmax(dim=-1)
        valid_idx = mask & (labels_p_fin != ignore_index)
        if valid_idx.sum() > 0:
            balanced_acc_fin = balanced_accuracy_multiclass(labels_p_fin[valid_idx], preds_p_fin[valid_idx], 4)
        else:
            balanced_acc_fin = 0.0

        # cap (multiclass)
        preds_cap = logits_cap.argmax(dim=-1)
        valid_idx2 = mask & (labels_cap != ignore_index)
        if valid_idx2.sum() > 0:
            balanced_acc_cap = balanced_accuracy_multiclass(labels_cap[valid_idx2], preds_cap[valid_idx2], 4)
        else:
            balanced_acc_cap = 0.0

    metrics = {
        "loss_p_ini": float(loss_p_ini.item()),
        "loss_p_fin": float(loss_p_fin.item()),
        "loss_cap": float(loss_cap.item()),
        "total_loss": float(total_loss.item()),
        "balanced_acc_p_ini": balanced_acc_ini,
        "balanced_acc_p_fin": balanced_acc_fin,
        "balanced_acc_cap": balanced_acc_cap
    }
    return total_loss, metrics


def train_one_epoch(model, dataloader, optimizer, device, epoch, log_every=50, use_wandb=False, wandb_log_steps=False, 
                   pos_weight=None, class_weights_p_fin=None, class_weights_cap=None):
    model.train()
    running_loss = 0.0
    n_steps = 0
    agg_metrics = {"total_loss": 0.0, "balanced_acc_p_ini": 0.0, "balanced_acc_p_fin": 0.0, "balanced_acc_cap": 0.0}
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train epoch {epoch}")
    for step, batch in pbar:
        optimizer.zero_grad()
        bert_embeddings = batch["bert_embeddings"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(bert_embeddings, attention_mask=attention_mask)
        loss, metrics = compute_losses_and_metrics(outputs, batch, device, 
                                                 pos_weight=pos_weight,
                                                 class_weights_p_fin=class_weights_p_fin,
                                                 class_weights_cap=class_weights_cap)

        loss.backward()
        optimizer.step()

        running_loss += metrics["total_loss"]
        n_steps += 1
        agg_metrics["total_loss"] += metrics["total_loss"]
        agg_metrics["balanced_acc_p_ini"] += metrics["balanced_acc_p_ini"]
        agg_metrics["balanced_acc_p_fin"] += metrics["balanced_acc_p_fin"]
        agg_metrics["balanced_acc_cap"] += metrics["balanced_acc_cap"]

        # Log step metrics to wandb (only if enabled)
        if use_wandb and wandb_log_steps:
            wandb.log({
                "step_loss": metrics["total_loss"],
                "step_loss_p_ini": metrics["loss_p_ini"],
                "step_loss_p_fin": metrics["loss_p_fin"],
                "step_loss_cap": metrics["loss_cap"],
                "step_balanced_acc_p_ini": metrics["balanced_acc_p_ini"],
                "step_balanced_acc_p_fin": metrics["balanced_acc_p_fin"],
                "step_balanced_acc_cap": metrics["balanced_acc_cap"],
                "epoch": epoch,
                "step": step + 1
            })

        if (step + 1) % log_every == 0:
            pbar.set_postfix({
                "loss": f"{agg_metrics['total_loss'] / n_steps:.4f}",
                "bal_acc_ini": f"{agg_metrics['balanced_acc_p_ini'] / n_steps:.3f}",
                "bal_acc_fin": f"{agg_metrics['balanced_acc_p_fin'] / n_steps:.3f}",
                "bal_acc_cap": f"{agg_metrics['balanced_acc_cap'] / n_steps:.3f}",
            })

    # epoch averages
    avg_metrics = {k: (v / n_steps) if n_steps > 0 else 0.0 for k, v in agg_metrics.items()}
    return avg_metrics


def create_confusion_matrices(all_preds, all_labels, class_names_dict):
    """
    Create confusion matrices for each task using seaborn heatmaps and log to wandb
    
    Args:
        all_preds: dict with keys 'p_ini', 'p_fin', 'cap' containing predictions
        all_labels: dict with keys 'p_ini', 'p_fin', 'cap' containing true labels  
        class_names_dict: dict with class names for each task
    
    Returns:
        dict with wandb image objects for each confusion matrix
    """
    wandb_cms = {}
    
    # Set seaborn style for better looking plots
    sns.set_style("whitegrid")
    
    # Punctuation initial (binary)
    if len(all_preds['p_ini']) > 0 and len(all_labels['p_ini']) > 0:
        cm_p_ini = confusion_matrix(all_labels['p_ini'], all_preds['p_ini'])
        
        # Create seaborn heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_p_ini, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names_dict['p_ini'],
                   yticklabels=class_names_dict['p_ini'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Punctuation Initial')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Log to wandb as image
        wandb_cms['confusion_matrix_p_ini'] = wandb.Image(plt)
        plt.close()
    
    # Punctuation final (multiclass)  
    if len(all_preds['p_fin']) > 0 and len(all_labels['p_fin']) > 0:
        cm_p_fin = confusion_matrix(all_labels['p_fin'], all_preds['p_fin'])
        
        # Create seaborn heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_p_fin, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names_dict['p_fin'],
                   yticklabels=class_names_dict['p_fin'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Punctuation Final')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Log to wandb as image
        wandb_cms['confusion_matrix_p_fin'] = wandb.Image(plt)
        plt.close()
        
    # Capitalization (multiclass)
    if len(all_preds['cap']) > 0 and len(all_labels['cap']) > 0:
        cm_cap = confusion_matrix(all_labels['cap'], all_preds['cap'])
        
        # Create seaborn heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_cap, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_names_dict['cap'],
                   yticklabels=class_names_dict['cap'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Capitalization')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Log to wandb as image
        wandb_cms['confusion_matrix_cap'] = wandb.Image(plt)
        plt.close()
    
    return wandb_cms


def evaluate_one_epoch(model, dataloader, device, pos_weight=None, class_weights_p_fin=None, class_weights_cap=None, 
                      create_confusion_matrices_flag=False, ignore_index=-100):
    """
    Evaluate model on validation/test set
    
    Args:
        create_confusion_matrices_flag: If True, collect predictions for confusion matrices
    """
    model.eval()
    running_loss = 0.0
    n_steps = 0
    agg_metrics = {"total_loss": 0.0, "balanced_acc_p_ini": 0.0, "balanced_acc_p_fin": 0.0, "balanced_acc_cap": 0.0}
    
    # For confusion matrices
    all_preds = {'p_ini': [], 'p_fin': [], 'cap': []} if create_confusion_matrices_flag else None
    all_labels = {'p_ini': [], 'p_fin': [], 'cap': []} if create_confusion_matrices_flag else None
    
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")
        for step, batch in pbar:
            bert_embeddings = batch["bert_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(bert_embeddings, attention_mask=attention_mask)
            loss, metrics = compute_losses_and_metrics(outputs, batch, device, 
                                                     pos_weight=pos_weight,
                                                     class_weights_p_fin=class_weights_p_fin,
                                                     class_weights_cap=class_weights_cap)

            running_loss += metrics["total_loss"]
            n_steps += 1
            agg_metrics["total_loss"] += metrics["total_loss"]
            agg_metrics["balanced_acc_p_ini"] += metrics["balanced_acc_p_ini"]
            agg_metrics["balanced_acc_p_fin"] += metrics["balanced_acc_p_fin"]
            agg_metrics["balanced_acc_cap"] += metrics["balanced_acc_cap"]

            # Collect predictions for confusion matrices
            if create_confusion_matrices_flag:
                # Get predictions
                logits_p_ini = outputs["logits_p_ini"].to(device)
                logits_p_fin = outputs["logits_p_fin"].to(device)
                logits_cap = outputs["logits_cap"].to(device)
                
                labels_p_ini = batch["labels_p_ini"].to(device)
                labels_p_fin = batch["labels_p_fin"].to(device)
                labels_cap = batch["labels_cap"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Binary predictions for p_ini
                probs_ini = torch.sigmoid(logits_p_ini)
                preds_ini = (probs_ini >= 0.5).long()
                mask = attention_mask.bool()
                true_ini = labels_p_ini.long()
                
                # Only collect valid tokens
                valid_preds_ini = preds_ini[mask].cpu().numpy()
                valid_labels_ini = true_ini[mask].cpu().numpy()
                all_preds['p_ini'].extend(valid_preds_ini)
                all_labels['p_ini'].extend(valid_labels_ini)
                
                # Multiclass predictions for p_fin
                preds_p_fin = logits_p_fin.argmax(dim=-1)
                valid_idx_fin = mask & (labels_p_fin != ignore_index)
                valid_preds_fin = preds_p_fin[valid_idx_fin].cpu().numpy()
                valid_labels_fin = labels_p_fin[valid_idx_fin].cpu().numpy()
                all_preds['p_fin'].extend(valid_preds_fin)
                all_labels['p_fin'].extend(valid_labels_fin)
                
                # Multiclass predictions for cap
                preds_cap = logits_cap.argmax(dim=-1)
                valid_idx_cap = mask & (labels_cap != ignore_index)
                valid_preds_cap = preds_cap[valid_idx_cap].cpu().numpy()
                valid_labels_cap = labels_cap[valid_idx_cap].cpu().numpy()
                all_preds['cap'].extend(valid_preds_cap)
                all_labels['cap'].extend(valid_labels_cap)

            pbar.set_postfix({
                "val_loss": f"{agg_metrics['total_loss'] / n_steps:.4f}",
                "val_bal_acc_ini": f"{agg_metrics['balanced_acc_p_ini'] / n_steps:.3f}",
                "val_bal_acc_fin": f"{agg_metrics['balanced_acc_p_fin'] / n_steps:.3f}",
                "val_bal_acc_cap": f"{agg_metrics['balanced_acc_cap'] / n_steps:.3f}",
            })

    # epoch averages
    avg_metrics = {k: (v / n_steps) if n_steps > 0 else 0.0 for k, v in agg_metrics.items()}
    
    # Add confusion matrices if requested
    if create_confusion_matrices_flag:
        # Define class names
        class_names_dict = {
            'p_ini': ['No Punctuation', 'Punctuation'],
            'p_fin': ['None', 'Comma', 'Period', 'Question'],  # Adjust based on your actual classes
            'cap': ['Lowercase', 'Uppercase', 'Title', 'Other']  # Adjust based on your actual classes  
        }
        
        confusion_matrices = create_confusion_matrices(all_preds, all_labels, class_names_dict)
        avg_metrics.update(confusion_matrices)
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cpu'))
    print("Using device:", device)

    # Initialize wandb if enabled
    use_wandb = config.get('use_wandb', False)
    wandb_log_steps = config.get('wandb_log_steps', False) 
    if use_wandb:
        wandb.init(
            project=config.get('wandb_project', 'punctcapBERT'),
            name=config.get('wandb_run_name', None),
            config=config,
            tags=config.get('wandb_tags', []),
            notes=config.get('wandb_notes', '')
        )
        print("Initialized wandb logging")
        if wandb_log_steps:
            print("Step-level logging enabled")
        else:
            print("Only epoch-level logging enabled")

    # Since we're using pre-computed BERT embeddings, we don't need vocab_size
    # tokenizer = BertTokenizerFast.from_pretrained(config.get('tokenizer', 'bert-base-multilingual-cased'))
    # vocab_size = tokenizer.vocab_size
    # print("Tokenizer vocab size:", vocab_size)
    vocab_size = None  # Not needed for pre-computed embeddings

    # Load training dataset
    dataset = TokenChunkDataset(config.get('metadata'))
    if config.get('max_examples', 0) > 0:
        # shrink dataset for quick debug
        dataset.paths = dataset.paths[: config['max_examples']]
        dataset.df = dataset.df.iloc[: config['max_examples']].reset_index(drop=True)

    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=4,
                            collate_fn=collate_fn_pad)

    # Load validation dataset if specified
    val_dataloader = None
    if config.get('validation_metadata'):
        print(f"Loading validation dataset from: {config['validation_metadata']}")
        val_dataset = TokenChunkDataset(config.get('validation_metadata'))
        val_dataloader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=4,
                                   collate_fn=collate_fn_pad)
        print(f"Validation dataset size: {len(val_dataset)} chunks")

    print(f"Training dataset size: {len(dataset)} chunks")

    # Compute class weights if enabled
    use_class_weights = config.get('use_class_weights', False)
    pos_weight, class_weights_p_fin, class_weights_cap = None, None, None
    
    if use_class_weights:
        pos_weight, class_weights_p_fin, class_weights_cap = compute_class_weights(dataloader, device)
        
        # Log class weights to wandb if enabled
        if use_wandb:
            wandb.log({
                "pos_weight_p_ini": pos_weight.item(),
                "class_weights_p_fin": class_weights_p_fin.tolist(),
                "class_weights_cap": class_weights_cap.tolist()
            })

    module = importlib.import_module(f'src.models.{config.get("model_name")}')
    model = module.RNNModel(**config.get('model_params', {}))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    config_name = os.path.basename(args.config).replace('.yaml', '')
    run_name = '-'.join([config_name, datetime.now().strftime("%Y%m%d_%H%M%S")])
    out_dir = os.path.join('results', run_name)
    os.makedirs(out_dir, exist_ok=True)

    best_loss = math.inf
    best_val_loss = math.inf
    evaluate_every = config.get('evaluate_every_n_epochs', 1)
    
    for epoch in range(1, config.get('epochs', 10) + 1):
        # Training
        epoch_metrics = train_one_epoch(model, dataloader, optimizer, device, epoch, 
                                      use_wandb=use_wandb, wandb_log_steps=wandb_log_steps,
                                      pos_weight=pos_weight, 
                                      class_weights_p_fin=class_weights_p_fin,
                                      class_weights_cap=class_weights_cap)
        print(f"Epoch {epoch} TRAIN summary:", epoch_metrics)

        # Validation (if available and it's time to evaluate)
        val_metrics = None
        if val_dataloader is not None and epoch % evaluate_every == 0:
            # Create confusion matrices every N epochs (configurable)
            create_confusion_matrices_flag = config.get('log_confusion_matrices', True) and epoch % config.get('confusion_matrix_every_n_epochs', 5) == 0
            
            val_metrics = evaluate_one_epoch(model, val_dataloader, device,
                                           pos_weight=pos_weight,
                                           class_weights_p_fin=class_weights_p_fin, 
                                           class_weights_cap=class_weights_cap,
                                           create_confusion_matrices_flag=create_confusion_matrices_flag)
            print(f"Epoch {epoch} VAL summary:", val_metrics)

        # Log epoch metrics to wandb
        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": epoch_metrics["total_loss"],
                "train_balanced_acc_p_ini": epoch_metrics["balanced_acc_p_ini"],
                "train_balanced_acc_p_fin": epoch_metrics["balanced_acc_p_fin"],
                "train_balanced_acc_cap": epoch_metrics["balanced_acc_cap"],
                "best_train_loss": best_loss
            }
            
            # Add validation metrics if available
            if val_metrics is not None:
                # Regular metrics
                val_update = {
                    "val_loss": val_metrics["total_loss"],
                    "val_balanced_acc_p_ini": val_metrics["balanced_acc_p_ini"],
                    "val_balanced_acc_p_fin": val_metrics["balanced_acc_p_fin"],
                    "val_balanced_acc_cap": val_metrics["balanced_acc_cap"],
                    "best_val_loss": best_val_loss
                }
                log_dict.update(val_update)
                
                # Add confusion matrices if they exist
                cm_keys = ['confusion_matrix_p_ini', 'confusion_matrix_p_fin', 'confusion_matrix_cap']
                for cm_key in cm_keys:
                    if cm_key in val_metrics:
                        log_dict[cm_key] = val_metrics[cm_key]
            
            wandb.log(log_dict)

        # save checkpoint
        if config.get('store_checkpoints', False):
            ckpt_path = os.path.join(out_dir, f"rnn_epoch{epoch}_loss{epoch_metrics['total_loss']:.4f}.pt")
            torch.save({
                "epoch": epoch, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": epoch_metrics,
                "val_metrics": val_metrics,
                "vocab_size": vocab_size,
                "args": vars(args)
            }, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

        # Update best models based on validation loss (if available) or training loss
        current_loss = val_metrics["total_loss"] if val_metrics is not None else epoch_metrics["total_loss"]
        loss_type = "val" if val_metrics is not None else "train"
        
        if current_loss < (best_val_loss if val_metrics is not None else best_loss):
            if val_metrics is not None:
                best_val_loss = current_loss
            else:
                best_loss = current_loss
                
            best_path = os.path.join(out_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path} (best {loss_type} loss: {current_loss:.4f})")
            
            # Log best model to wandb
            if use_wandb:
                wandb.log({
                    "best_epoch": epoch, 
                    "best_val_loss" if val_metrics is not None else "best_train_loss": current_loss
                })

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()