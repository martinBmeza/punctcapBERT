import argparse
import math
import os
import yaml

from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformers import BertTokenizerFast

from src.data.dataset import TokenChunkDataset, collate_fn_pad
# TODO: cambiar por un import dinamico segun config
from src.models.simpleRNN import SimpleRNNModel


def compute_losses_and_metrics(outputs, batch, device, ignore_index=-100, pos_weight=None):
    """
    outputs: dict of logits from model
    batch: dict from collate_fn
    returns: (total_loss, dict_metrics)
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

    # p_fin and cap: CrossEntropyLoss with ignore_index
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
    B, T, C = logits_p_fin.shape
    loss_p_fin = ce_loss(logits_p_fin.view(B * T, C), labels_p_fin.view(B * T))
    _, Cc = logits_cap.shape[2], logits_cap.shape[2]
    loss_cap = ce_loss(logits_cap.view(B * T, Cc), labels_cap.view(B * T))

    # average by valid tokens
    loss_p_fin = loss_p_fin / (valid_tokens + 1e-8)
    loss_cap = loss_cap / (valid_tokens + 1e-8)

    # simple weighting: equal weight
    total_loss = loss_p_ini + loss_p_fin + loss_cap

    # metrics (simple accuracy per task on valid tokens)
    with torch.no_grad():
        probs_ini = torch.sigmoid(logits_p_ini)
        preds_ini = (probs_ini >= 0.5).long()
        # flatten only valid tokens
        mask = attention_mask.bool()
        true_ini = labels_p_ini.long()

        acc_ini = (preds_ini[mask] == true_ini[mask]).float().mean().item() if mask.sum() > 0 else 0.0

        # p_fin
        preds_p_fin = logits_p_fin.argmax(dim=-1)
        valid_idx = mask & (labels_p_fin != ignore_index)
        acc_fin = (preds_p_fin[valid_idx] == labels_p_fin[valid_idx]).float().mean().item() if valid_idx.sum() > 0 else 0.0

        # cap
        preds_cap = logits_cap.argmax(dim=-1)
        valid_idx2 = mask & (labels_cap != ignore_index)
        acc_cap = (preds_cap[valid_idx2] == labels_cap[valid_idx2]).float().mean().item() if valid_idx2.sum() > 0 else 0.0

    metrics = {
        "loss_p_ini": float(loss_p_ini.item()),
        "loss_p_fin": float(loss_p_fin.item()),
        "loss_cap": float(loss_cap.item()),
        "total_loss": float(total_loss.item()),
        "acc_p_ini": acc_ini,
        "acc_p_fin": acc_fin,
        "acc_cap": acc_cap
    }
    return total_loss, metrics


def train_one_epoch(model, dataloader, optimizer, device, epoch, log_every=50):
    model.train()
    running_loss = 0.0
    n_steps = 0
    agg_metrics = {"total_loss": 0.0, "acc_p_ini": 0.0, "acc_p_fin": 0.0, "acc_cap": 0.0}
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train epoch {epoch}")
    for step, batch in pbar:
        optimizer.zero_grad()
        token_ids = batch["token_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(token_ids, attention_mask=attention_mask)
        loss, metrics = compute_losses_and_metrics(outputs, batch, device)

        loss.backward()
        optimizer.step()

        running_loss += metrics["total_loss"]
        n_steps += 1
        agg_metrics["total_loss"] += metrics["total_loss"]
        agg_metrics["acc_p_ini"] += metrics["acc_p_ini"]
        agg_metrics["acc_p_fin"] += metrics["acc_p_fin"]
        agg_metrics["acc_cap"] += metrics["acc_cap"]

        if (step + 1) % log_every == 0:
            pbar.set_postfix({
                "loss": f"{agg_metrics['total_loss'] / n_steps:.4f}",
                "acc_ini": f"{agg_metrics['acc_p_ini'] / n_steps:.3f}",
                "acc_fin": f"{agg_metrics['acc_p_fin'] / n_steps:.3f}",
                "acc_cap": f"{agg_metrics['acc_cap'] / n_steps:.3f}",
            })

    # epoch averages
    avg_metrics = {k: (v / n_steps) if n_steps > 0 else 0.0 for k, v in agg_metrics.items()}
    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cpu'))
    print("Using device:", device)

    # load tokenizer to obtain vocab_size
    tokenizer = BertTokenizerFast.from_pretrained(config.get('tokenizer', 'bert-base-multilingual-cased'))
    vocab_size = tokenizer.vocab_size
    print("Tokenizer vocab size:", vocab_size)

    dataset = TokenChunkDataset(config.get('metadata'))
    if config.get('max_examples', 0) > 0:
        # shrink dataset for quick debug
        dataset.paths = dataset.paths[: config['max_examples']]
        dataset.df = dataset.df.iloc[: config['max_examples']].reset_index(drop=True)

    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 32), shuffle=True, num_workers=4,
                            collate_fn=collate_fn_pad)

    model = SimpleRNNModel(vocab_size=vocab_size, embed_dim=config.get('embed_dim', 256), hidden_dim=config.get('hidden_dim', 512))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    config_name = os.path.basename(args.config).replace('.yaml', '')
    run_name = '-'.join([config_name, datetime.now().strftime("%Y%m%d_%H%M%S")])
    out_dir = os.path.join('results', run_name)
    os.makedirs(out_dir, exist_ok=True)

    best_loss = math.inf
    for epoch in range(1, config.get('epochs', 10) + 1):
        epoch_metrics = train_one_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch} summary:", epoch_metrics)

        # save checkpoint
        if config.get('store_checkpoints', False):
            ckpt_path = os.path.join(out_dir, f"rnn_epoch{epoch}_loss{epoch_metrics['total_loss']:.4f}.pt")
            torch.save({
                "epoch": epoch, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_metrics,
                "vocab_size": vocab_size,
                "args": vars(args)
            }, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

        if epoch_metrics["total_loss"] < best_loss:
            best_loss = epoch_metrics["total_loss"]
            best_path = os.path.join(out_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print("Saved best model to", best_path)


if __name__ == "__main__":
    main()