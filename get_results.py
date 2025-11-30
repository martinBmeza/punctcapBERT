import torch
import yaml
import pandas as pd

from src.data.dataset import TokenChunkDataset, collate_fn_pad
from src.models.GRU import RNNModel
from torch.utils.data import DataLoader

import numpy as np
from torch.nn.functional import softmax, sigmoid

# INPUT
model_cfg = "configs/wiki_BiLSTM.yaml" 
model_pt = 'results/wiki_BiLSTM-20251129_052431/best_model.pt'



def extract_predictions(model, dataloader, device='cpu', ignore_index=-100):
    model.to(device)
    model.eval()

    # Collections for each task
    all_true = {'p_ini': [], 'p_fin': [], 'cap': []}
    all_pred = {'p_ini': [], 'p_fin': [], 'cap': []}
    all_proba = {'p_ini': [], 'p_fin': [], 'cap': []}

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to device
            bert_embeddings = batch["bert_embeddings"].to(device)
            attention_mask = batch["attention_mask"].to(device)  # shape (B, T)
            labels_p_ini = batch["labels_p_ini"].to(device)
            labels_p_fin = batch["labels_p_fin"].to(device)
            labels_cap = batch["labels_cap"].to(device)

            outputs = model(bert_embeddings, attention_mask=attention_mask)

            # Binary p_ini (sigmoid)
            logits_ini = outputs["logits_p_ini"].to(device)  # (B, T)
            probs_ini = sigmoid(logits_ini)  # (B, T) in [0,1]
            preds_ini = (probs_ini >= 0.5).long()  # (B,T)

            # Multiclass p_fin & cap (softmax)
            logits_fin = outputs["logits_p_fin"].to(device)  # (B, T, C)
            probs_fin = softmax(logits_fin, dim=-1)          # (B, T, C)
            preds_fin = logits_fin.argmax(dim=-1)            # (B, T)

            logits_cap = outputs["logits_cap"].to(device)
            probs_cap = softmax(logits_cap, dim=-1)
            preds_cap = logits_cap.argmax(dim=-1)

            # Flatten only valid tokens and (for p_fin/cap) ignore ignore_index
            mask = attention_mask.bool()  # True where tokens are valid

            # p_ini: use mask
            valid_mask_ini = mask
            valid_true_ini = labels_p_ini[valid_mask_ini].cpu().numpy()
            valid_pred_ini = preds_ini[valid_mask_ini].cpu().numpy()
            valid_proba_ini = probs_ini[valid_mask_ini].cpu().numpy()  # floats

            all_true['p_ini'].extend(valid_true_ini.tolist())
            all_pred['p_ini'].extend(valid_pred_ini.tolist())
            all_proba['p_ini'].extend(valid_proba_ini.tolist())

            # p_fin: mask + labels != ignore_index
            valid_mask_fin = mask & (labels_p_fin != ignore_index)
            if valid_mask_fin.any():
                valid_true_fin = labels_p_fin[valid_mask_fin].cpu().numpy()
                valid_pred_fin = preds_fin[valid_mask_fin].cpu().numpy()
                valid_proba_fin = probs_fin[valid_mask_fin].cpu().numpy()  # (N, C)
                all_true['p_fin'].extend(valid_true_fin.tolist())
                all_pred['p_fin'].extend(valid_pred_fin.tolist())
                all_proba['p_fin'].extend(valid_proba_fin.tolist())

            # cap: mask & labels != ignore_index (if cap uses ignore_index)
            valid_mask_cap = mask & (labels_cap != ignore_index)
            if valid_mask_cap.any():
                valid_true_cap = labels_cap[valid_mask_cap].cpu().numpy()
                valid_pred_cap = preds_cap[valid_mask_cap].cpu().numpy()
                valid_proba_cap = probs_cap[valid_mask_cap].cpu().numpy()
                all_true['cap'].extend(valid_true_cap.tolist())
                all_pred['cap'].extend(valid_pred_cap.tolist())
                all_proba['cap'].extend(valid_proba_cap.tolist())

    # Convert lists to numpy arrays for convenience
    for k in all_true.keys():
        all_true[k] = np.array(all_true[k])
        all_pred[k] = np.array(all_pred[k])
        all_proba[k] = np.array(all_proba[k])

    return all_true, all_pred, all_proba





# cfg
with open(model_cfg, "r") as f:
    config = yaml.safe_load(f)


# trained model
loaded_model = RNNModel(
  **config.get("model_params")
)
loaded_model.load_state_dict(torch.load(model_pt), map_location=torch.device('cpu') ) # Loading the state dictionary
loaded_model.eval() # Set to evaluation mode


# validation dataset
val_dataset = TokenChunkDataset(config.get('validation_metadata'))
val_dataloader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=4, collate_fn=collate_fn_pad)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_true, all_pred, all_proba = extract_predictions(loaded_model, val_dataloader, device=device)

# Example: inspect shapes
print("p_ini:", all_true['p_ini'].shape, all_pred['p_ini'].shape, all_proba['p_ini'].shape)     # () -> (N,)
print("p_fin:", all_true['p_fin'].shape, all_pred['p_fin'].shape, all_proba['p_fin'].shape)     # proba -> (N, C)
print("cap  :", all_true['cap'].shape, all_pred['cap'].shape, all_proba['cap'].shape)


df_cap = pd.DataFrame()
df_cap["y_cap"] = all_true["cap"]
df_cap["y_pred"] = all_pred["cap"]
df_cap["y_proba_0"] = all_proba["cap"][:,0]
df_cap["y_proba_1"] = all_proba["cap"][:,1]
df_cap["y_proba_2"] = all_proba["cap"][:,2]
df_cap["y_proba_3"] = all_proba["cap"][:,3]
df_cap.to_csv('data/processed/test_cap_rnn.csv', index=False)



df_p_ini = pd.DataFrame()
df_p_ini["y_punt_ini"] = all_true["p_ini"]
df_p_ini["y_pred"] = all_pred["p_ini"]
df_p_ini["y_proba_0"] = 1 - all_proba["p_ini"]
df_p_ini["y_proba_1"] = all_proba["p_ini"]
df_p_ini.to_csv('data/processed/test_ini_rnn.csv', index=False)


df_p_fin = pd.DataFrame()
df_p_fin["y_punt_fin"] = all_true["p_fin"]
df_p_fin["y_pred"] = all_pred["p_fin"]
df_p_fin["y_proba_0"] = all_proba["p_fin"][:,0]
df_p_fin["y_proba_1"] = all_proba["p_fin"][:,1]
df_p_fin["y_proba_2"] = all_proba["p_fin"][:,2]
df_p_fin["y_proba_3"] = all_proba["p_fin"][:,3]
df_p_fin.to_csv('data/processed/test_fin_rnn.csv', index=False)
