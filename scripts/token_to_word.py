"""
Ejemplo corto: convertir predicciones por token (logits) en predicciones a nivel palabra
usando batch["word_ids"] y batch["original_texts"] devueltos por el DataLoader.

Funciones incluidas:
 - token_preds_from_logits(outputs, attention_mask, threshold=0.5)
 - aggregate_token_preds_to_word_level(token_preds, word_ids)
 - detokenize_word_from_subtokens(tokens, token_positions)
 - write_token_level_csv_for_batch(...)

Uso rápido (pseudocódigo):
    outputs = model(token_ids, attention_mask=attention_mask)
    token_preds = token_preds_from_logits(outputs, attention_mask)
    word_preds = aggregate_token_preds_to_word_level(token_preds, batch["word_ids"])
    write_token_level_csv_for_batch(token_preds, batch, tokenizer, out_dir="results/pred_csv")
"""

from typing import Dict, Any, List, Tuple
import os
import csv
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F


def token_preds_from_logits(outputs: Dict[str, torch.Tensor],
                            attention_mask: torch.Tensor,
                            threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Convierte logits del modelo en predicciones por token (numpy arrays).
    - outputs keys: logits_p_ini (B,T), logits_p_fin (B,T,4), logits_cap (B,T,4)
    - attention_mask: (B,T) tensor with 1 for valid tokens

    Retorna dict con arrays shape (B,T):
    - pred_p_ini (0/1)
    - pred_p_fin (0..3)
    - pred_cap (0..3)
    - probs_p_ini (float)
    """
    device = outputs["logits_p_ini"].device
    with torch.no_grad():
        logits_p_ini = outputs["logits_p_ini"].detach().cpu()  # (B,T)
        logits_p_fin = outputs["logits_p_fin"].detach().cpu()  # (B,T,4)
        logits_cap = outputs["logits_cap"].detach().cpu()      # (B,T,4)

        probs_p_ini = torch.sigmoid(logits_p_ini)             # (B,T)
        pred_p_ini = (probs_p_ini >= threshold).long()        # (B,T)

        pred_p_fin = torch.argmax(logits_p_fin, dim=-1).long()  # (B,T)
        pred_cap = torch.argmax(logits_cap, dim=-1).long()      # (B,T)

        mask = attention_mask.detach().cpu().long()            # (B,T)

    # convert to numpy, but keep masked positions as -1 to indicate padding
    pred_p_ini = pred_p_ini.numpy()
    probs_p_ini = probs_p_ini.numpy()
    pred_p_fin = pred_p_fin.numpy()
    pred_cap = pred_cap.numpy()
    mask = mask.numpy()

    # set padded positions to -1 for discrete preds, and 0.0 for probs (or np.nan if preferred)
    pred_p_ini_masked = np.where(mask == 1, pred_p_ini, -1)
    probs_p_ini_masked = np.where(mask == 1, probs_p_ini, 0.0)
    pred_p_fin_masked = np.where(mask == 1, pred_p_fin, -1)
    pred_cap_masked = np.where(mask == 1, pred_cap, -1)

    return {
        "pred_p_ini": pred_p_ini_masked,
        "probs_p_ini": probs_p_ini_masked,
        "pred_p_fin": pred_p_fin_masked,
        "pred_cap": pred_cap_masked
    }


def aggregate_token_preds_to_word_level(token_preds: Dict[str, np.ndarray],
                                        word_ids_batch: np.ndarray) -> List[List[Dict[str, Any]]]:
    """
    Agrega predicciones token-wise a nivel palabra por ejemplo del batch.

    - token_preds: dict con arrays (B,T) for discrete preds ("pred_p_ini","pred_p_fin","pred_cap")
    - word_ids_batch: np.ndarray (B,T) with values >=0 for token->word mapping, -1 for padding

    Devuelve lista (batch) de listas (words_in_example) con dicts:
      {"word_index": int, "token_positions": [int,...], "pred_p_ini": 0/1, "pred_p_fin": 0..3, "pred_cap": 0..3}
    Reglas de agregación usadas:
      - punt_inicial (binary): OR sobre tokens de la palabra (si algún token tiene 1 => palabra 1)
      - punt_final (multiclass): majority vote (mode) sobre tokens; si empate usa el valor con mayor id (arbitrario)
      - capitalización (multiclass): majority vote (mode)
    """
    B, T = token_preds["pred_p_ini"].shape
    batch_word_preds = []

    for i in range(B):
        word_ids = word_ids_batch[i]  # (T,) ints, -1 for padding
        pred_p_ini = token_preds["pred_p_ini"][i]
        pred_p_fin = token_preds["pred_p_fin"][i]
        pred_cap = token_preds["pred_cap"][i]

        # collect token indices per word index
        word_to_tokenpos = defaultdict(list)
        for t, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid < 0:
                continue
            word_to_tokenpos[int(wid)].append(t)

        word_preds = []
        for wid in sorted(word_to_tokenpos.keys()):
            toks = word_to_tokenpos[wid]
            if len(toks) == 0:
                continue
            # punt_inicial: OR
            vals_ini = pred_p_ini[toks]
            word_ini = int(np.max(vals_ini)) if vals_ini.size > 0 else 0

            # punt_final: majority vote via bincount on valid labels (>=0)
            vals_fin = pred_p_fin[toks]
            vals_fin = vals_fin[vals_fin >= 0]
            if vals_fin.size == 0:
                word_fin = 0
            else:
                counts = np.bincount(vals_fin)
                # choose argmax; tie-breaker: pick the larger label (arbitrary but deterministic)
                max_count = counts.max()
                candidates = np.where(counts == max_count)[0]
                word_fin = int(candidates[-1])

            # cap: majority vote
            vals_cap = pred_cap[toks]
            vals_cap = vals_cap[vals_cap >= 0]
            if vals_cap.size == 0:
                word_cap = 0
            else:
                counts = np.bincount(vals_cap)
                max_count = counts.max()
                candidates = np.where(counts == max_count)[0]
                word_cap = int(candidates[-1])

            word_preds.append({
                "word_index": int(wid),
                "token_positions": toks,
                "pred_p_ini": word_ini,
                "pred_p_fin": word_fin,
                "pred_cap": word_cap
            })
        batch_word_preds.append(word_preds)
    return batch_word_preds


def detokenize_word_from_subtokens(subtokens: List[str], token_positions: List[int]) -> str:
    """
    Construye (aproximadamente) la forma de la palabra a partir de subtokens BERT.
    Regla simple:
      - tokens que empiezan con '##' se concatenan a la anterior sin espacio
      - otros tokens empiezan nueva pieza con espacio entre palabras (aquí se usa sólo la pieza)
    Retorna string 'word' reconstruida.
    """
    pieces = []
    for pos in token_positions:
        tok = subtokens[pos]
        if tok.startswith("##"):
            if len(pieces) == 0:
                pieces.append(tok[2:])
            else:
                pieces[-1] = pieces[-1] + tok[2:]
        else:
            pieces.append(tok)
    # join pieces without spaces (they belong to same word), but if there are multiple pieces they should be concatenated
    word = "".join(pieces)
    # some tokens use special chars like '▁' in other tokenizers; BERT uses '##' so above works
    return word


def write_token_level_csv_for_batch(token_preds: Dict[str, np.ndarray],
                                    batch: Dict[str, Any],
                                    tokenizer,
                                    out_dir: str,
                                    prefix: str = "predictions"):
    """
    Escribe archivos CSV por ejemplo del batch con formato requerido por la cátedra:
    instancia_id, token_id, token, punt_inicial, punt_final, capitalizacion

    - token_preds: dict from token_preds_from_logits (numpy arrays B,T)
    - batch: output of collate_fn (token_ids, attention_mask, word_ids, lengths, paths, original_texts)
    - tokenizer: HuggingFace tokenizer para convertir token_ids -> tokens strings
    - out_dir: folder where to write CSVs (one CSV per example)
    - prefix: filename prefix
    """
    os.makedirs(out_dir, exist_ok=True)
    B, T = token_preds["pred_p_ini"].shape

    token_ids_np = batch["token_ids"].cpu().numpy() if isinstance(batch["token_ids"], torch.Tensor) else batch["token_ids"]
    word_ids_np = batch["word_ids"].cpu().numpy() if isinstance(batch["word_ids"], torch.Tensor) else batch["word_ids"]
    attention_mask = batch["attention_mask"].cpu().numpy() if isinstance(batch["attention_mask"], torch.Tensor) else batch["attention_mask"]
    original_texts = batch.get("original_texts", [None] * B)
    paths = batch.get("paths", [None] * B)

    for i in range(B):
        inst_path = paths[i] or f"batch{i}"
        # derive instance id from path or use prefix+index
        inst_id = os.path.splitext(os.path.basename(inst_path))[0]
        csv_path = os.path.join(out_dir, f"{prefix}_{inst_id}.csv")

        tokens = tokenizer.convert_ids_to_tokens(list(token_ids_np[i]))  # list length T (includes pad ids)
        with open(csv_path, "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["instancia_id", "token_id", "token", "punt_inicial", "punt_final", "capitalizacion"])
            for t in range(T):
                if attention_mask[i, t] == 0:
                    continue  # skip padded positions
                token_id = int(token_ids_np[i, t])
                token_str = tokens[t]
                punt_ini = int(token_preds["pred_p_ini"][i, t])
                punt_fin = int(token_preds["pred_p_fin"][i, t])
                cap = int(token_preds["pred_cap"][i, t])
                # for the CSV required by la cátedra, punctuation ',' should be quoted in output consumer
                writer.writerow([inst_id, token_id, token_str, punt_ini, punt_fin, cap])
        # optional: also write a human-friendly aggregated word-level CSV or JSON if desired


# ---------------------------
# Ejemplo de uso dentro de un loop de evaluación
# ---------------------------
# Supongamos:
#   outputs = model(token_ids, attention_mask=attention_mask)
#   batch = next(iter(dataloader))
#   tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
#
# Example pseudo-run:
#
# outputs = model(batch["token_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
# token_preds = token_preds_from_logits(outputs, batch["attention_mask"], threshold=0.5)
# # convertir word_ids (torch tensor) a numpy
# word_ids_np = batch["word_ids"].cpu().numpy() if isinstance(batch["word_ids"], torch.Tensor) else batch["word_ids"]
# word_level = aggregate_token_preds_to_word_level(token_preds, word_ids_np)
# # escribir CSV token-level por cada ejemplo del batch:
# write_token_level_csv_for_batch(token_preds, batch, tokenizer, out_dir="results/pred_csv")
#
# # word_level contiene la predicción agregada por palabra por cada ejemplo en el batch:
# # word_level is a list (B) of lists (words), cada word dict tiene word_index y predicciones agregadas.
#
# # Ejemplo de acceso:
# for i, words in enumerate(word_level):
#     print("Example", i)
#     for w in words:
#         print(f"word_idx={w['word_index']}, token_positions={w['token_positions']}, "
#               f"p_ini={w['pred_p_ini']}, p_fin={w['pred_p_fin']}, cap={w['pred_cap']}")
#
# Ajustes posibles:
# - Cambiar reglas de agregación (por ejemplo usar OR/AND, usar prob average o weighted vote).
# - Manejar palabras que quedan partidas entre chunks: agrupar por original_instance_id + word_index
#   usando metadata (token_start/token_end) para unir predicciones en post-proc si querés evaluación a nivel palabra global.
