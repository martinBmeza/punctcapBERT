#!/usr/bin/env python3
"""
Script combinado para preprocesar datos y calcular embeddings de BERT en un solo paso.

Procesamiento:
1. Leer CSV con textos raw
2. Tokenizar con bert-base-multilingual-cased
3. Alinear etiquetas a subtokens
4. Segmentar en ventanas
5. Calcular embeddings de BERT para cada chunk
6. Guardar todo en archivos .npz

Entrada esperada: CSV con columnas: instance_id, text
Salida:
  - data/processed/{split}/{split}/chunk_{origid}_{chunkidx}.npz (con embeddings incluidos)
  - data/processed/metadata_{split}.parquet

Uso:
  python scripts/preprocess_with_embeddings.py \
    --input data/raw/train.csv \
    --out_dir data/processed/train \
    --split train \
    --max_len 128 \
    --stride 64 \
    --tokenizer bert-base-multilingual-cased \
    --device cuda
"""

import argparse
import os  
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import BertTokenizerFast, BertModel

from src.data.utils import split_words_and_punct, build_word_level_labels, is_punct_token


def normalize_word_for_input(word: str) -> str:
    """Normalizar palabra: sin puntuación, todo en minúsculas"""
    return word.lower()


def compute_bert_embeddings(token_ids: np.ndarray, model: BertModel, device: str, max_length: int = 512) -> np.ndarray:
    """
    Calcular embeddings de BERT para token_ids
    
    Args:
        token_ids: numpy array de token ids (T,)
        model: modelo BERT
        device: device a usar
        max_length: longitud máxima de secuencia para BERT
    
    Returns:
        embeddings: numpy array (T, 768)
    """
    # Si la secuencia es muy larga, cortarla
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    
    # Convertir a tensor y agregar batch dimension
    input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)  # (1, T)
    attention_mask = torch.ones_like(input_ids).to(device)  # (1, T)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Usar last_hidden_state: (1, T, 768)
        embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (T, 768)
    
    return embeddings


def process_row(instance_id: str, text: str, tokenizer: BertTokenizerFast, 
                model: BertModel, device: str, max_len: int, stride: int,
                compute_embeddings: bool = True) -> List[Dict]:
    """
    Procesa un texto original -> genera lista de chunks. Cada chunk es dict con:
      token_ids, tokens, word_ids, labels_punt_ini, labels_punt_fin, labels_cap,
      bert_embeddings (opcional), orig_instance_id, token_start, token_end
    """
    words_and_punct = split_words_and_punct(text)
    # obtener listas de palabras (sin puntuación) y labels a nivel palabra
    words_list, lab_p_ini, lab_p_fin, lab_cap = build_word_level_labels(words_and_punct)

    # construir lista de palabras normalizadas (input al tokenizer) (sin puntuación)
    normalized_words = [normalize_word_for_input(w) for w in words_list]

    if len(normalized_words) == 0:
        return []  # caso borde

    # tokenizar pre-segmentado por palabras (mantiene mapping word_ids())
    encoding = tokenizer(normalized_words,
                         is_split_into_words=True,
                         add_special_tokens=False,
                         return_attention_mask=False)

    input_ids = encoding["input_ids"]
    # obtener mapping token -> word index (para esta secuencia)
    word_ids_per_token = encoding.word_ids()

    # crear arrays por token: labels mapeando por word id
    n_tokens = len(input_ids)
    labels_p_ini_tok = np.zeros(n_tokens, dtype=np.int8)
    labels_p_fin_tok = np.zeros(n_tokens, dtype=np.int8)
    labels_cap_tok = np.zeros(n_tokens, dtype=np.int8)

    for t, wid in enumerate(word_ids_per_token):
        if wid is None:
            # no debería ocurrir si add_special_tokens=False
            labels_p_ini_tok[t] = 0
            labels_p_fin_tok[t] = 0
            labels_cap_tok[t] = 0
        else:
            labels_p_ini_tok[t] = lab_p_ini[wid]
            labels_p_fin_tok[t] = lab_p_fin[wid]
            labels_cap_tok[t] = lab_cap[wid]

    # tokens strings (opcional) para debug
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segmentación en ventanas de tokens
    chunks = []
    start = 0
    while start < n_tokens:
        end = min(start + max_len, n_tokens)
        chunk_input_ids = np.array(input_ids[start:end], dtype=np.int32)
        chunk_tokens = np.array(tokens[start:end], dtype=object)
        chunk_word_ids = np.array([wid if wid is None else int(wid) for wid in word_ids_per_token[start:end]], dtype=np.int32)
        chunk_lab_p_ini = labels_p_ini_tok[start:end]
        chunk_lab_p_fin = labels_p_fin_tok[start:end]
        chunk_lab_cap = labels_cap_tok[start:end]

        chunk_data = {
            "token_ids": chunk_input_ids,
            "tokens": chunk_tokens,
            "word_ids": chunk_word_ids,
            "labels_punt_ini": chunk_lab_p_ini,
            "labels_punt_fin": chunk_lab_p_fin,
            "labels_cap": chunk_lab_cap,
            "orig_instance_id": instance_id,
            "token_start": int(start),
            "token_end": int(end),
            "original_text": text
        }

        # Calcular embeddings de BERT si se requiere
        if compute_embeddings and model is not None:
            try:
                bert_embeddings = compute_bert_embeddings(chunk_input_ids, model, device)
                chunk_data["bert_embeddings"] = bert_embeddings
            except Exception as e:
                print(f"Error calculando embeddings para chunk {instance_id}: {e}")
                # Continuar sin embeddings en caso de error
                pass

        chunks.append(chunk_data)

        if end == n_tokens:
            break
        start += stride

    return chunks


def save_chunk_npz(chunk: dict, out_dir: Path, chunk_id: str) -> str:
    """Guardar chunk en archivo .npz comprimido"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{chunk_id}.npz"
    
    save_dict = {
        "token_ids": chunk["token_ids"],
        "tokens": chunk["tokens"],
        "word_ids": chunk["word_ids"],
        "labels_punt_ini": chunk["labels_punt_ini"],
        "labels_punt_fin": chunk["labels_punt_fin"],
        "labels_cap": chunk["labels_cap"],
        "orig_instance_id": chunk["orig_instance_id"],
        "token_start": np.int32(chunk["token_start"]),
        "token_end": np.int32(chunk["token_end"]),
        "original_text": np.array(chunk["original_text"], dtype=object)
    }
    
    # Agregar embeddings si existen
    if "bert_embeddings" in chunk:
        save_dict["bert_embeddings"] = chunk["bert_embeddings"]
    
    np.savez_compressed(path, **save_dict)
    return str(path)


def main(args):
    # Cargar tokenizer
    print(f"Cargando tokenizer: {args.tokenizer}")
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    
    # Cargar modelo BERT si se requieren embeddings
    model = None
    device = args.device
    
    if args.compute_embeddings:
        # Verificar disponibilidad de CUDA
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA no disponible, usando CPU")
            device = 'cpu'
        
        print(f"Cargando modelo BERT: {args.tokenizer}")
        print(f"Usando device: {device}")
        model = BertModel.from_pretrained(args.tokenizer)
        model = model.to(device)
        model.eval()
    else:
        print("Modo sin embeddings - solo tokenización")

    # Leer CSV
    print(f"Leyendo datos desde: {args.input}")
    df = pd.read_csv(args.input)
    assert "instance_id" in df.columns and "text" in df.columns, \
        "CSV debe tener columnas instance_id,text"
    
    out_base = Path(args.out_dir)
    meta_rows = []
    split_name = args.split

    # Procesar cada fila del CSV
    print(f"Procesando {len(df)} instancias...")
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Procesando filas"):
        inst_id = str(row["instance_id"])
        text = str(row["text"])
        
        # Procesar y generar chunks (con embeddings si se requiere)
        chunks = process_row(inst_id, text, tokenizer, model, device, 
                           args.max_len, args.stride, args.compute_embeddings)
        
        # Guardar cada chunk
        for cidx, chunk in enumerate(chunks):
            chunk_name = f"inst{inst_id}_chunk{cidx:04d}"
            saved_path = save_chunk_npz(chunk, out_base / split_name, chunk_name)
            meta_rows.append({
                "original_instance_id": inst_id,
                "chunk_id": chunk_name,
                "path": saved_path,
                "n_tokens": int(chunk["token_end"] - chunk["token_start"]),
                "token_start": int(chunk["token_start"]),
                "token_end": int(chunk["token_end"]),
                "split": split_name,
                "has_embeddings": "bert_embeddings" in chunk
            })

    # Guardar metadata
    meta_df = pd.DataFrame(meta_rows)
    meta_df_path = out_base / f"metadata_{split_name}.parquet"
    meta_df.to_parquet(meta_df_path, index=False)
    print(f"Metadata guardado en: {meta_df_path}")
    print(f"Total de chunks generados: {len(meta_rows)}")
    
    if args.compute_embeddings:
        n_with_embeddings = sum(meta_rows[i]["has_embeddings"] for i in range(len(meta_rows)))
        print(f"Chunks con embeddings: {n_with_embeddings}/{len(meta_rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocesar datos y calcular embeddings de BERT en un solo paso'
    )
    parser.add_argument("--input", type=str, required=True, 
                       help="CSV input con columnas instance_id,text")
    parser.add_argument("--out_dir", type=str, required=True, 
                       help="Directorio donde guardar processed chunks")
    parser.add_argument("--tokenizer", type=str, default="bert-base-multilingual-cased",
                       help="Nombre del tokenizer/modelo BERT")
    parser.add_argument("--max_len", type=int, default=128, 
                       help="Máx tokens por chunk")
    parser.add_argument("--stride", type=int, default=64, 
                       help="Stride (overlap) entre chunks")
    parser.add_argument("--split", type=str, default="train", 
                       help="Nombre de split (train/val/test)")
    parser.add_argument("--compute_embeddings", action="store_true",
                       help="Calcular embeddings de BERT (requiere GPU/más memoria)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device a usar para BERT (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)
