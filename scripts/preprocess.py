"""
1 - Leer textos raw
2 - Tokenizar con bert-base-multilingual-cased
3 - Alinear etiquetas a subtokens: la etiqueta de la palabra debe propagarse a todos sus subtokens
4 - Convertir tokens a tokens ids
5 - Segmentacion en ventanas 
6 - Guardar instancias:
    -token_ids (subtoken ids)
    -tokens (strings, opcional para debug)
    -word_ids (mapping token→word index dentro la instancia)
    -labels_punt_inicial (0/1)
    -labels_punt_final (0..3) (map: 0=none,1=',' 2='.' 3='?')
    -labels_cap (0..3)
    -original_instance_id, token_start, token_end, original_text
length (T)

--- esto puede ir en otro script, extract_embeddings.py ---
5 - Extraer embeddings 
6 - Guardar arrays token-level (ids, tokens_str, embeddings, labels) y un registro en un DataFrame con metadatos por instancia.
"""

"""
Entrada esperada: CSV con columnas mínimas:
  instance_id,text
Salida:
  data/processed/{split}/chunk_{origid}_{chunkidx}.npz
  data/processed/metadata.parquet
Uso:
  python src/data/preprocess.py --input data/raw/train.csv --out_dir data/processed/train --split train \
       --max_len 128 --stride 64 --tokenizer bert-base-multilingual-cased
Por ahi mejorar el armado del path de salida?
"""

import argparse
import os  
import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
from tqdm import tqdm

from src.data.utils import split_words_and_punct, build_word_level_labels, is_punct_token

def normalize_word_for_input(word: str) -> str:
    # text normalizado: sin puntuación, todo en minúsculas (según consigna)
    return word.lower()

def process_row(instance_id: str, text: str, tokenizer: BertTokenizerFast, max_len: int, stride: int):
    """
    Procesa un texto original -> genera lista de chunks. Cada chunk es dict con:
      token_ids, tokens, word_ids, labels_punt_ini, labels_punt_fin, labels_cap,
      orig_instance_id, token_start, token_end
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
    # HuggingFace: encoding.word_ids() devuelve lista len = n_subtokens con valores en [0..n_words-1]
    word_ids_per_token = encoding.word_ids()

    # crear arrays por token: labels mapeando por word id
    n_tokens = len(input_ids)
    labels_p_ini_tok = np.zeros(n_tokens, dtype=np.int8)
    labels_p_fin_tok = np.zeros(n_tokens, dtype=np.int8)
    labels_cap_tok = np.zeros(n_tokens, dtype=np.int8)

    for t, wid in enumerate(word_ids_per_token):
        if wid is None:
            # debería no ocurrir si add_special_tokens=False, pero por si acaso
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

        chunks.append({
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
        })

        if end == n_tokens:
            break
        start += stride

    return chunks

def save_chunk_npz(chunk: dict, out_dir: Path, chunk_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{chunk_id}.npz"
    np.savez_compressed(
        path,
        token_ids=chunk["token_ids"],
        tokens=chunk["tokens"],
        word_ids=chunk["word_ids"],
        labels_punt_ini=chunk["labels_punt_ini"],
        labels_punt_fin=chunk["labels_punt_fin"],
        labels_cap=chunk["labels_cap"],
        orig_instance_id=chunk["orig_instance_id"],
        token_start=np.int32(chunk["token_start"]),
        token_end=np.int32(chunk["token_end"]),
        original_text=np.array(chunk["original_text"], dtype=object)
    )
    return str(path)

def main(args):
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    df = pd.read_csv(args.input)
    assert "instance_id" in df.columns and "text" in df.columns, "CSV debe tener columnas instance_id,text"
    out_base = Path(args.out_dir)
    meta_rows = []

    # crear carpeta por split si se pasó
    split_name = args.split

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Procesando filas"):
        inst_id = str(row["instance_id"])
        text = str(row["text"])
        chunks = process_row(inst_id, text, tokenizer, args.max_len, args.stride)
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
                "split": split_name
            })

    # guardar metadata
    meta_df = pd.DataFrame(meta_rows)
    meta_df_path = out_base / f"metadata_{split_name}.parquet"
    meta_df.to_parquet(meta_df_path, index=False)
    print("Metadata saved to:", meta_df_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="CSV input con columnas instance_id,text")
    parser.add_argument("--out_dir", type=str, required=True, help="Directorio donde guardar processed chunks")
    parser.add_argument("--tokenizer", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max_len", type=int, default=128, help="Máx tokens por chunk")
    parser.add_argument("--stride", type=int, default=64, help="Stride (overlap) entre chunks")
    parser.add_argument("--split", type=str, default="train", help="Nombre de split (train/val/test)")
    args = parser.parse_args()
    main(args)