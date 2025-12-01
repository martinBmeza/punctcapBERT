#!/usr/bin/env python3
"""
Script combinado para preprocesar datos y calcular embeddings de BERT en un solo paso.
VERSIÓN CONTINUA: Procesa todo el dataset como un texto continuo, permitiendo 
que las ventanas crucen entre oraciones.

Procesamiento:
1. Leer CSV con textos raw
2. Concatenar todas las oraciones en un texto continuo
3. Tokenizar con bert-base-multilingual-cased sobre el texto completo
4. Alinear etiquetas a subtokens manteniendo trazabilidad a oraciones originales
5. Segmentar en ventanas deslizantes que PUEDEN cruzar oraciones
6. Calcular embeddings de BERT para cada chunk
7. Guardar todo en archivos .npz con metadatos de oraciones contributivas

Entrada esperada: CSV con columnas: instance_id, text
Salida:
  - data/processed/{split}/{split}/continuous_chunk{id:06d}.npz (con embeddings incluidos)
  - data/processed/metadata_{split}.parquet (con info de oraciones contributivas)

Ventajas de este enfoque:
- Las ventanas pueden contener información de contexto entre oraciones
- Mejor para aprender transiciones entre oraciones
- Más realista para texto continuo

Uso:
  python scripts/preprocess_with_embeddings.py \
    --input data/raw/train.csv \
    --out_dir data/processed/train \
    --split train \
    --max_len 128 \
    --stride 64 \
    --tokenizer bert-base-multilingual-cased \
    --compute_embeddings \
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


def process_continuous_text_batched(df: pd.DataFrame, tokenizer: BertTokenizerFast, 
                                   model: BertModel, device: str, max_len: int, stride: int,
                                   compute_embeddings: bool = True, batch_size: int = 100) -> List[Dict]:
    """
    Procesa DataFrame en lotes para reducir uso de memoria.
    Las ventanas pueden cruzar entre oraciones dentro de cada lote.
    
    Args:
        df: DataFrame con columnas instance_id, text
        tokenizer: Tokenizer BERT
        model: Modelo BERT (puede ser None si no se calculan embeddings)
        device: Device para BERT
        max_len: Longitud máxima de chunk en tokens
        stride: Stride para ventanas deslizantes
        compute_embeddings: Si calcular embeddings de BERT
        batch_size: Número de oraciones por lote para reducir uso de memoria
    
    Returns:
        Lista de chunks con información de ventanas y mapeo a instancias originales
    """
    print(f"Procesando {len(df)} oraciones en lotes de {batch_size} para reducir memoria...")
    
    all_chunks = []
    total_chunks_generated = 0
    
    # Procesar en lotes
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Procesando lotes"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()
        
        print(f"\nLote {batch_idx + 1}/{num_batches}: oraciones {start_idx}-{end_idx-1}")
        
        # Concatenar oraciones del lote actual
        batch_words = []
        batch_labels_p_ini = []
        batch_labels_p_fin = []
        batch_labels_cap = []
        batch_instance_mapping = []
        
        print(f"  Concatenando {len(batch_df)} oraciones del lote...")
        for i, (_, row) in enumerate(tqdm(batch_df.iterrows(), total=len(batch_df), desc="  Concatenando", leave=False)):
            inst_id = str(row["instance_id"])
            text = str(row["text"])
            
            # Procesar la oración individual
            words_and_punct = split_words_and_punct(text)
            words_list, lab_p_ini, lab_p_fin, lab_cap = build_word_level_labels(words_and_punct)
            
            if len(words_list) == 0:
                continue  # Skip empty sentences
                
            # Normalizar palabras
            normalized_words = [normalize_word_for_input(w) for w in words_list]
            
            # Agregar a las listas del lote
            start_word_idx = len(batch_words)
            batch_words.extend(normalized_words)
            batch_labels_p_ini.extend(lab_p_ini)
            batch_labels_p_fin.extend(lab_p_fin)
            batch_labels_cap.extend(lab_cap)
            
            # Guardar mapeo de palabras a instancia para este lote
            for word_idx in range(len(normalized_words)):
                batch_instance_mapping.append({
                    'original_instance_id': inst_id,
                    'global_word_idx': start_word_idx + word_idx,
                    'local_word_idx': word_idx,
                    'original_text': text
                })
        
        if len(batch_words) == 0:
            print(f"  Lote {batch_idx + 1} vacío, saltando...")
            continue
            
        print(f"  Total palabras en lote: {len(batch_words)}")
        
        # Tokenizar el lote
        print("  Tokenizando lote...")
        encoding = tokenizer(batch_words,
                            is_split_into_words=True,
                            add_special_tokens=False,
                            return_attention_mask=False)
        
        input_ids = encoding["input_ids"]
        word_ids_per_token = encoding.word_ids()
        
        # Crear arrays de labels por token para el lote
        n_tokens = len(input_ids)
        labels_p_ini_tok = np.zeros(n_tokens, dtype=np.int8)
        labels_p_fin_tok = np.zeros(n_tokens, dtype=np.int8)
        labels_cap_tok = np.zeros(n_tokens, dtype=np.int8)
        
        for t, wid in enumerate(word_ids_per_token):
            if wid is None:
                labels_p_ini_tok[t] = 0
                labels_p_fin_tok[t] = 0
                labels_cap_tok[t] = 0
            else:
                labels_p_ini_tok[t] = batch_labels_p_ini[wid]
                labels_p_fin_tok[t] = batch_labels_p_fin[wid]
                labels_cap_tok[t] = batch_labels_cap[wid]
        
        # Tokens strings para debug
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"  Tokens en lote: {n_tokens}")
        print("  Generando chunks con ventanas deslizantes...")
        
        # Segmentación en ventanas deslizantes para este lote
        batch_chunks = []
        start = 0
        
        chunk_progress = tqdm(desc="  Generando chunks", leave=False)
        while start < n_tokens:
            end = min(start + max_len, n_tokens)
            
            # Extraer datos del chunk
            chunk_input_ids = np.array(input_ids[start:end], dtype=np.int32)
            chunk_tokens = np.array(tokens[start:end], dtype=object)
            chunk_word_ids = np.array([wid if wid is None else int(wid) for wid in word_ids_per_token[start:end]], dtype=np.int32)
            chunk_lab_p_ini = labels_p_ini_tok[start:end]
            chunk_lab_p_fin = labels_p_fin_tok[start:end]
            chunk_lab_cap = labels_cap_tok[start:end]
            
            # Determinar instancias originales que contribuyen a este chunk
            chunk_word_ids_valid = [wid for wid in chunk_word_ids if wid != -1 and wid < len(batch_instance_mapping)]
            contributing_instances = set()
            
            if len(chunk_word_ids_valid) > 0:
                for word_idx in chunk_word_ids_valid:
                    if word_idx < len(batch_instance_mapping):
                        inst_info = batch_instance_mapping[word_idx]
                        contributing_instances.add(inst_info['original_instance_id'])
            
            # Usar la primera instancia como ID principal, o crear uno global
            if contributing_instances:
                primary_instance = sorted(contributing_instances)[0]
                orig_text_summary = f"Batch{batch_idx}_Chunk spans {len(contributing_instances)} sentences"
            else:
                primary_instance = f"batch{batch_idx}_chunk_{total_chunks_generated}"
                orig_text_summary = "Unknown content"
            
            chunk_data = {
                "token_ids": chunk_input_ids,
                "tokens": chunk_tokens,
                "word_ids": chunk_word_ids,
                "labels_punt_ini": chunk_lab_p_ini,
                "labels_punt_fin": chunk_lab_p_fin,
                "labels_cap": chunk_lab_cap,
                "orig_instance_id": primary_instance,
                "token_start": int(start),
                "token_end": int(end),
                "original_text": orig_text_summary,
                "contributing_instances": list(contributing_instances),
                "chunk_id": total_chunks_generated,
                "batch_id": batch_idx
            }
            
            # Calcular embeddings de BERT si se requiere
            if compute_embeddings and model is not None:
                try:
                    bert_embeddings = compute_bert_embeddings(chunk_input_ids, model, device)
                    chunk_data["bert_embeddings"] = bert_embeddings
                except Exception as e:
                    print(f"  Error calculando embeddings para chunk {total_chunks_generated}: {e}")
            
            batch_chunks.append(chunk_data)
            total_chunks_generated += 1
            chunk_progress.update(1)
            
            if end == n_tokens:
                break
            start += stride
        
        chunk_progress.close()
        print(f"  Generados {len(batch_chunks)} chunks en este lote")
        
        # Agregar chunks del lote a la lista global
        all_chunks.extend(batch_chunks)
        
        # Limpiar memoria del lote
        del batch_words, batch_labels_p_ini, batch_labels_p_fin, batch_labels_cap
        del batch_instance_mapping, encoding, input_ids, word_ids_per_token
        del labels_p_ini_tok, labels_p_fin_tok, labels_cap_tok, tokens
        del batch_chunks
        
        print(f"  Memoria del lote liberada. Total chunks: {len(all_chunks)}")
    
    print(f"\n✅ Procesamiento completo: {total_chunks_generated} chunks generados")
    return all_chunks


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
    
    # Agregar información adicional para chunks continuos
    if "contributing_instances" in chunk:
        save_dict["contributing_instances"] = np.array(chunk["contributing_instances"], dtype=object)
    
    if "chunk_id" in chunk:
        save_dict["chunk_id"] = np.int32(chunk["chunk_id"])
        
    if "batch_id" in chunk:
        save_dict["batch_id"] = np.int32(chunk["batch_id"])
    
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
    
    print(f"Dataset cargado: {len(df)} instancias")
    print(f"Tamaño del lote configurado: {args.batch_size} oraciones")
    print(f"Configuración de memoria optimizada: {'ON' if args.batch_size < len(df) else 'OFF'}")
    
    out_base = Path(args.out_dir)
    meta_rows = []
    split_name = args.split

    # Procesar todo el dataset como texto continuo en lotes
    print(f"Iniciando procesamiento continuo con lotes de {args.batch_size}...")
    chunks = process_continuous_text_batched(df, tokenizer, model, device, 
                                           args.max_len, args.stride, 
                                           args.compute_embeddings, args.batch_size)
    
    # Guardar cada chunk
    print(f"\nGuardando {len(chunks)} chunks...")
    for chunk in tqdm(chunks, desc="Guardando chunks"):
        chunk_name = f"continuous_chunk{chunk['chunk_id']:06d}"
        saved_path = save_chunk_npz(chunk, out_base / split_name, chunk_name)
        meta_rows.append({
            "original_instance_id": chunk["orig_instance_id"],
            "chunk_id": chunk_name,
            "path": saved_path,
            "n_tokens": int(chunk["token_end"] - chunk["token_start"]),
            "token_start": int(chunk["token_start"]),
            "token_end": int(chunk["token_end"]),
            "split": split_name,
            "has_embeddings": "bert_embeddings" in chunk,
            "contributing_instances": ",".join(chunk.get("contributing_instances", [])),
            "spans_multiple_sentences": len(chunk.get("contributing_instances", [])) > 1,
            "batch_id": chunk.get("batch_id", -1)
        })

    # Guardar metadata
    print("Guardando metadata...")
    meta_df = pd.DataFrame(meta_rows)
    meta_df_path = out_base / f"metadata_{split_name}.parquet"
    meta_df.to_parquet(meta_df_path, index=False)
    print(f"Metadata guardado en: {meta_df_path}")
    
    # Estadísticas finales
    print(f"\n{'='*50}")
    print(f"✅ PROCESAMIENTO COMPLETO")
    print(f"{'='*50}")
    print(f"Oraciones originales: {len(df)}")
    print(f"Chunks generados: {len(meta_rows)}")
    print(f"Tokens por chunk: {args.max_len}")
    print(f"Stride: {args.stride}")
    print(f"Tamaño de lote usado: {args.batch_size}")
    
    # Estadísticas de chunks cruzados
    chunks_spanning_multiple = sum(row["spans_multiple_sentences"] for row in meta_rows)
    percentage = chunks_spanning_multiple/len(meta_rows)*100 if len(meta_rows) > 0 else 0
    print(f"Chunks que cruzan oraciones: {chunks_spanning_multiple}/{len(meta_rows)} ({percentage:.1f}%)")
    
    # Estadísticas de embeddings
    if args.compute_embeddings:
        n_with_embeddings = sum(row["has_embeddings"] for row in meta_rows)
        print(f"Chunks con embeddings: {n_with_embeddings}/{len(meta_rows)}")
    
    # Estadísticas de lotes
    num_batches = meta_df['batch_id'].nunique() if 'batch_id' in meta_df.columns else 1
    print(f"Número de lotes procesados: {num_batches}")
    avg_chunks_per_batch = len(meta_rows) / num_batches if num_batches > 0 else 0
    print(f"Promedio de chunks por lote: {avg_chunks_per_batch:.1f}")
    
    print(f"\n🎯 Ventanas cruzando oraciones: {percentage:.1f}%")
    print("🚀 Procesamiento optimizado completado exitosamente!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocesar datos y calcular embeddings de BERT en un solo paso (optimizado para memoria)'
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
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Número de oraciones por lote (reducir si hay problemas de memoria)")
    
    args = parser.parse_args()
    main(args)
