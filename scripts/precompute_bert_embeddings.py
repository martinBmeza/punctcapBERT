#!/usr/bin/env python3
"""
Script para precalcular embeddings de BERT y guardarlos en los archivos .npz
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertModel


def load_bert_model(model_name='bert-base-multilingual-cased', device='cuda'):
    """Cargar modelo BERT y tokenizer"""
    print(f"Cargando modelo BERT: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def compute_bert_embeddings(token_ids, model, device='cuda', max_length=512):
    """
    Compute BERT embeddings for token_ids
    
    Args:
        token_ids: numpy array of token ids (T,)
        model: BERT model
        device: device to run on
        max_length: maximum sequence length for BERT
    
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


def process_npz_file(npz_path, model, device='cuda'):
    """
    Procesar un archivo .npz, agregar bert_embeddings y guardarlo
    
    Args:
        npz_path: path al archivo .npz
        model: modelo BERT
        device: device
    """
    # Cargar datos existentes
    with np.load(npz_path, allow_pickle=True) as data:
        data_dict = dict(data)
    
    # Verificar si ya tiene bert_embeddings
    if 'bert_embeddings' in data_dict:
        print(f"Saltando {npz_path} - ya tiene bert_embeddings")
        return
    
    # Obtener token_ids
    token_ids = data_dict.get('token_ids')
    if token_ids is None:
        print(f"Advertencia: {npz_path} no tiene token_ids, saltando...")
        return
    
    # Calcular embeddings
    try:
        bert_embeddings = compute_bert_embeddings(token_ids, model, device)
        
        # Agregar embeddings al diccionario
        data_dict['bert_embeddings'] = bert_embeddings
        
        # Guardar archivo actualizado
        np.savez_compressed(npz_path, **data_dict)
        
    except Exception as e:
        print(f"Error procesando {npz_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Precompute BERT embeddings for .npz files')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing .npz files')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased',
                        help='BERT model name')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Verificar device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA no disponible, usando CPU")
        args.device = 'cpu'
    
    print(f"Usando device: {args.device}")
    
    # Cargar modelo BERT
    tokenizer, model = load_bert_model(args.model_name, args.device)
    
    # Encontrar todos los archivos .npz
    data_dir = Path(args.data_dir)
    npz_files = list(data_dir.rglob('*.npz'))
    
    if args.max_files:
        npz_files = npz_files[:args.max_files]
    
    print(f"Encontrados {len(npz_files)} archivos .npz")
    
    # Procesar archivos
    for npz_path in tqdm(npz_files, desc="Procesando archivos"):
        process_npz_file(npz_path, model, args.device)
    
    print("¡Completado!")


if __name__ == '__main__':
    main()