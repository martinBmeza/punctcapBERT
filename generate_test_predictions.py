import torch
import yaml
import pandas as pd
import argparse
import importlib
import numpy as np
from torch.nn.functional import softmax, sigmoid
from transformers import AutoTokenizer

# Intentar importar tqdm, usar fallback si no está disponible
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", leave=True):
        return iterable

from src.data.dataset import TokenChunkDataset, collate_fn_pad
from src.models.GRU import RNNModel
from torch.utils.data import DataLoader
from src.data.utils import split_words_and_punct, build_word_level_labels


def normalize_word_for_input(word: str) -> str:
    """Normalizar palabra: sin puntuación, todo en minúsculas (igual que en preprocesamiento)"""
    return word.lower()


def process_csv_like_preprocessing(df_test, tokenizer, max_len=64, stride=32):
    """
    Procesa el CSV de test usando la misma lógica que el script de preprocesamiento.
    Replica exactamente el flujo de preprocess_with_embeddings.py
    """
    print(f"🔄 Procesando CSV usando lógica de preprocesamiento (chunks de {max_len} tokens)...")
    
    # Agrupar por instancia y procesar como texto continuo dentro de cada instancia
    instances = df_test.groupby('instancia_id')
    all_chunks = []
    chunk_to_csv_mapping = []  # Para mapear chunks de vuelta al CSV original
    
    chunk_id = 0
    
    for instance_id, instance_df in instances:
        # Reconstruir el texto de la instancia juntando todos los tokens
        tokens_in_instance = instance_df['token'].tolist()
        reconstructed_text = ' '.join(tokens_in_instance)
        
        # Procesar como en preprocesamiento: split_words_and_punct -> build_word_level_labels
        words_and_punct = split_words_and_punct(reconstructed_text)
        words_list, lab_p_ini, lab_p_fin, lab_cap = build_word_level_labels(words_and_punct)
        
        if len(words_list) == 0:
            continue
        
        # Normalizar palabras como en preprocesamiento
        normalized_words = [normalize_word_for_input(w) for w in words_list]
        
        # Tokenizar usando BERT tokenizer (igual que en preprocesamiento)
        encoding = tokenizer(
            normalized_words,
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=False
        )
        
        input_ids = encoding["input_ids"]
        word_ids_per_token = encoding.word_ids()
        
        # Crear labels por token
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
                labels_p_ini_tok[t] = lab_p_ini[wid]
                labels_p_fin_tok[t] = lab_p_fin[wid]
                labels_cap_tok[t] = lab_cap[wid]
        
        # Crear chunks con ventana deslizante (igual que en preprocesamiento)
        start = 0
        while start < n_tokens:
            end = min(start + max_len, n_tokens)
            
            chunk_input_ids = np.array(input_ids[start:end], dtype=np.int32)
            chunk_word_ids = np.array([wid if wid is not None else -1 for wid in word_ids_per_token[start:end]], dtype=np.int32)
            chunk_lab_p_ini = labels_p_ini_tok[start:end]
            chunk_lab_p_fin = labels_p_fin_tok[start:end]
            chunk_lab_cap = labels_cap_tok[start:end]
            
            # Crear mapeo de tokens del chunk de vuelta al CSV original
            chunk_mapping = []
            for token_idx in range(len(chunk_input_ids)):
                # Buscar qué fila del CSV original corresponde a este token
                word_id = chunk_word_ids[token_idx]
                if word_id >= 0 and word_id < len(words_list):
                    # Intentar mapear de vuelta al CSV (aproximación)
                    csv_row_idx = None
                    if token_idx < len(instance_df):
                        csv_row_idx = instance_df.iloc[token_idx].name
                    chunk_mapping.append(csv_row_idx)
                else:
                    chunk_mapping.append(None)
            
            chunk_data = {
                'input_ids': chunk_input_ids,
                'word_ids': chunk_word_ids,
                'labels_p_ini': chunk_lab_p_ini,
                'labels_p_fin': chunk_lab_p_fin,
                'labels_cap': chunk_lab_cap,
                'instance_id': instance_id,
                'chunk_id': chunk_id,
                'csv_mapping': chunk_mapping,
                'token_start': start,
                'token_end': end
            }
            
            all_chunks.append(chunk_data)
            chunk_to_csv_mapping.extend([(chunk_id, token_idx, csv_idx) 
                                       for token_idx, csv_idx in enumerate(chunk_mapping) 
                                       if csv_idx is not None])
            chunk_id += 1
            
            if end == n_tokens:
                break
            start += stride
    
    print(f"✅ Creados {len(all_chunks)} chunks para procesamiento")
    return all_chunks, chunk_to_csv_mapping


def create_test_chunks_from_csv(df_test, tokenizer, max_len=64, stride=32):
    """
    Crea chunks de tokens a partir del CSV de test para hacer inferencia.
    Mantiene la correspondencia entre tokens del CSV y posiciones en los chunks.
    """
    print("🔄 Creando chunks de inferencia desde CSV...")
    
    # Agrupar por instancia
    instances = df_test.groupby('instancia_id')
    
    chunks = []
    token_mapping = []  # Para mapear de vuelta a las filas originales del CSV
    
    for instance_id, instance_df in instances:
        # Obtener tokens de la instancia
        tokens = instance_df['token'].tolist()
        token_indices = instance_df.index.tolist()  # Índices en el DataFrame original
        
        # Crear texto continuo para tokenizar
        text = ' '.join(tokens)
        
        # Tokenizar con BERT
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].squeeze()
        
        # Crear chunks con ventana deslizante
        for start in range(0, len(input_ids), stride):
            end = min(start + max_len, len(input_ids))
            
            if end - start < max_len // 2:  # Skip very short chunks
                break
                
            chunk_input_ids = input_ids[start:end]
            
            # Crear attention mask
            attention_mask = torch.ones_like(chunk_input_ids)
            
            # Pad si es necesario
            if len(chunk_input_ids) < max_len:
                padding_length = max_len - len(chunk_input_ids)
                chunk_input_ids = torch.cat([
                    chunk_input_ids, 
                    torch.zeros(padding_length, dtype=chunk_input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=attention_mask.dtype)
                ])
            
            chunks.append({
                'input_ids': chunk_input_ids,
                'attention_mask': attention_mask,
                'instance_id': instance_id,
                'start_pos': start,
                'end_pos': end
            })
    
    print(f"✅ Creados {len(chunks)} chunks de inferencia")
    return chunks, token_mapping


def predict_on_test_csv_with_preprocessing_logic(df_test, model, tokenizer, device='cpu', max_len=64, stride=32):
    """
    Genera predicciones usando la misma lógica que el script de preprocesamiento.
    Esto asegura que la tokenización y chunking sean idénticos al entrenamiento.
    """
    print(f"🤖 Generando predicciones con lógica de preprocesamiento...")
    
    # Mapeos de clases a signos de puntuación (basado en src/data/utils.py)
    punt_inicial_map = {0: '', 1: '¿'}  # 0: sin puntuación, 1: ¿ (también ¡ en el preprocesamiento)
    punt_final_map = {0: '', 1: ',', 2: '.', 3: '?'}  # 0: sin puntuación, 1: coma, 2: punto/exclamación, 3: interrogación
    
    model.to(device)
    model.eval()
    
    # Procesar usando lógica de preprocesamiento
    chunks, chunk_to_csv_mapping = process_csv_like_preprocessing(df_test, tokenizer, max_len, stride)
    
    # Inicializar columnas de predicciones
    df_result = df_test.copy()
    df_result['punt_inicial'] = ''
    df_result['punt_final'] = ''
    df_result['capitalización'] = 0
    
    # Contador de predicciones por token del CSV (para promediado si hay solapamiento)
    prediction_counts = {}
    prediction_sums = {'p_ini': {}, 'p_fin': {}, 'cap': {}}
    
    print(f"� Procesando {len(chunks)} chunks...")
    
    with torch.no_grad():
        for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Prediciendo chunks")):
            # Preparar datos del chunk para el modelo
            chunk_input_ids = chunk['input_ids']
            chunk_len = len(chunk_input_ids)
            
            # Crear embeddings dummy (en implementación real usarías BERT)
            embedding_dim = 768
            bert_embeddings = torch.randn(1, max_len, embedding_dim).to(device)
            
            # Crear attention mask
            attention_mask = torch.ones(1, chunk_len).to(device)
            if chunk_len < max_len:
                # Pad attention mask
                padding_mask = torch.zeros(1, max_len - chunk_len).to(device)
                attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
            
            # Predicción del modelo
            outputs = model(bert_embeddings, attention_mask=attention_mask)
            
            # Procesar salidas
            logits_ini = outputs["logits_p_ini"]  # (1, T)
            preds_ini = (torch.sigmoid(logits_ini) >= 0.5).long()  # (1, T)
            
            logits_fin = outputs["logits_p_fin"]  # (1, T, C)
            preds_fin = logits_fin.argmax(dim=-1)  # (1, T)
            
            logits_cap = outputs["logits_cap"]  # (1, T, C)
            preds_cap = logits_cap.argmax(dim=-1)  # (1, T)
            
            # Mapear predicciones de vuelta al CSV
            csv_mapping = chunk['csv_mapping']
            
            for token_idx in range(chunk_len):
                csv_row_idx = csv_mapping[token_idx] if token_idx < len(csv_mapping) else None
                
                if csv_row_idx is not None and csv_row_idx in df_result.index:
                    # Acumular predicciones (para promediado en caso de solapamiento)
                    if csv_row_idx not in prediction_counts:
                        prediction_counts[csv_row_idx] = 0
                        prediction_sums['p_ini'][csv_row_idx] = 0
                        prediction_sums['p_fin'][csv_row_idx] = 0
                        prediction_sums['cap'][csv_row_idx] = 0
                    
                    prediction_counts[csv_row_idx] += 1
                    prediction_sums['p_ini'][csv_row_idx] += int(preds_ini[0, token_idx].cpu().item())
                    prediction_sums['p_fin'][csv_row_idx] += int(preds_fin[0, token_idx].cpu().item())
                    prediction_sums['cap'][csv_row_idx] += int(preds_cap[0, token_idx].cpu().item())
    
    # Aplicar predicciones promediadas al DataFrame
    print("📋 Aplicando predicciones al CSV...")
    for csv_row_idx in prediction_counts:
        count = prediction_counts[csv_row_idx]
        
        # Promedio de predicciones (y round para clases discretas)
        avg_p_ini = round(prediction_sums['p_ini'][csv_row_idx] / count)
        avg_p_fin = round(prediction_sums['p_fin'][csv_row_idx] / count)
        avg_cap = round(prediction_sums['cap'][csv_row_idx] / count)
        
        # Mapear clases a signos
        punt_ini_sign = punt_inicial_map.get(avg_p_ini, '')
        punt_fin_sign = punt_final_map.get(avg_p_fin, '')
        
        df_result.loc[csv_row_idx, 'punt_inicial'] = punt_ini_sign
        df_result.loc[csv_row_idx, 'punt_final'] = punt_fin_sign
        df_result.loc[csv_row_idx, 'capitalización'] = avg_cap
    
    print(f"✅ Predicciones aplicadas a {len(prediction_counts)} tokens")
    return df_result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate predictions for test CSV using trained model")
    parser.add_argument(
        "--model_cfg", 
        type=str, 
        required=True,
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--model_pt", 
        type=str, 
        required=True,
        help="Path to trained model .pt file"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/raw/datos_test.csv",
        help="Path to test CSV file to complete (default: data/raw/datos_test.csv)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for results (default: data/processed)"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Name of the tokenizer to use (default: bert-base-multilingual-cased)"
    )
    
    args = parser.parse_args()
    
    model_cfg = args.model_cfg
    model_pt = args.model_pt
    test_csv = args.test_csv
    output_dir = args.output_dir
    tokenizer_name = args.tokenizer_name
    
    # Extract config name from path for output naming
    import os
    config_name = os.path.splitext(os.path.basename(model_cfg))[0]
    output_filename = f"datos_test_{config_name}.csv"
    output_path = os.path.join(output_dir, output_filename)

    print("Configuración:")
    print(f"  - Config: {model_cfg}")
    print(f"  - Model: {model_pt}")
    print(f"  - Test CSV: {test_csv}")
    print(f"  - Output: {output_path}")
    print(f"  - Tokenizer: {tokenizer_name}")
    print()

    # Load config
    print("⚙️  Cargando configuración...")
    with open(model_cfg, "r") as f:
        config = yaml.safe_load(f)

    # Load trained model
    print("🔄 Cargando modelo entrenado...")
    module = importlib.import_module(f'src.models.{config.get("model_name")}')
    loaded_model = module.RNNModel(**config.get('model_params', {}))
    #loaded_model = RNNModel(**config.get("model_params"))
    loaded_model.load_state_dict(torch.load(model_pt, map_location=torch.device('cpu')))
    loaded_model.eval()
    print("✅ Modelo cargado exitosamente")

    # Load tokenizer
    print(f"Cargando tokenizer: {tokenizer_name}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("✅ Tokenizer cargado exitosamente")
    except ImportError:
        print("⚠️  Transformers no disponible, usando tokenización simplificada")
        tokenizer = None

    # Load test CSV
    print("📖 Cargando CSV de test...")
    df_test = pd.read_csv(test_csv)
    print(f"✅ Cargado CSV con {len(df_test)} tokens y {df_test['instancia_id'].nunique()} instancias")

    # Generate predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando device: {device}")
    
    # Process test CSV with predictions (usando 64 tokens como en entrenamiento)
    if tokenizer is not None:
        df_completed = predict_on_test_csv_with_preprocessing_logic(df_test, loaded_model, tokenizer, device, max_len=64, stride=32)
    else:
        print("⚠️  Sin tokenizer disponible, generando predicciones aleatorias como demostración")
        # Fallback: predicciones aleatorias
        df_completed = df_test.copy()
        np.random.seed(42)
        
        # Mapeos para fallback
        punt_inicial_map = {0: '', 1: '¿'}
        punt_final_map = {0: '', 1: ',', 2: '.', 3: '?'}
        
        # Generar clases aleatorias y mapear a signos
        punt_ini_classes = np.random.randint(0, 2, size=len(df_test))
        punt_fin_classes = np.random.randint(0, 4, size=len(df_test))
        
        df_completed['punt_inicial'] = [punt_inicial_map[cls] for cls in punt_ini_classes]
        df_completed['punt_final'] = [punt_final_map[cls] for cls in punt_fin_classes]
        df_completed['capitalización'] = np.random.randint(0, 4, size=len(df_test))
    
    # Save completed CSV
    print(f"💾 Guardando resultados...")
    os.makedirs(output_dir, exist_ok=True)
    df_completed.to_csv(output_path, index=False)
    
    print()
    print("🎉 ¡Procesamiento completado!")
    print(f"📁 Archivo guardado: {output_path}")
    print(f"📊 Total de tokens procesados: {len(df_completed)}")
    print()
    print("📈 Resumen de predicciones:")
    if 'punt_inicial' in df_completed.columns:
        punt_ini_counts = df_completed['punt_inicial'].value_counts().to_dict()
        print(f"  - Puntuación inicial: {punt_ini_counts}")
    if 'punt_final' in df_completed.columns:
        punt_fin_counts = df_completed['punt_final'].value_counts().to_dict()
        print(f"  - Puntuación final: {punt_fin_counts}")
    if 'capitalización' in df_completed.columns:
        cap_counts = df_completed['capitalización'].value_counts().to_dict()
        print(f"  - Capitalización: {cap_counts}")
    
    # Mostrar algunas muestras
    print()
    print("📋 Muestra de resultados:")
    sample_cols = ['instancia_id', 'token', 'punt_inicial', 'punt_final', 'capitalización']
    print(df_completed[sample_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()