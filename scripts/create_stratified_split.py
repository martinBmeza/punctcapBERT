#!/usr/bin/env python3
"""
Script para crear un split estratificado 90-10 de los datos de entrenamiento
manteniendo el balance de clases y la integridad de las instancias originales.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


def analyze_instance_classes(metadata_df, data_dir):
    """
    Analiza las clases de cada instancia original para el split estratificado
    """
    print("Analizando distribución de clases por instancia...")
    
    instance_classes = {}
    
    # Agrupar por instancia original
    grouped = metadata_df.groupby('original_instance_id')
    
    for instance_id, group in tqdm(grouped, desc="Analizando instancias"):
        # Inicializar contadores para esta instancia
        p_ini_counts = Counter()
        p_fin_counts = Counter()
        cap_counts = Counter()
        
        # Analizar todos los chunks de esta instancia
        for _, row in group.iterrows():
            chunk_path = row['path']
            
            # Cargar el chunk para obtener las labels
            try:
                with np.load(chunk_path, allow_pickle=True) as data:
                    if 'labels_punt_ini' in data:
                        p_ini_counts.update(data['labels_punt_ini'])
                    if 'labels_punt_fin' in data:
                        p_fin_counts.update(data['labels_punt_fin'])
                    if 'labels_cap' in data:
                        cap_counts.update(data['labels_cap'])
            except Exception as e:
                print(f"Error loading {chunk_path}: {e}")
                continue
        
        # Calcular proporciones para esta instancia
        total_p_ini = sum(p_ini_counts.values())
        total_p_fin = sum(p_fin_counts.values())
        total_cap = sum(cap_counts.values())
        
        # Crear un "fingerprint" de la distribución de clases
        p_ini_prop = p_ini_counts.get(1, 0) / max(total_p_ini, 1)  # Proporción de clase positiva
        
        # Para multiclass, usar la clase más frecuente como característica
        p_fin_dominant = max(p_fin_counts, key=p_fin_counts.get) if p_fin_counts else 0
        cap_dominant = max(cap_counts, key=cap_counts.get) if cap_counts else 0
        
        # Crear bins para estratificar
        p_ini_bin = int(p_ini_prop * 10)  # Bins de 0-9
        
        # Combinar características para crear estratos
        strata = f"p_ini_{p_ini_bin}_p_fin_{p_fin_dominant}_cap_{cap_dominant}"
        
        instance_classes[instance_id] = {
            'strata': strata,
            'p_ini_prop': p_ini_prop,
            'p_fin_dominant': p_fin_dominant,
            'cap_dominant': cap_dominant,
            'total_tokens': total_p_ini
        }
    
    return instance_classes


def create_stratified_split(metadata_df, instance_classes, test_size=0.1, random_state=42):
    """
    Crea el split estratificado basado en las instancias
    """
    print(f"Creando split estratificado {int((1-test_size)*100)}-{int(test_size*100)}...")
    
    # Preparar datos para sklearn
    instances = list(instance_classes.keys())
    strata = [instance_classes[inst]['strata'] for inst in instances]
    
    # Contar instancias por estrato
    strata_counts = Counter(strata)
    print(f"Total de instancias: {len(instances)}")
    print(f"Número de estratos únicos: {len(strata_counts)}")
    
    # Filtrar estratos que tienen al menos 2 instancias para poder hacer split
    valid_instances = []
    valid_strata = []
    
    for inst, stratum in zip(instances, strata):
        if strata_counts[stratum] >= 2:
            valid_instances.append(inst)
            valid_strata.append(stratum)
        else:
            # Instancias únicas van a entrenamiento
            print(f"Instancia única {inst} con estrato {stratum} va a entrenamiento")
    
    print(f"Instancias válidas para split: {len(valid_instances)}")
    
    # Hacer el split estratificado
    if len(valid_instances) > 0:
        train_instances, val_instances = train_test_split(
            valid_instances,
            test_size=test_size,
            stratify=valid_strata,
            random_state=random_state
        )
    else:
        train_instances, val_instances = instances, []
    
    # Agregar instancias únicas a entrenamiento
    unique_instances = [inst for inst, stratum in zip(instances, strata) if strata_counts[stratum] < 2]
    train_instances.extend(unique_instances)
    
    print(f"Instancias de entrenamiento: {len(train_instances)}")
    print(f"Instancias de validación: {len(val_instances)}")
    
    # Crear los dataframes finales
    train_mask = metadata_df['original_instance_id'].isin(train_instances)
    val_mask = metadata_df['original_instance_id'].isin(val_instances)
    
    train_df = metadata_df[train_mask].copy()
    val_df = metadata_df[val_mask].copy()
    
    # Actualizar la columna split
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    
    return train_df, val_df


def copy_validation_files(val_df, original_dir, output_dir):
    """
    Copia los archivos de validación a un directorio separado
    """
    val_dir = output_dir / 'val'
    val_dir.mkdir(exist_ok=True)
    
    print(f"Copiando {len(val_df)} archivos de validación...")
    
    updated_paths = []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Copiando archivos"):
        original_path = Path(row['path'])
        new_path = val_dir / original_path.name
        
        # Copiar archivo
        shutil.copy2(original_path, new_path)
        
        # Actualizar path en el dataframe
        updated_paths.append(str(new_path))
    
    val_df = val_df.copy()
    val_df['path'] = updated_paths
    
    return val_df


def print_split_statistics(train_df, val_df, instance_classes):
    """
    Imprime estadísticas del split
    """
    print("\n" + "="*50)
    print("ESTADÍSTICAS DEL SPLIT")
    print("="*50)
    
    train_instances = set(train_df['original_instance_id'])
    val_instances = set(val_df['original_instance_id'])
    
    print(f"Total chunks entrenamiento: {len(train_df)}")
    print(f"Total chunks validación: {len(val_df)}")
    print(f"Total instancias entrenamiento: {len(train_instances)}")
    print(f"Total instancias validación: {len(val_instances)}")
    
    # Analizar distribución de clases
    train_p_ini_props = [instance_classes[inst]['p_ini_prop'] for inst in train_instances]
    val_p_ini_props = [instance_classes[inst]['p_ini_prop'] for inst in val_instances]
    
    print(f"\nProporción promedio p_ini (positivos):")
    print(f"  Entrenamiento: {np.mean(train_p_ini_props):.4f}")
    print(f"  Validación: {np.mean(val_p_ini_props):.4f}")
    
    # Distribución de clases dominantes
    train_p_fin = [instance_classes[inst]['p_fin_dominant'] for inst in train_instances]
    val_p_fin = [instance_classes[inst]['p_fin_dominant'] for inst in val_instances]
    
    print(f"\nDistribución p_fin dominante:")
    print(f"  Entrenamiento: {Counter(train_p_fin)}")
    print(f"  Validación: {Counter(val_p_fin)}")


def main():
    parser = argparse.ArgumentParser(description='Crear split estratificado de los datos')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path al archivo metadata_train.parquet')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directorio de salida para los archivos split')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Proporción para validación (default: 0.1)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Semilla para reproducibilidad')
    
    args = parser.parse_args()
    
    # Cargar metadata
    print(f"Cargando metadata desde {args.metadata}")
    metadata_df = pd.read_parquet(args.metadata)
    print(f"Total de chunks: {len(metadata_df)}")
    print(f"Total de instancias únicas: {metadata_df['original_instance_id'].nunique()}")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Analizar clases por instancia
    instance_classes = analyze_instance_classes(metadata_df, output_dir)
    
    # Crear split estratificado
    train_df, val_df = create_stratified_split(
        metadata_df, 
        instance_classes, 
        test_size=args.val_ratio,
        random_state=args.random_seed
    )
    
    # Copiar archivos de validación
    val_df = copy_validation_files(val_df, output_dir, output_dir)
    
    # Guardar metadata actualizados
    train_metadata_path = output_dir / 'metadata_train.parquet'
    val_metadata_path = output_dir / 'metadata_val.parquet'
    
    train_df.to_parquet(train_metadata_path, index=False)
    val_df.to_parquet(val_metadata_path, index=False)
    
    print(f"\nArchivos guardados:")
    print(f"  Entrenamiento: {train_metadata_path}")
    print(f"  Validación: {val_metadata_path}")
    
    # Estadísticas finales
    print_split_statistics(train_df, val_df, instance_classes)
    
    print(f"\n¡Split completado exitosamente!")


if __name__ == '__main__':
    main()