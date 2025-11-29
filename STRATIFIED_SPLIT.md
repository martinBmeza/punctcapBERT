# Split Estratificado de Datos

Se ha creado un sistema de split estratificado para dividir los datos en conjuntos de entrenamiento y validación manteniendo el balance de clases.

## ¿Qué se hizo?

1. **Análisis por instancia original**: Se analizó la distribución de clases por cada instancia completa (no por chunks individuales) para mantener la integridad de los documentos.

2. **Creación de estratos**: Se crearon estratos basados en:
   - Proporción de puntuación inicial (p_ini)
   - Clase dominante de puntuación final (p_fin)
   - Clase dominante de capitalización (cap)

3. **Split 90-10**: Se dividieron las instancias en 90% entrenamiento y 10% validación manteniendo la distribución de estratos.

4. **Organización de archivos**: Los archivos de validación se copiaron a un directorio separado `data/processed/wikipedia_split/val/`

## Resultados del Split

```
Total chunks entrenamiento: 5,390
Total chunks validación: 633
Total instancias entrenamiento: 720  
Total instancias validación: 80
Porcentaje validación: 10.5%
```

## Archivos Generados

```
data/processed/wikipedia_split/
├── metadata_train.parquet     # Metadata para entrenamiento
├── metadata_val.parquet       # Metadata para validación  
└── val/                       # Archivos .npz de validación
    ├── instXXXXXX_chunk0000.npz
    ├── instXXXXXX_chunk0001.npz
    └── ...
```

## Uso

### Crear el split (ya ejecutado):
```bash
python scripts/create_stratified_split.py \
  --metadata data/processed/wikipedia/metadata_train.parquet \
  --output_dir data/processed/wikipedia_split \
  --val_ratio 0.1 \
  --random_seed 42
```

### Entrenar con el split:
```bash
python train.py --config configs/baselineRNN_split.yaml
```

## Configuración

En `configs/baselineRNN_split.yaml`:

```yaml
# Usar datos de entrenamiento estratificado
metadata: data/processed/wikipedia_split/metadata_train.parquet

# Configurar validación
validation_metadata: data/processed/wikipedia_split/metadata_val.parquet
evaluate_every_n_epochs: 1  # Evaluar cada época
```

## Ventajas del Split Estratificado

1. **Balance de clases preservado**: Los conjuntos de entrenamiento y validación mantienen la misma distribución de clases que el conjunto original.

2. **Integridad de documentos**: Las instancias completas permanecen juntas (no se separan chunks de la misma instancia entre train y val).

3. **Reproducibilidad**: Con la misma semilla aleatoria, siempre se genera el mismo split.

4. **Evaluación válida**: Las métricas de validación son representativas del rendimiento real del modelo en datos no vistos.

## Próximos Pasos

El script de entrenamiento debe actualizarse para:
1. Cargar el dataset de validación
2. Evaluar el modelo en validación después de cada época
3. Reportar métricas separadas para entrenamiento y validación
4. Implementar early stopping basado en validación

Esto permitirá un entrenamiento más robusto y métricas más confiables del rendimiento del modelo.