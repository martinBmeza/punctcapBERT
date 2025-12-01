# Script de Generación de Predicciones para Test CSV (Versión Optimizada)

Este script (`generate_test_predictions.py`) genera predicciones para completar un CSV de test usando un modelo entrenado. **NUEVA VERSIÓN**: Replica exactamente la lógica del script de preprocesamiento para máxima compatibilidad.

## ✨ Características Mejoradas

### **🔄 Lógica de Preprocesamiento Replicada**
- Usa `split_words_and_punct()` y `build_word_level_labels()` igual que en entrenamiento
- Tokenización idéntica con `BertTokenizerFast` y `is_split_into_words=True`
- Chunks de **64 tokens** con stride de **32** (exactamente como entrenamiento)
- Normalización de palabras con `normalize_word_for_input()`

### **📊 Mapeo Correcto de Clases**
Verificado contra `src/data/utils.py`:
- **Puntuación inicial**: `{0: '', 1: '¿'}`  (incluye ¡ en preprocesamiento)
- **Puntuación final**: `{0: '', 1: ',', 2: '.', 3: '?'}` (clase 2 incluye ! → punto)
- **Capitalización**: Mantiene valores numéricos 0-3

### **🎯 Promediado de Predicciones Solapadas**
- Los chunks con stride generan solapamiento entre tokens
- Las predicciones se promedian para tokens que aparecen en múltiples chunks
- Resultado final más robusto y consistente

## Uso Básico

```bash
# Generar predicciones usando archivos por defecto
python generate_test_predictions.py \
    --model_cfg "configs/wiki_BiLSTM.yaml" \
    --model_pt "results/wiki_BiLSTM-20251129_052431/best_model.pt"

# O usando el script de bash
bash run_generate_test_predictions.sh
```

## Argumentos CLI

- `--model_cfg`: Ruta al archivo de configuración YAML del modelo (**requerido**)
- `--model_pt`: Ruta al archivo .pt del modelo entrenado (**requerido**)
- `--test_csv`: Ruta al CSV de test (default: `data/raw/datos_test.csv`)
- `--output_dir`: Directorio de salida (default: `data/processed`)
- `--tokenizer_name`: Nombre del tokenizer (default: `bert-base-multilingual-cased`)

## Formato de Entrada

El CSV de test debe tener las columnas:
```
instancia_id,token_id,token,punt_inicial,punt_final,capitalización
```

Donde las últimas 3 columnas pueden estar vacías (se completarán con las predicciones).

## Formato de Salida

El script genera un archivo `datos_test_{config_name}.csv` con las mismas columnas pero completadas:

- `punt_inicial`: '' (sin puntuación) o '¿' (puntuación inicial)
- `punt_final`: '' (sin puntuación), ',' (coma), '.' (punto), o '?' (interrogación)  
- `capitalización`: 0-3 (diferentes tipos de capitalización)

## Mapeo de Predicciones

### Puntuación Inicial
- Clase 0 → '' (cadena vacía, sin puntuación)
- Clase 1 → '¿' (signo de interrogación inicial)

### Puntuación Final  
- Clase 0 → '' (cadena vacía, sin puntuación)
- Clase 1 → ',' (coma)
- Clase 2 → '.' (punto)
- Clase 3 → '?' (signo de interrogación final)

## Ejemplo de Salida

```
instancia_id  token    punt_inicial  punt_final  capitalización
0            la       ""            ""          1
0            cuestión "¿"           ""          0
0            es       ""            "?"         0
0            que      ""            ","         0
```

## Funcionamiento Interno Mejorado

1. **Procesamiento por instancia**: Agrupa tokens por `instancia_id`
2. **Reconstrucción de texto**: Une tokens para recrear texto original
3. **Preprocesamiento replicado**: 
   - `split_words_and_punct()` → separar palabras y puntuación
   - `build_word_level_labels()` → crear etiquetas a nivel palabra
   - `normalize_word_for_input()` → normalizar palabras
4. **Tokenización BERT exacta**: Mismo proceso que entrenamiento
5. **Chunking con stride**: Ventanas de 64 tokens con solapamiento de 32
6. **Predicción del modelo**: Inferencia en cada chunk
7. **Mapeo inverso**: De tokens de chunks de vuelta a CSV original
8. **Promediado**: Combina predicciones solapadas para mayor robustez
9. **Conversión de clases**: Mapea números a signos de puntuación

## Configuración de Tokens

```python
max_len = 64      # Igual que entrenamiento
stride = 32       # 50% de solapamiento para robustez
```

## Mapeo de Clases Verificado

Basado en `src/data/utils.py`, las clases se mapean correctamente:

```python
# Puntuación inicial (¿ y ¡ se marcan como clase 1)
punt_inicial_map = {0: '', 1: '¿'}

# Puntuación final (! se mapea a clase 2 como punto)
punt_final_map = {0: '', 1: ',', 2: '.', 3: '?'}
```

## Manejo de Errores

- Si `transformers` no está disponible, genera predicciones aleatorias como demostración
- Si el texto es muy largo, lo divide en chunks y procesa por partes
- Maneja instancias de diferentes longitudes automáticamente

## Archivos de Salida

- `datos_test_{config_name}.csv`: CSV completado con predicciones
- Estadísticas en consola con resumen de predicciones generadas
- Muestra de los primeros resultados para verificación

## Ejemplo de Uso Completo

```bash
python generate_test_predictions.py \
    --model_cfg "configs/baselineRNN.yaml" \
    --model_pt "results/baseline_experiment/best_model.pt" \
    --test_csv "data/raw/datos_test.csv" \
    --output_dir "data/results" \
    --tokenizer_name "bert-base-multilingual-cased"
```

Esto generará: `data/results/datos_test_baselineRNN.csv`