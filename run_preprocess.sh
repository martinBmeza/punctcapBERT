#!/bin/bash

set -ex

# PYTHONPATH=. python scripts/preprocess_with_embeddings.py \
#     --input "data/raw/Emilianohack6950-Wikipedia-es-train.csv" \
#     --out_dir "data/processed/wikipedia" \
#     --split "train" \
#     --max_len "128" \
#     --stride "64" \
#     --tokenizer "bert-base-multilingual-cased" \
#     --compute_embeddings \
#     --device "cuda"


# PYTHONPATH=. python scripts/preprocess_with_embeddings.py \
#     --input "data/raw/damaboba.csv" \
#     --out_dir "data/processed/damaboba" \
#     --split "train" \
#     --max_len "64" \
#     --stride "32" \
#     --tokenizer "bert-base-multilingual-cased" \
#     --compute_embeddings \
#     --device "cuda"


# PYTHONPATH=. python scripts/preprocess_with_embeddings.py \
#     --input "data/raw/hamlet.csv" \
#     --out_dir "data/processed/hamlet" \
#     --split "train" \
#     --max_len "64" \
#     --stride "32" \
#     --tokenizer "bert-base-multilingual-cased" \
#     --compute_embeddings \
#     --device "cuda"

PYTHONPATH=. python scripts/preprocess_with_embeddings.py \
    --input "data/raw/dataset_final.csv" \
    --out_dir "data/processed/datasetfinal" \
    --split "train" \
    --max_len "64" \
    --stride "32" \
    --tokenizer "bert-base-multilingual-cased" \
    --compute_embeddings \
    --device "cuda" \
    --batch_size "50"

# PYTHONPATH=.python scripts/create_stratified_split.py \
#   --metadata data/processed/wikipedia/metadata_train.parquet \
#   --output_dir data/processed/wikipedia_split \
#   --val_ratio 0.1 \
#   --random_seed 42

# PYTHONPATH=. python scripts/create_stratified_split.py \
#   --metadata data/processed/damaboba/metadata_train.parquet \
#   --output_dir data/processed/damaboba_split \
#   --val_ratio 0.1 \
#   --random_seed 42

# PYTHONPATH=. python scripts/create_stratified_split.py \
#   --metadata data/processed/hamlet/metadata_train.parquet \
#   --output_dir data/processed/hamlet_split \
#   --val_ratio 0.1 \
#   --random_seed 42

PYTHONPATH=. python scripts/create_stratified_split.py \
  --metadata data/processed/datasetfinal/metadata_train.parquet \
  --output_dir data/processed/datasetfinal_split \
  --val_ratio 0.1 \
  --random_seed 42