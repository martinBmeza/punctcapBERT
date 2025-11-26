#!/bin/bash

PYTHONPATH=. python scripts/preprocess.py --input data/raw/Emilianohack6950-Wikipedia-es-train.csv --out_dir data/processed/train --split train \
       --max_len 128 --stride 64 --tokenizer bert-base-multilingual-cased