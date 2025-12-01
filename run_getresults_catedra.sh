#!/bin/bash
set -ex

PYTHONPATH=. python generate_test_predictions.py \
    --model_cfg "configs/datasetfinal_BiLSTM.yaml" \
    --model_pt "results/datasetfinal_BiLSTM-20251130_211734/best_model.pt" \
    --test_csv "data/raw/datos_test.csv" \
    --output_dir "data/results"