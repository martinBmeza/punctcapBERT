#!/bin/bash
set -ex

PYTHONPATH=. python get_results.py \
    --model_cfg "configs/datasetfinal_BiLSTM.yaml" \
    --model_pt "results/datasetfinal_BiLSTM-20251130_211734/best_model.pt" \
    --output_prefix "test" \
    --validation_metadata "data/processed/test/metadata_train.parquet"  

PYTHONPATH=. python get_results.py \
    --model_cfg "configs/datasetfinal_GRU.yaml" \
    --model_pt "results/datasetfinal_GRU-20251130_221518/best_model.pt" \
    --output_prefix "test" \
    --validation_metadata "data/processed/test/metadata_train.parquet"  

PYTHONPATH=. python get_results.py \
    --model_cfg "configs/datasetfinal_LSTM.yaml" \
    --model_pt "results/datasetfinal_LSTM-20251130_214648/best_model.pt" \
    --output_prefix "test" \
    --validation_metadata "data/processed/test/metadata_train.parquet"

PYTHONPATH=. python get_results.py \
    --model_cfg "configs/datasetfinal_RNN.yaml" \
    --model_pt "results/datasetfinal_RNN-20251130_224332/best_model.pt" \
    --output_prefix "test" \
    --validation_metadata "data/processed/test/metadata_train.parquet"

    