#!/bin/bash
set -ex

# PYTHONPATH=. python train.py --config configs/hamlet_RNN.yaml 
# PYTHONPATH=. python train.py --config configs/hamlet_GRU.yaml 
# PYTHONPATH=. python train.py --config configs/hamlet_LSTM.yaml 
# PYTHONPATH=. python train.py --config configs/hamlet_BiLSTM.yaml 

# PYTHONPATH=. python train.py --config configs/wiki_RNN.yaml 
# PYTHONPATH=. python train.py --config configs/wiki_GRU.yaml 
# PYTHONPATH=. python train.py --config configs/wiki_LSTM.yaml 
# PYTHONPATH=. python train.py --config configs/wiki_BiLSTM.yaml 

PYTHONPATH=. python train.py --config configs/datasetfinal_BiLSTM.yaml 
PYTHONPATH=. python train.py --config configs/datasetfinal_LSTM.yaml 
PYTHONPATH=. python train.py --config configs/datasetfinal_GRU.yaml 
PYTHONPATH=. python train.py --config configs/datasetfinal_RNN.yaml 


