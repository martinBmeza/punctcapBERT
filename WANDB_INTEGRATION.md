# Weights & Biases Integration

The training script now includes Weights & Biases (wandb) integration for experiment tracking.

## Setup

1. **Install wandb** (already included in requirements.txt):
```bash
pip install wandb
```

2. **Login to wandb**:
```bash
wandb login
```
Follow the instructions to authenticate with your wandb account.

## Configuration

In your config file (`configs/baselineRNN.yaml`), set the following parameters:

```yaml
# wandb logging
use_wandb: true  # Set to true to enable wandb logging
wandb_log_steps: false  # Set to true to log metrics every step, false for epoch-only
wandb_project: punctcapBERT
wandb_run_name: null  # Auto-generated if null, or specify custom name
wandb_tags: ["baseline", "rnn", "bert-embeddings"]
wandb_notes: "Baseline RNN model with pre-computed BERT embeddings"
```

## Logging Levels

You can control the granularity of logging with the `wandb_log_steps` parameter:

- **Epoch-only logging** (`wandb_log_steps: false`): Logs metrics only at the end of each epoch. This is recommended for most use cases as it reduces clutter and focuses on overall training progress.

- **Step-level logging** (`wandb_log_steps: true`): Logs metrics after every training step. Use this for detailed debugging or when you want to see fine-grained training dynamics.

## What Gets Logged

### Step-level metrics (logged every step):
- `step_loss`: Total loss for the current step
- `step_loss_p_ini`: Punctuation initial loss
- `step_loss_p_fin`: Punctuation final loss  
- `step_loss_cap`: Capitalization loss
- `step_acc_p_ini`: Punctuation initial accuracy
- `step_acc_p_fin`: Punctuation final accuracy
- `step_acc_cap`: Capitalization accuracy
- `epoch`: Current epoch number
- `step`: Current step number

### Epoch-level metrics (logged every epoch):
- `epoch_loss`: Average loss for the epoch
- `epoch_acc_p_ini`: Average punctuation initial accuracy
- `epoch_acc_p_fin`: Average punctuation final accuracy
- `epoch_acc_cap`: Average capitalization accuracy
- `best_loss`: Best loss achieved so far

### Model tracking:
- `best_epoch`: Epoch where best model was saved
- `best_loss`: Best loss value achieved

### Configuration:
- All hyperparameters from the config file are automatically logged

## Usage

Run training with wandb enabled:

```bash
python train.py --config configs/baselineRNN.yaml
```

If `use_wandb: true` in your config, the script will:
1. Initialize a new wandb run
2. Log all hyperparameters
3. Track metrics in real-time
4. Save best model information
5. Finish the run when training completes

## Viewing Results

Visit [wandb.ai](https://wandb.ai) to view your experiment results, including:
- Real-time loss and accuracy plots
- Hyperparameter comparisons
- System metrics (GPU/CPU usage)
- Model performance summaries

## Disabling wandb

To train without wandb logging, simply set:
```yaml
use_wandb: false
```

The training will proceed normally without any wandb integration.