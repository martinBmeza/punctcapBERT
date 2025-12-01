import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """
    Pre-computed BERT embeddings -> GRU -> per-token heads:
      - punt_inicial: binary (logit)
      - punt_final: multiclass (4 classes)
      - capitalizacion: multiclass (4 classes)

    Args:
      embed_dim: embedding dimension (should match BERT embedding size, typically 768)
      hidden_dim: hidden dimension of GRU
      num_layers: GRU layers
      dropout: dropout on GRU outputs
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256,
                 num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.1):
        super().__init__()
        # No embedding layer needed - we'll use pre-computed BERT embeddings
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.RNN(input_size=embed_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0,
                          bidirectional=bidirectional)

        out_dim = hidden_dim * self.num_directions
        # Heads
        self.head_p_ini = nn.Linear(out_dim, 1)       # binary logit
        self.head_p_fin = nn.Linear(out_dim, 4)       # 4 classes: none, ',', '.', '?'
        self.head_cap = nn.Linear(out_dim, 4)         # 4 classes: 0..3

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.FloatTensor, attention_mask: torch.BoolTensor = None):
        """
        embeddings: (B, T, E) pre-computed BERT embeddings
        attention_mask: (B, T) optional (not used inside GRU, only for loss masking)
        returns dict logits:
          - logits_p_ini: (B, T) float (raw logits)
          - logits_p_fin: (B, T, 4)
          - logits_cap: (B, T, 4)
        """
        # embeddings already computed, just apply dropout
        emb = self.dropout(embeddings)  # (B, T, E)
        outputs, _ = self.rnn(emb)       # (B, T, H * num_directions)
        outputs = self.dropout(outputs)

        logits_p_ini = self.head_p_ini(outputs).squeeze(-1)   # (B, T)
        logits_p_fin = self.head_p_fin(outputs)              # (B, T, 4)
        logits_cap = self.head_cap(outputs)                  # (B, T, 4)

        return {
            "logits_p_ini": logits_p_ini,
            "logits_p_fin": logits_p_fin,
            "logits_cap": logits_cap
        }