import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextCompressionUnit(nn.Module):
    """
    CCU compresses the long-term context into a small, dense vector ("consciousness").
    Implemented with a GRU-like mechanism and learned pooling.
    """
    def __init__(self, dim, compressed_dim):
        super().__init__()
        self.compressed_dim = compressed_dim
        # Learned pooling: a small network to decide the importance of each token
        self.pooling_weights = nn.Linear(dim, 1)
        self.gru_cell = nn.GRUCell(dim, compressed_dim)

    def forward(self, x, compressed_state):
        """
        x: context from previous layer [batch, seq_len, dim]
        compressed_state: the "consciousness" vector from the previous batch [batch, compressed_dim]
        """
        # --- Learned Pooling ---
        # Get weights for each token in the sequence
        weights = self.pooling_weights(x).squeeze(-1)
        attn = F.softmax(weights, dim=-1)
        
        # Create a weighted average of the sequence
        # attn is [batch, seq_len], x is [batch, seq_len, dim]
        # We need to unsqueeze attn to multiply
        current_context_summary = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        
        # --- Recurrent Update ---
        if compressed_state is None:
            # Initialize state if it's the first step
            batch_size = x.size(0)
            device = x.device
            compressed_state = torch.zeros(batch_size, self.compressed_dim, device=device)
            
        new_compressed_state = self.gru_cell(current_context_summary, compressed_state)
        
        return new_compressed_state
