import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LinearAttention(nn.Module):
    """
    Linear Attention using ELU activation for stability, a common practice.
    """
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Use ELU activation for queries and keys for stability
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention mechanism
        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class LocalContextProcessor(nn.Module):
    """
    LCP processes tokens in local windows using linear attention.
    """
    def __init__(self, dim, heads=4, dim_head=64, window_size=256):
        super().__init__()
        self.attn = LinearAttention(dim, heads, dim_head)
        self.window_size = window_size

    def forward(self, x):
        b, n, d = x.shape
        ws = self.window_size

        # Pad sequence to be divisible by window size
        padding = (ws - n % ws) % ws
        if padding > 0:
            x = F.pad(x, (0, 0, 0, padding), 'constant', 0)
        
        # Rearrange into windows
        windows = rearrange(x, 'b (w n_w) d -> (b w) n_w d', n_w=ws)
        
        # Apply attention to each window
        processed_windows = self.attn(windows)
        
        # Merge windows back
        out = rearrange(processed_windows, '(b w) n_w d -> b (w n_w) d', b=b)

        # Remove padding
        if padding > 0:
            out = out[:, :-padding, :]

        return out
