import torch
import torch.nn as nn
from .layers.common import RMSNorm, FeedForward
from .layers.lcp import LocalContextProcessor
from .layers.gmg import GlobalMemoryGate
from .layers.ccu import ContextCompressionUnit
from .layers.ff import FeedbackFusion

class Efficia1Block(nn.Module):
    """
    A single block of the EFFICIA-1 model, combining all components.
    """
    def __init__(self, dim, compressed_dim, heads, ff_mult, window_size):
        super().__init__()
        self.lcp = LocalContextProcessor(dim, heads=heads, window_size=window_size)
        self.norm1 = RMSNorm(dim)
        
        self.gmg = GlobalMemoryGate(dim, heads=heads)
        self.norm2 = RMSNorm(dim)

        self.ccu = ContextCompressionUnit(dim, compressed_dim)
        # No norm after CCU as it produces a state, not a sequence modification

        self.ff = FeedbackFusion(dim, compressed_dim)
        self.norm3 = RMSNorm(dim)
        
        self.ffn = FeedForward(dim, int(dim * ff_mult))
        self.norm4 = RMSNorm(dim)

    def forward(self, x, global_memory, compressed_state):
        # 1. Local Context Processor
        lcp_out = x + self.lcp(self.norm1(x))
        
        # 2. Global Memory Gate
        gmg_out, new_memory_candidate = self.gmg(self.norm2(lcp_out), global_memory)
        
        # 3. Context Compression Unit
        new_compressed_state = self.ccu(gmg_out, compressed_state)
        
        # 4. Feedback Fusion
        fused_out = self.ff(lcp_out, gmg_out, new_compressed_state)
        
        # Add & Norm before FFN
        h = fused_out + self.norm3(fused_out)
        
        # 5. Feed Forward
        out = h + self.ffn(self.norm4(h))
        
        return out, new_memory_candidate, new_compressed_state

class Efficia1(nn.Module):
    """
    The full EFFICIA-1 model.
    """
    def __init__(self, num_tokens, dim, depth, compressed_dim, heads, window_size=256, ff_mult=4, mem_size=1024):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.mem_size = mem_size
        
        self.layers = nn.ModuleList([
            Efficia1Block(dim, compressed_dim, heads, ff_mult, window_size) for _ in range(depth)
        ])
        
        self.norm = RMSNorm(dim)
        self.output_head = nn.Linear(dim, num_tokens)

    def forward(self, x, global_memory=None, compressed_state=None):
        b, n = x.shape
        device = x.device
        
        x = self.token_emb(x)
        
        # Initialize states if not provided
        if global_memory is None:
            global_memory = torch.zeros(b, self.mem_size, x.size(-1), device=device)
        if compressed_state is None:
            compressed_dim = self.layers[0].ccu.compressed_dim
            compressed_state = torch.zeros(b, compressed_dim, device=device)

        # Process through layers
        memory_candidates = []
        for layer in self.layers:
            x, memory_candidate, compressed_state = layer(x, global_memory, compressed_state)
            memory_candidates.append(memory_candidate)

        # --- Memory Update ---
        # A simple FIFO buffer strategy for updating the global memory.
        # Concatenate all candidates from all layers.
        all_candidates = torch.cat(memory_candidates, dim=1) # Shape: [b, depth, dim]
        
        # How many new memory tokens we have
        num_new_mems = all_candidates.shape[1]
        
        # Make space in the buffer by removing the oldest tokens
        if num_new_mems > 0 and self.mem_size > 0:
            # Ensure we don't try to keep more memory than the buffer size
            if num_new_mems >= self.mem_size:
                updated_memory = all_candidates[:, -self.mem_size:, :]
            else:
                updated_memory = torch.cat((global_memory[:, num_new_mems:], all_candidates), dim=1)
        else:
            updated_memory = global_memory

        x = self.norm(x)
        logits = self.output_head(x)
        
        return logits, updated_memory, compressed_state

    def save_checkpoint(self, path, optimizer, epoch):
        """Saves the model and optimizer state."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, optimizer=None):
        """Loads the model and optimizer state."""
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        
        if epoch > 0:
            print(f"Checkpoint loaded from {path} at epoch {epoch}")
        else:
            print(f"Checkpoint loaded from {path}")
            
        return epoch
