import torch
import torch.nn as nn
from einops import rearrange

class GlobalMemoryGate(nn.Module):
    """
    GMG acts as an interface to the long-term memory.
    It uses cross-attention to read relevant information from memory and
    a gated mechanism to update the memory with new context.
    """
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.tensor(1.0 / (dim_head ** 0.5)), requires_grad=False)  # Stabilní inicializace
        inner_dim = heads * dim_head

        # For reading from memory (cross-attention)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # For gating and fusing
        self.gate = nn.Linear(dim * 2, dim)

        # Inicializace vah
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    def forward(self, x, global_memory):
        """
        x: current context from LCP [batch, seq_len, dim]
        global_memory: memory state from previous steps [batch, mem_len, dim]
        """
        if global_memory is None:
            # If no memory is provided, we can't read from it.
            # We'll just return the input and a candidate for the new memory.
            return x, x

        # --- Read from Memory (Cross-Attention) ---
        q = self.to_q(x)
        k, v = self.to_kv(global_memory).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v)
        )

        # Použití matmul místo einsum pro efektivitu a stabilitu
        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        sim = torch.clamp(sim, min=-10.0, max=10.0)  # Omezení pro stabilitu
        attn = sim.softmax(dim=-1)

        retrieved_mem = torch.matmul(attn, v)
        retrieved_mem = rearrange(retrieved_mem, 'b h n d -> b n (h d)')
        retrieved_mem = self.to_out(retrieved_mem)

        # --- Fuse current context with retrieved memory ---
        gate_input = torch.cat((x, retrieved_mem), dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        
        fused_context = (1 - gate) * x + gate * retrieved_mem

        # --- Prepare a candidate for memory update ---
        new_memory_candidate = x.mean(dim=1, keepdim=True)  # Summarize to a single vector

        return fused_context, new_memory_candidate
