import torch
import torch.nn as nn

class FeedbackFusion(nn.Module):
    """
    Dynamically combines information from LCP, GMG, and CCU.
    """
    def __init__(self, dim, compressed_dim):
        super().__init__()
        # Projection layer to match CCU output dimension to the main model dimension
        self.ccu_projection = nn.Linear(compressed_dim, dim)
        # A simple gating mechanism to fuse the different information sources
        self.gate = nn.Linear(dim * 3, 3)

    def forward(self, lcp_out, gmg_out, ccu_out):
        """
        lcp_out: Output from Local Context Processor [batch, seq_len, dim]
        gmg_out: Output from Global Memory Gate [batch, seq_len, dim]
        ccu_out: Output from Context Compression Unit [batch, compressed_dim]
        """
        # Project and expand CCU output to match sequence length and dimension
        projected_ccu = self.ccu_projection(ccu_out)
        ccu_out_expanded = projected_ccu.unsqueeze(1).expand_as(lcp_out)
        
        # Concatenate all inputs
        combined_input = torch.cat((lcp_out, gmg_out, ccu_out_expanded), dim=-1)
        
        # Calculate gates
        gates = torch.softmax(self.gate(combined_input), dim=-1)
        
        # Split gates for each input source
        g_lcp, g_gmg, g_ccu = gates.chunk(3, dim=-1)
        
        # Fuse the outputs
        fused_output = g_lcp * lcp_out + g_gmg * gmg_out + g_ccu * ccu_out_expanded
        
        return fused_output
