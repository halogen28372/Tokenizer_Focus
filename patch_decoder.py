import torch
import torch.nn as nn

class PatchTransformerDecoder(nn.Module):
    """
    Standard Transformer Decoder that predicts pixels/patches directly.
    Replaces the complex Differentiable Renderer.
    
    Input: 
    - Memory (from Encoder/Rule Inductor): (B, L_in, D)
    
    Output:
    - Logits: (B, H, W, Num_Colors)
    
    Mechanism:
    - Uses learnable queries for the output grid (since size is fixed/known or inferred).
    - For variable output sizes, we can:
        a) Predict a special "End of Object" token (complex)
        b) Or just predict a fixed max size grid and mask (simplest for Baseline).
    - Here we use a fixed set of learnable queries representing the output grid slots.
    """
    def __init__(self, 
                 num_colors=10, 
                 d_model=512, 
                 n_head=8, 
                 n_layers=6, 
                 max_len=1024): # 32x32 output
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.num_colors = num_colors
        
        # 1. Learnable Queries ("Canvas slots")
        # The decoder asks: "What goes in position (0,0)? What goes in (0,1)?"
        # interacting with the rule-enriched input memory.
        self.query_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # 2. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # 3. Output Head (Pixel classification)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_colors)

    def forward(self, memory, output_shape=None):
        """
        Args:
            memory: (B, L_in, D) - From Rule Inductor (Encoded Input + Rule Context)
            output_shape: tuple (H, W) - Desired output dimensions.
                          If None, defaults to 32x32 or max_len.
        Returns:
            logits: (B, H, W, Num_Colors)
        """
        B = memory.shape[0]
        
        # Prepare Queries
        # If we knew the target H,W, we could slice the positional embeddings intelligently.
        # For now, we just take the first H*W queries.
        if output_shape is None:
            H, W = 30, 30 # Standard ARC max
        else:
            H, W = output_shape
            
        L_out = H * W
        if L_out > self.max_len:
            raise ValueError(f"Requested output size {H}x{W}={L_out} exceeds max_len {self.max_len}")
            
        # (B, L_out, D)
        tgt = self.query_emb[:, :L_out, :].expand(B, -1, -1)
        
        # Decode
        # tgt is the Query, memory is Key/Value
        out_seq = self.transformer_decoder(tgt, memory)
        out_seq = self.norm(out_seq)
        
        # Classification
        logits = self.head(out_seq) # (B, L_out, Num_Colors)
        
        # Reshape to Grid
        logits = logits.view(B, H, W, self.num_colors)
        
        return logits

if __name__ == "__main__":
    # Test
    decoder = PatchTransformerDecoder()
    memory = torch.randn(1, 100, 512) # 100 input pixels
    
    # Predict a 10x10 grid
    logits = decoder(memory, output_shape=(10, 10))
    
    print(f"Memory: {memory.shape}")
    print(f"Output Logits: {logits.shape}") # Should be (1, 10, 10, 10)
    print("Patch Decoder Test passed!")

