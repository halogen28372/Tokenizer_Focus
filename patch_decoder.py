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
                 max_len=1024): # 32x32 = 1024
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        self.num_colors = num_colors
        
        # 1. Learnable Queries ("Canvas slots")
        # Reshapeable 2D queries: (1, H_max, W_max, D)
        # We initialize as 1D for compatibility but treat as 2D
        self.canvas_size = int(max_len**0.5) # 32
        self.query_emb = nn.Parameter(torch.randn(1, self.canvas_size, self.canvas_size, d_model) * 0.02)
        
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
                          If None, defaults to canvas_size (64x64).
        Returns:
            logits: (B, H, W, Num_Colors)
        """
        B = memory.shape[0]
        
        if output_shape is None:
            H, W = self.canvas_size, self.canvas_size
        else:
            H, W = output_shape
            
        if H > self.canvas_size or W > self.canvas_size:
            # Warn or error? Error for now to avoid silent failures
             raise ValueError(f"Requested output size {H}x{W} exceeds canvas size {self.canvas_size}x{self.canvas_size}")

        # Crop queries to target size: (1, H, W, D)
        tgt_queries = self.query_emb[:, :H, :W, :] 
        
        # Flatten to sequence: (1, L_out, D)
        tgt_flat = tgt_queries.reshape(1, -1, self.d_model)
        
        # Expand to batch: (B, L_out, D)
        tgt = tgt_flat.expand(B, -1, -1)
        
        # Decode
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
