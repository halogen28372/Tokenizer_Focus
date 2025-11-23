import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CanvasViTEncoder(nn.Module):
    """
    Canvas-based Vision Transformer Encoder.
    Standardizes all inputs to 64x64 Canvas.
    
    Input: (B, H, W) integer grid (variable size)
    Output: (B, 4096, d_model) sequence of pixel embeddings (flattened 64x64)
    """
    def __init__(self, 
                 num_colors=10, 
                 d_model=512, 
                 n_head=8, 
                 n_layers=6, 
                 canvas_size=32): # Reduced to 32 for memory (ARC is max 30x30)
        super().__init__()
        
        self.d_model = d_model
        self.canvas_size = canvas_size
        self.seq_len = canvas_size * canvas_size # 1024
        
        # 1. Embeddings
        self.color_emb = nn.Embedding(num_colors + 1, d_model) # +1 for padding/mask
        # 2D Positional Embeddings (Learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        # 2. Transformer Encoder
        # Using efficient attention if available (PyTorch 2.0+)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(d_model)

    def pad_to_canvas(self, x):
        """
        Pads input grid to 64x64.
        Centers the grid or aligns top-left? Top-left is standard for ARC.
        Padding value: 0 (usually background) or specialized padding token?
        We'll use 0 for now (Background/Black), but maybe we should track mask.
        """
        B, H, W = x.shape
        
        if H > self.canvas_size or W > self.canvas_size:
            # Resize/Crop? For ARC, losing data is bad.
            # But if it exceeds 64x64, it's huge.
            # We'll crop for now, but warn.
            # Ideally, we'd scale, but pixel-perfect scaling is hard.
            # Just crop.
            x = x[:, :self.canvas_size, :self.canvas_size]
            H, W = x.shape
            
        pad_h = self.canvas_size - H
        pad_w = self.canvas_size - W
        
        # Pad (Left, Right, Top, Bottom)
        # F.pad uses (last_dim_left, last_dim_right, 2nd_last_left, ...)
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), value=0) # 0 is usually black/background
        
        return x_padded

    def forward(self, x):
        """
        Args:
            x: (B, H, W) integer grid
        Returns:
            (B, 4096, d_model) encoded sequence
        """
        # 1. Create Mask BEFORE padding (True = ignore, False = keep)
        # shape: (B, H, W) -> but we need (B, L) where L=Canvas*Canvas
        B, H, W = x.shape
        
        # Initialize mask with TRUE (ignore)
        # We create a full canvas mask
        mask = torch.ones((B, self.canvas_size, self.canvas_size), dtype=torch.bool, device=x.device)
        
        # Set active region to FALSE (keep)
        # We need to be careful about batch items having different H,W.
        # But here x is a tensor (B,H,W), so H,W are same for the batch.
        # If we were processing a list, we'd need a list of masks.
        # The dataloader usually pads to max in batch or we process 1 by 1.
        # Assuming x is a dense tensor, valid region is [:H, :W] relative to the input x?
        # Wait, pad_to_canvas pads from H,W to canvas_size.
        # If x comes from a batch where things were already padded, we don't know original H,W per item.
        # BUT, in this pipeline, x usually enters as (B, H_actual, W_actual) if batch_size=1,
        # or if batch_size > 1, they are padded by collate_fn.
        # If collate_fn pads, we don't know the real H,W here unless passed.
        # However, pad_to_canvas pads x (which might already be padded?) NO.
        # Our current flow assumes x is the raw grid. 
        # If B>1, `collate_fn` pads to max in batch. 
        # But `pad_to_canvas` pads to 32x32.
        # We'll assume valid pixels are non-negative (they are). 
        # Ideally we should pass lengths.
        # For now, we assume the input x contains valid data up to H,W.
        
        if H <= self.canvas_size and W <= self.canvas_size:
            mask[:, :H, :W] = False
        else:
            # If input was larger and we crop, everything is valid
            mask[:, :, :] = False
            
        # Flatten mask to (B, 1024)
        mask_flat = mask.reshape(B, -1)

        # 2. Pad to Canvas
        x_canvas = self.pad_to_canvas(x) # (B, 32, 32)
        
        # 3. Flatten
        x_flat = x_canvas.reshape(x.shape[0], -1) # (B, 1024)
        
        # 4. Embed
        x_emb = self.color_emb(x_flat) # (B, 1024, D)
        x_emb = x_emb + self.pos_emb
        
        # 5. Transformer with Mask
        x_encoded = self.transformer(x_emb, src_key_padding_mask=mask_flat)
        x_encoded = self.norm(x_encoded)
        
        return x_encoded

