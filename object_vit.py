import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import label

class ObjectAugmentedViTEncoder(nn.Module):
    """
    Object-Augmented Vision Transformer (ViT) Encoder.
    
    Input: (B, H, W) integer grid
    Output: (B, H*W, d_model) sequence of patch embeddings
    
    Features:
    - Treats each pixel as a patch (1x1 patch size for ARC)
    - Injects 'Object_ID' embeddings derived from fast Connected Component Analysis
    - Uses standard Transformer Encoder for global reasoning
    """
    def __init__(self, 
                 num_colors=10, 
                 d_model=512, 
                 n_head=8, 
                 n_layers=6, 
                 max_objects=64, 
                 max_len=1024): # 32x32 = 1024
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 1. Embeddings
        self.color_emb = nn.Embedding(num_colors, d_model)
        self.object_emb = nn.Embedding(max_objects, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Optional: Output projection or norm
        self.norm = nn.LayerNorm(d_model)

    def _get_object_ids(self, x_np):
        """
        Compute object IDs for a single grid using Connected Components.
        Strategy: Group contiguous pixels of the SAME COLOR.
        
        Args:
            x_np: (H, W) numpy integer array
        Returns:
            (H, W) integer array of object IDs
        """
        H, W = x_np.shape
        object_map = np.zeros((H, W), dtype=np.int32)
        current_id = 1 # ID 0 is reserved for "no object" or background if needed
        
        # Find unique colors present
        unique_colors = np.unique(x_np)
        
        for c in unique_colors:
            # Create binary mask for this color
            mask = (x_np == c)
            # Label connected components
            # structure=[[0,1,0],[1,1,1],[0,1,0]] defines 4-connectivity
            labeled_array, num_features = label(mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
            
            if num_features > 0:
                # Add current_id offset to non-zero labels
                labeled_array[labeled_array > 0] += (current_id - 1)
                object_map += labeled_array
                current_id += num_features
                
        # Clip to max_objects - 1 (ID 0 is valid, so max index is max_objects-1)
        # We use modulo or clip. Clipping is safer to avoid random collisions.
        # Reserve ID 0 for... actually let's just mod it to be safe if it explodes.
        # But usually ARC has < 20 objects.
        object_map = object_map % self.object_emb.num_embeddings
        return object_map

    def forward(self, x):
        """
        Args:
            x: (B, H, W) integer grid
        Returns:
            (B, L, d_model) encoded sequence
        """
        B, H, W = x.shape
        device = x.device
        
        # 1. Compute Object IDs (on CPU/Numpy for scipy)
        # Note: This is a bottleneck if B is large, but for ARC B=1 usually.
        x_cpu = x.detach().cpu().numpy()
        obj_ids_list = []
        for b in range(B):
            obj_ids = self._get_object_ids(x_cpu[b])
            obj_ids_list.append(obj_ids)
        
        obj_ids = torch.tensor(np.stack(obj_ids_list), device=device).long() # (B, H, W)
        
        # 2. Flatten to Sequence
        # (B, H, W) -> (B, H*W)
        x_flat = x.reshape(B, -1)
        obj_ids_flat = obj_ids.reshape(B, -1)
        L = x_flat.shape[1]
        
        # 3. Embed
        # E = E_color + E_object + E_pos
        e_color = self.color_emb(x_flat) # (B, L, D)
        e_obj = self.object_emb(obj_ids_flat) # (B, L, D)
        
        # Positional embedding (truncate to current length)
        # We assume a raster scan order. 
        # For variable sized grids, we might want 2D pos encodings, 
        # but 1D learnable is standard for ViT.
        if L > self.max_len:
            # Fallback for huge grids: interpolate or crop. 
            # ARC is usually 30x30=900 < 1024.
             e_pos = self.pos_emb[:, :L, :]
        else:
             e_pos = self.pos_emb[:, :L, :]
             
        # Sum embeddings
        x_emb = e_color + e_obj + e_pos
        
        # 4. Transformer
        x_encoded = self.transformer(x_emb)
        x_encoded = self.norm(x_encoded)
        
        return x_encoded

if __name__ == "__main__":
    # Simple test
    encoder = ObjectAugmentedViTEncoder()
    x = torch.randint(0, 10, (1, 10, 10)) # 10x10 grid
    out = encoder(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print("Test passed!")

