"""
Object-Aware Tokenizer: A dual-stream encoder that combines pixel-level and object-level features.

This tokenizer:
1. Pixel Stream: Standard CNN for local pixel context
2. Shape Stream: Extracts objects via connected components and encodes their shapes
3. Relative Position: Encodes position within each object's bounding box
4. Fusion: Combines all streams into enhanced spatial features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label, regionprops


class ObjectAwareTokenizer(nn.Module):
    def __init__(self, 
                 input_channels=11,  # 0-9 colors + padding
                 embed_dim=64, 
                 shape_embed_dim=32):
        super().__init__()
        
        # 1. The "Dumb" Stream (Local Pixel Context)
        # Standard 1x1 or 3x3 conv to embed the raw grid colors
        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        
        # 2. The "Shape" Stream (Object Context)
        # A mini-CNN that looks at a cropped shape and outputs a vector
        self.shape_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Input is binary mask of shape
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4)),  # Force to fixed size
            nn.Flatten(),
            nn.Linear(16*4*4, shape_embed_dim),
            nn.ReLU()
        )
        
        # 3. Positional Encoding within the Object
        # (y_rel, x_rel) normalized coordinates inside the object bbox
        self.rel_pos_encoder = nn.Linear(2, shape_embed_dim)
        
        # 4. Fusion
        self.fusion = nn.Linear(embed_dim + 2*shape_embed_dim, embed_dim)
        
        self.embed_dim = embed_dim
        self.shape_embed_dim = shape_embed_dim

    def get_object_crops(self, grid_np):
        """
        Extracts object bounding boxes using connected components.
        Returns list of (mask_crop, slice_y, slice_x, rel_coords_grid).
        """
        # grid_np: (H, W)
        labeled = label(grid_np, connectivity=1, background=0)
        regions = regionprops(labeled)
        
        object_meta = []
        
        for r in regions:
            # Extract binary mask of the object
            minr, minc, maxr, maxc = r.bbox
            h, w = maxr-minr, maxc-minc
            
            # Create the mask crop (1 for object, 0 for bg)
            mask_crop = r.image.astype(np.float32)  # (h, w)
            
            # Create relative coordinates grid for this object
            # range [0, 1]
            yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
            rel_coords = np.stack([yy, xx], axis=-1)  # (h, w, 2)
            
            object_meta.append({
                'mask': mask_crop,
                'bbox': (minr, maxr, minc, maxc),
                'rel_coords': rel_coords
            })
            
        return object_meta

    def forward(self, x):
        """
        x: (B, C, H, W) - One-hot encoded grid or raw integers
        """
        B, C, H, W = x.shape
        device = x.device
        
        # --- Stream 1: Standard Pixel Embedding ---
        pixel_emb = self.pixel_encoder(x)  # (B, E, H, W)
        
        # --- Stream 2: Shape Injection (non-differentiable but informative) ---
        # Initialize shape embedding map (requires_grad=False to avoid breaking pixel stream)
        shape_map = torch.zeros(B, self.shape_embed_dim, H, W, device=device, requires_grad=False)
        rel_pos_map = torch.zeros(B, self.shape_embed_dim, H, W, device=device, requires_grad=False)
        
        # Process each item in batch (cannot easily batch across B due to variable object counts)
        # Note: Shape stream is non-differentiable but provides useful inductive bias
        with torch.no_grad():
            x_argmax = torch.argmax(x, dim=1).cpu().numpy()
        
        for b in range(B):
            objects = self.get_object_crops(x_argmax[b])
            
            if not objects:
                continue
                
            # Prepare batch of shape crops for the Shape Encoder
            # shape_crops: (N_obj, 1, h, w) -> padded to max or handled via adaptive pool
            # For simplicity here, we loop (in real impl, pad and batch)
            
            for obj in objects:
                # 1. Embed the Shape (Invariant to position)
                mask_tensor = torch.tensor(obj['mask'], device=device, requires_grad=False).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
                shape_vec = self.shape_encoder(mask_tensor)  # (1, shape_dim)
                
                # 2. Embed Relative Position
                coords_tensor = torch.tensor(obj['rel_coords'], device=device, requires_grad=False).float()  # (h, w, 2)
                pos_vecs = self.rel_pos_encoder(coords_tensor)  # (h, w, shape_dim)
                pos_vecs = pos_vecs.permute(2, 0, 1)  # (shape_dim, h, w)
                
                # 3. Scatter back to the main grid
                minr, maxr, minc, maxc = obj['bbox']
                
                # Identify pixels belonging to this object in the bbox
                # (r.image is the binary mask)
                mask_bool = torch.tensor(obj['mask'], device=device, requires_grad=False).bool()
                
                # Fill the maps
                # We mask the update so we only write to the pixels that are actually part of the object
                current_shape_slice = shape_map[b, :, minr:maxr, minc:maxc]
                # Broadcast shape_vec (1, D) -> (D, h, w)
                broadcast_shape = shape_vec.view(-1, 1, 1).expand(-1, maxr-minr, maxc-minc)
                
                shape_map[b, :, minr:maxr, minc:maxc] = torch.where(mask_bool, broadcast_shape, current_shape_slice)
                
                current_pos_slice = rel_pos_map[b, :, minr:maxr, minc:maxc]
                rel_pos_map[b, :, minr:maxr, minc:maxc] = torch.where(mask_bool, pos_vecs, current_pos_slice)
        
        # Ensure shape maps are detached to prevent gradient issues
        shape_map = shape_map.detach()
        rel_pos_map = rel_pos_map.detach()
        
        # --- Final Fusion ---
        # Concatenate: Pixel(E) + Shape(S) + RelPos(S)
        # Pixel stream has gradients, shape stream provides inductive bias
        combined = torch.cat([pixel_emb, shape_map, rel_pos_map], dim=1)  # (B, E+2S, H, W)
        
        # Fuse to original dim (or keep expanded)
        # Permute for Linear layer: (B, H, W, Channels)
        combined = combined.permute(0, 2, 3, 1)
        fused = self.fusion(combined)  # (B, H, W, E)
        
        # Reshape back to image format if needed for next layers
        fused = fused.permute(0, 3, 1, 2)  # (B, E, H, W)
        
        # Residual connection: pixel stream + fused features
        # This ensures gradients flow even when shape stream is empty
        out = pixel_emb + fused
        
        return out


class ObjectAwareEncoder(nn.Module):
    """
    Wrapper that uses ObjectAwareTokenizer and converts spatial features to global features.
    
    Input: (B, H, W) integer grid
    Output: (B, d_feat) feature vector
    """
    
    def __init__(self, num_colors=None, d_feat=None, embed_dim=None, shape_embed_dim=None):
        super().__init__()
        from config import config
        
        self.num_colors = num_colors or config.NUM_COLORS
        self.d_feat = d_feat or config.D_FEAT
        embed_dim = embed_dim or 64
        shape_embed_dim = shape_embed_dim or 32
        
        # Convert integer grid to one-hot
        self.one_hot = nn.Identity()  # We'll do this manually in forward
        
        # Object-aware tokenizer
        self.tokenizer = ObjectAwareTokenizer(
            input_channels=self.num_colors,
            embed_dim=embed_dim,
            shape_embed_dim=shape_embed_dim
        )
        
        # Add Transformer for global reasoning over spatial features
        # This allows the model to understand relationships between distant parts of the grid
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=4, 
                dim_feedforward=embed_dim*4, 
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ),
            num_layers=3  # 3 layers of global reasoning
        )
        
        # Positional encoding for the transformer
        self.pos_emb = nn.Parameter(torch.randn(1, 900, embed_dim) * 0.02) # Max 30x30 grid

        # Global pooling and projection to d_feat
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, self.d_feat),
            nn.LayerNorm(self.d_feat),
            nn.GELU(),
            nn.Linear(self.d_feat, self.d_feat)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W) integer grid
        Returns:
            (B, d_feat) feature vector
        """
        B, H, W = x.shape
        device = x.device
        
        # Convert integer grid to one-hot: (B, H, W) -> (B, num_colors, H, W)
        x_onehot = F.one_hot(x.long(), num_classes=self.num_colors).float()
        x_onehot = x_onehot.permute(0, 3, 1, 2)  # (B, num_colors, H, W)
        
        # Get spatial features from tokenizer
        spatial_feat = self.tokenizer(x_onehot)  # (B, embed_dim, H, W)
        
        # Flatten for transformer: (B, C, H, W) -> (B, H*W, C)
        flat_feat = spatial_feat.flatten(2).permute(0, 2, 1)  # (B, L, C)
        L = flat_feat.shape[1]
        
        # Add positional embedding (interpolate if grid size changes)
        if L <= self.pos_emb.shape[1]:
            pos = self.pos_emb[:, :L, :]
        else:
            # This shouldn't happen for ARC (max 30x30), but for safety:
            pos = F.interpolate(self.pos_emb.permute(0, 2, 1), size=L, mode='linear').permute(0, 2, 1)
            
        flat_feat = flat_feat + pos
        
        # Apply transformer for global reasoning
        trans_feat = self.transformer(flat_feat)  # (B, L, C)
        
        # Global average pooling over sequence length
        global_feat = trans_feat.mean(dim=1)  # (B, C)
        
        # Project to d_feat
        out = self.proj(global_feat)  # (B, d_feat)
        
        return out

