"""
Tokenizer-based encoder that uses the ARC/LAGO tokenizer to extract structural features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_lago_tokenizer import ArcLagoTokenizer, TokenizerConfig
from config import config


class TokenizerEncoder(nn.Module):
    """
    Encodes grids using the ARC/LAGO tokenizer to extract structural features.
    
    This encoder:
    1. Tokenizes the input grid to extract shapes, relations, and metadata
    2. Embeds each token type and its parameters
    3. Aggregates token embeddings into a fixed-size feature vector
    """
    
    def __init__(self, 
                 num_colors=None,
                 d_feat=None,
                 max_tokens=None,
                 token_dim=None,
                 use_relations=None,
                 use_meta=None):
        super().__init__()
        
        self.num_colors = num_colors or config.NUM_COLORS
        self.d_feat = d_feat or config.D_FEAT
        self.max_tokens = max_tokens or config.TOKENIZER_MAX_TOKENS
        self.token_dim = token_dim or config.TOKENIZER_TOKEN_DIM
        self.use_relations = use_relations if use_relations is not None else config.TOKENIZER_USE_RELATIONS
        self.use_meta = use_meta if use_meta is not None else config.TOKENIZER_USE_META
        
        # Initialize tokenizer
        tokenizer_cfg = TokenizerConfig(
            max_layers=4,
            emit_symmetry_meta=self.use_meta,
            emit_relations=self.use_relations,
            assume_rect_occlusion_prior=True,
            merge_diagonal_lines=True
        )
        self.tokenizer = ArcLagoTokenizer(tokenizer_cfg)
        
        # Token type embeddings
        self.shape_types = [
            'RECT', 'HOLLOW_RECT', 'BORDER', 'LINE', 'DIAG_LINE', 'DIAG_CROSS_X',
            'TETROMINO', 'PENTOMINO', 'CHECKER', 'CONCENTRIC_RECTS', 'C_SHAPE',
            'SPARSE_DOTS', 'CROSSHATCH', 'RADIAL', 'ZIGZAG', 'SPIRAL', 'REGION'
        ]
        self.type_embedding = nn.Embedding(len(self.shape_types) + 2, self.token_dim)  # +2 for REL and META
        self.type_to_id = {t: i for i, t in enumerate(self.shape_types)}
        self.rel_type_id = len(self.shape_types)
        self.meta_type_id = len(self.shape_types) + 1
        
        # Color embedding
        self.color_embedding = nn.Embedding(self.num_colors, 32)
        
        # Layer embedding (for occlusion)
        self.layer_embedding = nn.Embedding(4, 16)  # max 4 layers
        
        # Parameter encoders for different shape parameters
        self.param_encoders = nn.ModuleDict({
            'spatial': nn.Linear(4, 32),  # x, y, w, h
            'scale': nn.Linear(1, 16),    # scale factor
            'thickness': nn.Linear(1, 16), # border thickness
            'count': nn.Linear(1, 16),    # various counts
            'orientation': nn.Embedding(8, 16),  # H, V, NE, NW, SE, SW, diagonal, anti_diagonal
        })
        
        # Relation encoder
        if self.use_relations:
            self.rel_types = ['inside', 'overlaps', 'touches', 'left_of', 'right_of', 
                            'above', 'below', 'aligned_row', 'aligned_col']
            self.rel_embedding = nn.Embedding(len(self.rel_types), 32)
            self.rel_to_id = {r: i for i, r in enumerate(self.rel_types)}
        
        # Meta encoder
        if self.use_meta:
            self.meta_types = ['symmetry_axis', 'rotational', 'tiling']
            self.meta_embedding = nn.Embedding(len(self.meta_types), 32)
            self.meta_to_id = {m: i for i, m in enumerate(self.meta_types)}
        
        # Token fusion
        # Each token will have: type_emb + color_emb + layer_emb + param_emb
        self.token_fusion = nn.Sequential(
            nn.Linear(self.token_dim + 32 + 16 + 64, self.token_dim),  # Fuse to token_dim
            nn.LayerNorm(self.token_dim),
            nn.GELU()
        )
        
        # Positional encoding for tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_tokens, self.token_dim) * 0.02)
        
        # Self-attention to aggregate tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Final projection to feature dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.token_dim, self.d_feat),
            nn.LayerNorm(self.d_feat),
            nn.GELU(),
            nn.Linear(self.d_feat, self.d_feat)
        )
        
    def encode_shape_params(self, params):
        """Encode shape parameters into a fixed-size vector."""
        param_feats = []
        
        # Spatial parameters (x, y, w, h)
        spatial = torch.zeros(4, dtype=torch.float32)
        if 'x' in params:
            spatial[0] = float(params['x'])
        if 'y' in params:
            spatial[1] = float(params['y'])
        if 'w' in params:
            spatial[2] = float(params.get('w', 1))
        if 'h' in params:
            spatial[3] = float(params.get('h', 1))
        # Handle center coordinates
        if 'cx' in params:
            spatial[0] = float(params['cx'])
        if 'cy' in params:
            spatial[1] = float(params['cy'])
        param_feats.append(self.param_encoders['spatial'](spatial))
        
        # Scale
        scale = torch.tensor([params.get('scale', 1.0)], dtype=torch.float32)
        param_feats.append(self.param_encoders['scale'](scale))
        
        # Thickness/border
        thickness = torch.tensor([float(params.get('border_thickness', params.get('thickness', 0)))], dtype=torch.float32)
        param_feats.append(self.param_encoders['thickness'](thickness))
        
        # Count parameters
        count = torch.tensor([float(params.get('num_layers', params.get('num_spokes', 
                                        params.get('coil_count', params.get('size', 0)))))], dtype=torch.float32)
        param_feats.append(self.param_encoders['count'](count))
        
        # Pad to fixed size
        param_vec = torch.cat(param_feats, dim=-1)
        if param_vec.shape[-1] < 64:
            param_vec = F.pad(param_vec, (0, 64 - param_vec.shape[-1]))
        else:
            param_vec = param_vec[..., :64]
        
        return param_vec
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W) integer grid
        Returns:
            (B, d_feat) feature vector
        """
        B, H, W = x.shape
        device = x.device
        
        all_features = []
        
        for b in range(B):
            # Convert to list of lists for tokenizer
            grid = x[b].cpu().numpy().tolist()
            
            # Tokenize
            result = self.tokenizer.tokenize(grid)
            shapes = result['shapes']
            relations = result['relations'] if self.use_relations else []
            meta = result['meta'] if self.use_meta else []
            
            # Limit number of tokens
            total_tokens = len(shapes) + len(relations) + len(meta)
            if total_tokens > self.max_tokens:
                # Prioritize shapes, then relations, then meta
                shapes = shapes[:min(len(shapes), self.max_tokens - 10)]
                relations = relations[:min(len(relations), 10)]
                meta = meta[:min(len(meta), 5)]
            
            token_embeddings = []
            
            # Encode shape tokens
            for shape_dict in shapes:
                # Type embedding
                shape_type = shape_dict['type']
                if shape_type in self.type_to_id:
                    type_id = torch.tensor([self.type_to_id[shape_type]], device=device)
                else:
                    type_id = torch.tensor([self.type_to_id['REGION']], device=device)  # fallback
                type_emb = self.type_embedding(type_id).squeeze(0)
                
                # Color embedding
                color_id = torch.tensor([shape_dict['color']], device=device)
                color_emb = self.color_embedding(color_id).squeeze(0)
                
                # Layer embedding
                layer_id = torch.tensor([min(shape_dict['layer'], 3)], device=device)
                layer_emb = self.layer_embedding(layer_id).squeeze(0)
                
                # Parameter embedding
                param_emb = self.encode_shape_params(shape_dict).to(device)
                
                # Combine
                token = torch.cat([type_emb, color_emb, layer_emb, param_emb])
                token_embeddings.append(self.token_fusion(token))
            
            # Encode relation tokens
            if self.use_relations:
                for rel_dict in relations:
                    type_emb = self.type_embedding(torch.tensor([self.rel_type_id], device=device)).squeeze(0)
                    
                    rel_type = rel_dict['rel']
                    if rel_type in self.rel_to_id:
                        rel_id = torch.tensor([self.rel_to_id[rel_type]], device=device)
                        rel_emb = self.rel_embedding(rel_id).squeeze(0)
                    else:
                        rel_emb = torch.zeros(32, device=device)
                    
                    # Pad to match token fusion input size
                    token = torch.cat([type_emb, rel_emb, 
                                     torch.zeros(16 + 64, device=device)])  # padding for layer + params
                    token_embeddings.append(self.token_fusion(token))
            
            # Encode meta tokens
            if self.use_meta:
                for meta_dict in meta:
                    type_emb = self.type_embedding(torch.tensor([self.meta_type_id], device=device)).squeeze(0)
                    
                    meta_type = meta_dict.get('meta', '')
                    if meta_type in self.meta_to_id:
                        meta_id = torch.tensor([self.meta_to_id[meta_type]], device=device)
                        meta_emb = self.meta_embedding(meta_id).squeeze(0)
                    else:
                        meta_emb = torch.zeros(32, device=device)
                    
                    # Pad to match token fusion input size
                    token = torch.cat([type_emb, meta_emb,
                                     torch.zeros(16 + 64, device=device)])  # padding for layer + params
                    token_embeddings.append(self.token_fusion(token))
            
            # Pad or truncate to max_tokens
            if len(token_embeddings) == 0:
                # Empty tokenization - use a learned "empty" token
                token_embeddings = [torch.zeros(self.token_dim, device=device)]
            
            tokens_tensor = torch.stack(token_embeddings)
            num_tokens = tokens_tensor.shape[0]
            
            if num_tokens < self.max_tokens:
                # Pad with zeros
                padding = torch.zeros(self.max_tokens - num_tokens, self.token_dim, device=device)
                tokens_tensor = torch.cat([tokens_tensor, padding], dim=0)
            else:
                tokens_tensor = tokens_tensor[:self.max_tokens]
            
            # Add positional encoding
            tokens_tensor = tokens_tensor.unsqueeze(0)  # (1, max_tokens, token_dim)
            tokens_tensor = tokens_tensor + self.pos_embedding[:, :self.max_tokens]
            
            # Self-attention to aggregate
            attended, _ = self.self_attn(tokens_tensor, tokens_tensor, tokens_tensor)
            
            # Global pooling
            pooled = attended.mean(dim=1)  # (1, token_dim)
            
            # Project to output dimension
            features = self.output_proj(pooled)  # (1, d_feat)
            all_features.append(features)
        
        return torch.cat(all_features, dim=0)  # (B, d_feat)


class HybridGridEncoder(nn.Module):
    """
    Hybrid encoder that combines tokenizer-based structural features with 
    traditional CNN-based visual features.
    Supports both ARC/LAGO tokenizer and Object-Aware tokenizer.
    """
    
    def __init__(self, num_colors=None, d_feat=None, use_cnn=True, use_tokenizer=True):
        super().__init__()
        
        self.num_colors = num_colors or config.NUM_COLORS
        self.d_feat = d_feat or config.D_FEAT
        self.use_cnn = use_cnn
        self.use_tokenizer = use_tokenizer
        
        if self.use_tokenizer:
            tokenizer_type = getattr(config, 'TOKENIZER_TYPE', 'arc_lago')
            
            if tokenizer_type == 'object_aware':
                # Use Object-Aware tokenizer
                from object_aware_tokenizer import ObjectAwareEncoder
                self.tokenizer_encoder = ObjectAwareEncoder(
                    num_colors=self.num_colors,
                    d_feat=self.d_feat,
                    embed_dim=getattr(config, 'OBJECT_AWARE_EMBED_DIM', 64),
                    shape_embed_dim=getattr(config, 'OBJECT_AWARE_SHAPE_EMBED_DIM', 32)
                )
            else:
                # Use ARC/LAGO tokenizer (default)
                self.tokenizer_encoder = TokenizerEncoder(
                    num_colors=self.num_colors,
                    d_feat=self.d_feat
                )
        
        if self.use_cnn:
            # Simple CNN for visual features
            self.color_embed = nn.Embedding(self.num_colors, 64)
            self.cnn = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.GroupNorm(16, 256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.cnn_proj = nn.Linear(256, self.d_feat)
        
        # Fusion layer if using both
        if self.use_cnn and self.use_tokenizer:
            self.fusion = nn.Sequential(
                nn.Linear(self.d_feat * 2, self.d_feat),
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
        features = []
        
        if self.use_tokenizer:
            tok_feat = self.tokenizer_encoder(x)
            features.append(tok_feat)
        
        if self.use_cnn:
            # CNN path
            z = self.color_embed(x).permute(0, 3, 1, 2)  # (B, 64, H, W)
            h = self.cnn(z).flatten(1)  # (B, 256)
            cnn_feat = self.cnn_proj(h)  # (B, d_feat)
            features.append(cnn_feat)
        
        if len(features) == 1:
            return features[0]
        else:
            # Fuse both features
            combined = torch.cat(features, dim=-1)  # (B, 2*d_feat)
            return self.fusion(combined)  # (B, d_feat)
