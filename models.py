"""
Neural network architectures for Energy-Based Transformer (EBT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm
from config import config


def l2norm(v):
    """L2 normalize vectors"""
    return F.normalize(v, p=2, dim=-1)


def create_canvas(n_items, H, W, device='cuda'):
    """
    Create a multi-item canvas for tiled layout with shared RoPE and segment IDs.
    
    Args:
        n_items: Number of (input, output) pairs
        H, W: Grid dimensions
        device: Device for tensors
    
    Returns:
        MultiItemCanvas object (if USE_MULTI_ITEM_CANVAS=True) or None
    """
    if not config.USE_MULTI_ITEM_CANVAS:
        return None
    
    try:
        from canvas import MultiItemCanvas  # type: ignore
    except ImportError:
        raise ImportError(
            "canvas module not found. Please create canvas.py with MultiItemCanvas class "
            "or set USE_MULTI_ITEM_CANVAS=False in config."
        )
    
    return MultiItemCanvas(
        n_items=n_items,
        H=H,
        W=W,
        Sx=config.CANVAS_TILE_STRIDE_X,
        Sy=config.CANVAS_TILE_STRIDE_Y,
        Ly=config.CANVAS_LANE_GAP,
        d_seg=config.CANVAS_SEGMENT_DIM,
        use_mask=config.CANVAS_USE_HARD_MASK,
        device=device
    )


def neighbor_features(P, num_classes, scales=(3, 5)):
    """
    Compute color-agnostic neighbor consensus features.
    
    Args:
        P: (B, C, H, W) soft class probabilities (from softmax)
        num_classes: number of classes
        scales: tuple of window sizes (e.g., (3, 5))
    
    Returns:
        features: (B, 4*len(scales), H, W)
            4 features per scale: match, majority, uniformity, variance
    """
    B, C, H, W = P.shape
    feats = []
    
    for k in scales:
        # Create box filter (all ones)
        box = torch.ones(1, 1, k, k, device=P.device, dtype=P.dtype)
        pad = k // 2
        
        # Apply replicate padding first
        P_padded = F.pad(P, (pad, pad, pad, pad), mode='replicate')
        
        # Window sums per class using depthwise convolution
        S = F.conv2d(P_padded, box.expand(C, 1, k, k), bias=None, stride=1, 
                     padding=0, groups=C)  # (B, C, H, W)
        K = float(k * k)
        
        # (1) Center-match (exclude center so min=0, max=1)
        # Average dot product with neighbors: <P_i, sum_j P_j> / K
        match_excl_center = (P * (S - P)).sum(1, keepdim=True) / (K - 1)
        
        # (2) Majority strength: max_c (mean count of class c in window)
        maj = S.max(dim=1, keepdim=True).values / K
        
        # (3) Uniformity: 1 - normalized entropy of mean distribution
        Pbar = S / K
        ent = -(Pbar.clamp_min(1e-8) * Pbar.clamp_min(1e-8).log()).sum(1, keepdim=True)
        uni = 1.0 - ent / math.log(C)
        
        # (4) Variance across window (heterogeneity)
        P_sq_padded = F.pad(P * P, (pad, pad, pad, pad), mode='replicate')
        S2 = F.conv2d(P_sq_padded, box.expand(C, 1, k, k), padding=0, groups=C)
        var = (S2 / K - (Pbar * Pbar)).sum(1, keepdim=True)
        
        feats += [match_excl_center, maj, uni, var]
    
    return torch.cat(feats, dim=1)  # (B, 4*len(scales), H, W)


class NeighborConsensusEncoder(nn.Module):
    """
    Color-agnostic neighbor-aware encoder.
    Produces features that reflect local structure/agreement without 
    leaking color identity.
    """
    
    def __init__(self, num_colors=None, d_neighbor=None, scales=None):
        super().__init__()
        self.num_colors = num_colors or config.NUM_COLORS
        self.d_neighbor = d_neighbor or config.NCE_DIM
        self.scales = scales or config.NCE_SCALES
        
        # Input: 4 features per scale
        in_channels = 4 * len(self.scales)
        
        # Small MLP to produce neighbor embedding
        # Use GroupNorm for 2D spatial data (more appropriate than LayerNorm)
        num_groups = min(8, self.d_neighbor)  # Ensure divisibility
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self.d_neighbor, kernel_size=1),
            nn.GroupNorm(num_groups, self.d_neighbor),
            nn.GELU(),
            nn.Conv2d(self.d_neighbor, self.d_neighbor, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W) integer grid OR (B, C, H, W) logits
        Returns:
            (B, d_neighbor, H, W) neighbor features
        """
        # Convert to soft probabilities
        if x.dim() == 3:
            # Hard labels -> one-hot -> probabilities
            P = F.one_hot(x, num_classes=self.num_colors).permute(0, 3, 1, 2).float()
        else:
            # Logits -> probabilities
            P = torch.softmax(x, dim=1)
        
        # Compute neighbor consensus features
        nce_feats = neighbor_features(P, self.num_colors, self.scales)
        
        # Project to embedding space
        return self.head(nce_feats)


class GridEncoder(nn.Module):
    """
    Encodes grids into fixed-size feature vectors.
    Can use tokenizer-based encoding, color embedding with neighbor-consensus, or both.
    Optionally uses canvas infrastructure for multi-item tiled layouts.
    
    Input: (B, H, W) integer color grids
    Output: (B, d_feat) feature vectors
    """
    
    def __init__(self, num_colors=None, d_color=None, d_feat=None, use_nce=None, use_canvas=None, use_tokenizer=None):
        super().__init__()
        num_colors = num_colors or config.NUM_COLORS
        d_color = d_color or config.D_COLOR
        d_feat = d_feat or config.D_FEAT
        self.use_nce = use_nce if use_nce is not None else config.USE_NEIGHBOR_ENCODER
        self.use_canvas = use_canvas if use_canvas is not None else config.USE_MULTI_ITEM_CANVAS
        self.use_tokenizer = use_tokenizer if use_tokenizer is not None else getattr(config, 'USE_TOKENIZER', False)
        
        # Choose encoding strategy
        if self.use_tokenizer:
            # Use tokenizer-based encoder (can be hybrid with CNN)
            from tokenizer_encoder import HybridGridEncoder
            self.encoder = HybridGridEncoder(
                num_colors=num_colors,
                d_feat=d_feat,
                use_cnn=True,  # Hybrid mode: use both tokenizer and CNN
                use_tokenizer=True
            )
            # Skip the rest of the traditional encoder setup
        else:
            # Traditional encoding path
            # Color embedding
            self.embed = nn.Embedding(num_colors, d_color)
            
            # Optional canvas segment IDs
            if self.use_canvas:
                try:
                    from canvas import SegmentIDEmbedding  # type: ignore
                except ImportError:
                    raise ImportError(
                        "canvas module not found. Please create canvas.py with SegmentIDEmbedding class "
                        "or set USE_MULTI_ITEM_CANVAS=False in config."
                    )
                d_seg = config.CANVAS_SEGMENT_DIM
                self.seg_emb = SegmentIDEmbedding(n_codes=512, d_seg=d_seg)
                # Add segment dimension to color channels
                d_color_in = d_color + d_seg
                self.color_fusion = nn.Conv2d(d_color_in, d_color, kernel_size=1)
            else:
                self.seg_emb = None
            
            # Optional neighbor-consensus encoder
            if self.use_nce:
                self.nce = NeighborConsensusEncoder(num_colors=num_colors)
                d_nce = config.NCE_DIM
                # Fusion layer to combine color + neighbor features
                self.nce_fusion = nn.Conv2d(d_color + d_nce, d_color, kernel_size=1)
            
            self.cnn = nn.Sequential(
                nn.Conv2d(d_color, 128, 3, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.GroupNorm(16, 256),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1)
            )
            
            self.proj = nn.Sequential(
                nn.Linear(256, d_feat),
                nn.GELU(),
                nn.Linear(d_feat, d_feat)
            )
    
    def forward(self, x, seg_ids=None, x_coords=None, y_coords=None, return_neighbor_feats=False, 
                return_color_feats=False):
        """
        Args:
            x: (B, H, W) integer grid
            seg_ids: (B,) optional segment IDs for canvas mode
            x_coords, y_coords: (B, H, W) optional global coordinates for canvas mode
            return_neighbor_feats: if True, return (features, neighbor_feats) tuple
            return_color_feats: if True, return (features, z_color_only, z_neighbor) tuple
        
        Returns:
            (B, d_feat) feature vector
            OR tuple ((B, d_feat), (B, d_nce, H, W)) if return_neighbor_feats=True
            OR tuple ((B, d_feat), (B, d_color, H, W), (B, d_nce, H, W)) if return_color_feats=True
        """
        # Use tokenizer-based encoder if enabled
        if self.use_tokenizer:
            feat = self.encoder(x)
            # For compatibility, return just features (tokenizer doesn't have neighbor/color separate features)
            if return_color_feats or return_neighbor_feats:
                # Return dummy values for compatibility
                return feat, None, None if return_color_feats else (feat, None)
            return feat
        
        # Traditional encoding path
        # Color embedding
        z_color = self.embed(x).permute(0, 3, 1, 2)  # (B, d_color, H, W)
        
        # Optionally add segment ID embeddings
        if self.use_canvas and seg_ids is not None:
            B, _, H, W = z_color.shape
            z_seg = self.seg_emb(seg_ids, H, W)  # (B, d_seg, H, W)
            z_color = torch.cat([z_color, z_seg], dim=1)
            z_color = self.color_fusion(z_color)
        
        # Store color-only features before fusion (for neighbor solver)
        z_color_only = z_color
        
        # Optionally add neighbor features
        z_neighbor = None
        if self.use_nce:
            z_neighbor = self.nce(x)  # (B, d_nce, H, W)
            
            # Stochastic color dropout: force model to use neighbor features
            if self.training and config.USE_NEIGHBOR_SOLVER:
                if torch.rand(1).item() < config.NEIGHBOR_COLOR_DROPOUT:
                    z_color_dropout = torch.zeros_like(z_color)
                    z = torch.cat([z_color_dropout, z_neighbor], dim=1)
                else:
                    z = torch.cat([z_color, z_neighbor], dim=1)
            else:
                z = torch.cat([z_color, z_neighbor], dim=1)  # (B, d_color + d_nce, H, W)
            z = self.nce_fusion(z)  # (B, d_color, H, W)
        else:
            z = z_color
        
        # Note: x_coords, y_coords would be used in attention layers with 2D RoPE
        # For now, stored as metadata; will be used when we add attention-based encoders
        
        h = self.cnn(z).flatten(1)              # (B, 256)
        feat = self.proj(h)                     # (B, d_feat)
        
        if return_color_feats and z_neighbor is not None:
            return feat, z_color_only, z_neighbor
        elif return_neighbor_feats and z_neighbor is not None:
            return feat, z_neighbor
        return feat


class RuleHeads(nn.Module):
    """
    Rule encoders: pair rule and input-only rule heads.
    """
    
    def __init__(self, d_feat=None, d_rule=None):
        super().__init__()
        d_feat = d_feat or config.D_FEAT
        d_rule = d_rule or config.D_RULE
        
        self.pair = nn.Sequential(
            nn.Linear(3 * d_feat, d_rule),
            nn.GELU(),
            nn.Linear(d_rule, d_rule)
        )
        
        self.inp = nn.Sequential(
            nn.Linear(d_feat, d_rule),
            nn.GELU(),
            nn.Linear(d_rule, d_rule)
        )
    
    def pair_rule(self, fx, fy):
        """
        Compute rule from input-output pair.
        Args:
            fx: (B, d_feat) input features
            fy: (B, d_feat) output features
        Returns:
            (B, d_rule) normalized rule vector
        """
        r = self.pair(torch.cat([fx, fy, fy - fx], dim=-1))
        return l2norm(r)
    
    def inp_rule(self, fx):
        """
        Compute rule from input only.
        Args:
            fx: (B, d_feat) input features
        Returns:
            (B, d_rule) normalized rule vector
        """
        r = self.inp(fx)
        return l2norm(r)


class Decoder(nn.Module):
    """
    Conditional decoder that generates output grids from input and rule.
    Uses FiLM (Feature-wise Linear Modulation) for conditioning.
    """
    
    def __init__(self, d_rule=None, num_colors=None):
        super().__init__()
        d_rule = d_rule or config.D_RULE
        num_colors = num_colors or config.NUM_COLORS
        dec_ch = config.DEC_CHANNELS
        
        # FiLM projection
        self.film = nn.Linear(d_rule, dec_ch * 2)  # gamma, beta
        
        # Decoder body
        self.body = nn.Sequential(
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
            nn.GELU(),
        )
        
        # Output head
        self.head = nn.Conv2d(dec_ch, num_colors, 1)
        
        # Input projection
        self.xproj = nn.Conv2d(64, dec_ch, 1)
        self.emb = nn.Embedding(num_colors, 64)
    
    def forward(self, x, r, return_features=False):
        """
        Args:
            x: (B, H, W) input grid
            r: (B, d_rule) rule vector
            return_features: if True, return (logits, features) tuple
        Returns:
            (B, H, W, C) logits
            OR ((B, H, W, C) logits, (B, D, H, W) features) if return_features=True
        """
        z = self.emb(x).permute(0, 3, 1, 2)  # (B, 64, H, W)
        h = self.xproj(z)                    # (B, dec_ch, H, W)
        
        # FiLM conditioning
        gamma, beta = self.film(r).chunk(2, dim=-1)  # (B, dec_ch) each
        h = gamma.view(-1, config.DEC_CHANNELS, 1, 1) * h + beta.view(-1, config.DEC_CHANNELS, 1, 1)
        
        h = self.body(h)  # (B, dec_ch, H, W) - features before final head
        logits = self.head(h).permute(0, 2, 3, 1)  # (B, H, W, C)
        
        if return_features:
            return logits, h
        return logits


class Energy(nn.Module):
    """
    Conditional energy function that evaluates (x, y, r) triplets.
    Lower energy = better alignment.
    
    Uses cross-attention to make energy truly conditional on x:
    x features attend to y features before computing energy.
    """
    
    def __init__(self, num_colors=None, d_color=None, d_feat=None, d_rule=None):
        super().__init__()
        num_colors = num_colors or config.NUM_COLORS
        d_color = d_color or config.D_COLOR
        d_feat = d_feat or config.D_FEAT
        d_rule = d_rule or config.D_RULE
        
        self.num_colors = num_colors
        self.cembed = nn.Embedding(num_colors, d_color)
        
        # Canonical energy: frozen sign + EMA location-scale
        # energy_mu: EMA of E
        # energy_m2: EMA of E^2 (for computing variance over time)
        self.register_buffer('energy_sign', torch.tensor(1.0))
        self.register_buffer('energy_mu', torch.zeros(1))
        self.register_buffer('energy_m2', torch.ones(1))  # Second moment
        
        # Candidate encoder (y) - SPECTRAL NORM for stability
        self.cnn = nn.Sequential(
            spectral_norm(nn.Conv2d(d_color, 128, 3, padding=1)),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 3, padding=1)),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # CRITICAL FIX: Input encoder (x_test) - mirrors candidate encoder - SPECTRAL NORM
        self.input_cnn = nn.Sequential(
            spectral_norm(nn.Conv2d(d_color, 128, 3, padding=1)),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 3, padding=1)),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Enable/disable conditioning (for backward compatibility)
        self.use_cross_attn = getattr(config, 'ENERGY_USE_CROSS_ATTN', False)
        self.use_film = getattr(config, 'ENERGY_USE_FILM', False)
        
        # Cross-attention: x attends to y (only create if enabled)
        # This makes energy TRULY conditional by letting x gate/modulate y features
        d_hidden = 256
        if self.use_cross_attn:
            # All projections now use 256 (CNN output size) - SPECTRAL NORM
            self.x_proj = spectral_norm(nn.Linear(256, d_hidden))
            self.y_proj = spectral_norm(nn.Linear(256, d_hidden))
            self.r_proj = spectral_norm(nn.Linear(d_rule, d_hidden))
            
            # Multi-head attention (2 heads)
            # CRITICAL: NO DROPOUT - energy must be stationary for S2!
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_hidden,
                num_heads=2,
                dropout=0.0,
                batch_first=True
            )
            
            # Joint takes [h_x, h_r, attended_y] = 3*d_hidden - SPECTRAL NORM
            # CRITICAL: NO DROPOUT - energy must be stationary for S2!
            self.joint = nn.Sequential(
                spectral_norm(nn.Linear(d_hidden * 3, 512)),
                nn.GELU(),
                spectral_norm(nn.Linear(512, 256)),
                nn.GELU(),
                spectral_norm(nn.Linear(256, 1))
            )
        else:
            # Old architecture: simple concatenation [x, r, y] = 256+d_rule+256 - SPECTRAL NORM
            # CRITICAL: NO DROPOUT - energy must be stationary for S2!
            self.joint = nn.Sequential(
                spectral_norm(nn.Linear(256 + d_rule + 256, 512)),
                nn.GELU(),
                spectral_norm(nn.Linear(512, 256)),
                nn.GELU(),
                spectral_norm(nn.Linear(256, 1))
            )
        
        # FiLM modulation as alternative/complement (only create if enabled)
        if self.use_film:
            self.film_gamma = nn.Linear(256, 256)  # Changed from d_feat to 256
            self.film_beta = nn.Linear(256, 256)
        
        # Joint network for shared embeddings (d_feat=512 + d_rule=256 + d_feat=512) - SPECTRAL NORM
        # CRITICAL: NO DROPOUT - energy must be stationary for S2!
        self.joint_shared = nn.Sequential(
            spectral_norm(nn.Linear(d_feat + d_rule + d_feat, 512)),
            nn.GELU(),
            spectral_norm(nn.Linear(512, 256)),
            nn.GELU(),
            spectral_norm(nn.Linear(256, 1))
        )
        
        # Local energy branch for spatial signal (no gradient attenuation) - SPECTRAL NORM
        self.local_energy = nn.Sequential(
            spectral_norm(nn.Conv2d(d_color, 128, 3, padding=1)),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            spectral_norm(nn.Conv2d(128, 1, 1))  # energy map
        )
    
    def encode_input(self, x_test):
        """
        CRITICAL FIX: Encode raw test input grid to features.
        
        Args:
            x_test: (B, H, W) integer grid
        Returns:
            (B, 256) features
        """
        if x_test.dim() == 2:
            x_test = x_test.unsqueeze(0)
        
        # Convert to one-hot and embed
        one_hot = F.one_hot(x_test.long(), num_classes=self.num_colors).float()
        x_embed = torch.einsum('bhwc,cd->bhwd', one_hot, self.cembed.weight)
        x_conv = x_embed.permute(0, 3, 1, 2)  # (B, d_color, H, W)
        
        # Encode through CNN
        h_x = self.input_cnn(x_conv).flatten(1)  # (B, 256)
        
        # Scale to counter gradient attenuation
        area = x_conv.shape[2] * x_conv.shape[3]
        h_x = h_x * math.sqrt(area)
        
        return h_x
    
    def energy_from_embeds(self, h_r, h_x, h_y):
        """
        NEW: Compute energy from pre-computed shared embeddings.
        This allows InfoNCE to train the shared encoder pathway.

        Args:
            h_r: (B, d_rule=256) rule embedding from model.encode_rule_shared
            h_x: (B, d_feat=512) input embedding from model.encode_input_shared
            h_y: (B, d_feat=512) candidate embedding from model.encode_candidate_shared
        Returns:
            (B,) energy scores
        """
        # NOTE: Shared encoder outputs d_feat=512 dims; rule is d_rule=256
        # Concatenate: [h_x, h_r, h_y] = [512, 256, 512] = 1280
        features = torch.cat([h_x, h_r, h_y], dim=-1)  # (B, 1280)

        # Use joint_shared network (now initialized in __init__)
        E = self.joint_shared(features).squeeze(-1)
        return E

    def energy_with_shared_input(self, h_x_shared, rbar, y_like, temperature=1.0, canonical=False):
        """
        Compute energy using shared encoder for x_test, but energy's own encoding for y.
        This provides a middle ground: x uses shared encoder, y uses differentiable encoding.

        Args:
            h_x_shared: (B, d_feat) pre-computed input features from shared encoder
            rbar: (B, d_rule) aggregated rule
            y_like: (B, H, W, C) logits or (B, H, W) discrete labels
            temperature: temperature for softmax
            canonical: if True, return canonicalized energy
        Returns:
            (B,) energy scores
        """
        # Convert discrete grids to one-hot logits (same as forward method)
        if y_like.dim() == 2:  # (H, W) discrete
            y_like = y_like.long().unsqueeze(0)  # (1, H, W)

        if y_like.dim() == 3 and y_like.dtype in (torch.long, torch.int64):
            # (B, H, W) discrete -> one-hot sharp logits
            B, H, W = y_like.shape
            C = self.num_colors
            y_oh = F.one_hot(y_like, num_classes=C).float()  # (B, H, W, C)
            BIG = 12.0
            y_logits = y_oh * BIG + (1.0 - y_oh) * (-BIG)  # Sharp class-conditional logits
        elif y_like.dim() == 4:
            y_logits = y_like  # Already (B, H, W, C) logits
        else:
            raise ValueError(f"y_like must be (B,H,W,C) logits or (B,H,W)/(H,W) labels, got {y_like.shape}")

        # Use shared encoder features for x, energy's own encoding for y
        logit_scale = 1.0
        if hasattr(config, 'ENERGY_LOGIT_SCALE') and config.ENERGY_LOGIT_SCALE != 1.0:
            logit_scale = config.ENERGY_LOGIT_SCALE
        if getattr(config, 'DEBUG_ENABLE_SCALING', False):
            logit_scale *= getattr(config, 'DEBUG_LOGIT_SCALE', 1.0)

        p = (y_logits * logit_scale / temperature).softmax(-1)
        emb = torch.einsum('bhwc,cd->bhwd', p, self.cembed.weight)  # (B, H, W, d_color)

        # Encode candidate using energy's own CNN
        y_conv = emb.permute(0, 3, 1, 2)  # (B, d_color, H, W)
        h_y = self.cnn(y_conv).flatten(1)  # (B, 256)
        h_y = h_y * math.sqrt(y_conv.shape[2] * y_conv.shape[3])

        if self.use_film:
            gamma = self.film_gamma(h_x_shared)  # Use shared features for FiLM
            beta = self.film_beta(h_x_shared)
            h_y = h_y * (1 + gamma) + beta

        if self.use_cross_attn:
            h_x_proj = self.x_proj(h_x_shared).unsqueeze(1)      # (B, 1, d_hidden)
            h_y_proj = self.y_proj(h_y).unsqueeze(1)       # (B, 1, d_hidden)
            h_r_proj = self.r_proj(rbar).unsqueeze(1)    # (B, 1, d_hidden)

            query = h_x_proj + h_r_proj
            key_value = h_y_proj

            attended, _ = self.cross_attn(query, key_value, key_value)
            features = torch.cat([h_x_shared, rbar, attended.squeeze(1)], dim=-1)
        else:
            # Simple concatenation: [h_x_shared, rbar, h_y]
            # Note: h_x_shared is 512, rbar is 256, h_y is 256 -> 1024 total
            # But joint network expects 768 (256+256+256), so we need to project h_x_shared down
            h_x_proj = h_x_shared[..., :256]  # Take first 256 dims, or could use a projection layer
            features = torch.cat([h_x_proj, rbar, h_y], dim=-1)  # (B, 256+256+256)

        E = self.joint(features).squeeze(-1)

        if canonical:
            E = self.canonicalize_energy(E)

        return E
    
    def forward(self, x_test, rbar, y_like, temperature=1.0, canonical=False, update_stats=None):
        """
        Evaluate energy E(x, y, r).
        
        S1-EBT: Use RAW energy (canonical=False) for training/inference.
        Canonical energy is ONLY for logging/monitoring.
        
        Args:
            x_test: (B, H, W) test input GRID
            rbar: (B, d_rule) aggregated rule
            y_like: EITHER (B, H, W, C) logits OR (B, H, W)/(H, W) discrete labels
            temperature: temperature for softmax (default 1.0)
            canonical: if True, return canonicalized energy (for logging only!)
                      if False (default), return raw energy (for S1 training)
            update_stats: if None, defaults to (canonical and self.training)
        Returns:
            (B,) energy scores
        """
        # Encode test input to features
        h_x = self.encode_input(x_test)  # (B, 256)
        
        # CRITICAL: Convert discrete grids to one-hot logits
        if y_like.dim() == 2:  # (H, W) discrete
            y_like = y_like.long().unsqueeze(0)  # (1, H, W)
        
        if y_like.dim() == 3 and y_like.dtype in (torch.long, torch.int64):
            # (B, H, W) discrete -> one-hot sharp logits
            B, H, W = y_like.shape
            C = self.num_colors
            y_oh = F.one_hot(y_like, num_classes=C).float()  # (B, H, W, C)
            BIG = 12.0
            y_logits = y_oh * BIG + (1.0 - y_oh) * (-BIG)  # Sharp class-conditional logits
        elif y_like.dim() == 4:
            y_logits = y_like  # Already (B, H, W, C) logits
        else:
            raise ValueError(f"y_like must be (B,H,W,C) logits or (B,H,W)/(H,W) labels, got {y_like.shape}")
        
        # Convert logits to embeddings via softmax weighting
        logit_scale = 1.0
        if hasattr(config, 'ENERGY_LOGIT_SCALE') and config.ENERGY_LOGIT_SCALE != 1.0:
            logit_scale = config.ENERGY_LOGIT_SCALE
        if getattr(config, 'DEBUG_ENABLE_SCALING', False):
            logit_scale *= getattr(config, 'DEBUG_LOGIT_SCALE', 1.0)
        
        p = (y_logits * logit_scale / temperature).softmax(-1)
        emb = torch.einsum('bhwc,cd->bhwd', p, self.cembed.weight)  # (B, H, W, d_color)
        
        # Encode candidate
        y_conv = emb.permute(0, 3, 1, 2)  # (B, d_color, H, W)
        h_y = self.cnn(y_conv).flatten(1)  # (B, 256)
        
        # Fix gradient attenuation from average pooling by scaling by sqrt(area)
        area = y_conv.shape[2] * y_conv.shape[3]  # H * W
        h_y = h_y * math.sqrt(area)  # Safer scaling than full area
        
        # Make energy truly conditional on x
        if self.use_film:
            # FiLM: x modulates y features
            gamma = self.film_gamma(h_x)  # (B, 256)
            beta = self.film_beta(h_x)    # (B, 256)
            h_y = h_y * (1 + gamma) + beta
        
        if self.use_cross_attn:
            # Cross-attention: (x+rule) attends to y
            # Project to common dimension
            h_x_proj = self.x_proj(h_x).unsqueeze(1)      # (B, 1, d_hidden)
            h_y_proj = self.y_proj(h_y).unsqueeze(1)       # (B, 1, d_hidden)
            h_r_proj = self.r_proj(rbar).unsqueeze(1)    # (B, 1, d_hidden)
            
            # Query: x + rule, Key/Value: y
            query = h_x_proj + h_r_proj  # (B, 1, d_hidden)
            key_value = h_y_proj    # (B, 1, d_hidden)
            
            # Cross-attend
            attended, _ = self.cross_attn(
                query, key_value, key_value
            )  # (B, 1, d_hidden)
            
            # Concatenate [h_x, h_r, attended_y] for final energy
            features = torch.cat([
                h_x_proj.squeeze(1),
                h_r_proj.squeeze(1),
                attended.squeeze(1)
            ], dim=-1)  # (B, 3*d_hidden)
        else:
            # Fallback: simple concat [x, r, y]
            features = torch.cat([h_x, rbar, h_y], dim=-1)
        
        # Global energy
        E_global = self.joint(features).squeeze(-1)  # (B,)
        
        # Local energy branch (direct pixelwise path, no gradient attenuation)
        E_local = self.local_energy(y_conv).mean(dim=(1, 2, 3))  # (B,) scalar per sample
        
        # Combine global and local energy
        raw_energy = E_global + 0.05 * E_local  # Smaller weight for stability
        
        if getattr(config, 'DEBUG_ENABLE_SCALING', False):
            raw_energy = raw_energy * getattr(config, 'DEBUG_ENERGY_SCALE', 1.0)
        
        # DEFAULT: Return RAW energy (for S1 training/inference)
        if not canonical:
            return raw_energy
        
        # OPTIONAL: Canonical energy (for logging/monitoring only)
        if update_stats is None:
            update_stats = self.training
        
        return self.canonicalize_energy(raw_energy, update_stats=update_stats)
    
    def canonicalize_energy(self, raw_E, update_stats=False):
        """
        Batch-size-1-safe canonicalization (READ-ONLY for logging/S2).
        
        Tracks E[E] and E[E^2] over TIME (not per-batch std):
        - energy_mu stores EMA of E
        - energy_m2 stores EMA of E^2
        - std = sqrt(E[E^2] - (E[E])^2)
        
        Works even when batch_size=1!
        """
        # 1) Apply frozen orientation
        E = self.energy_sign * raw_E  # [B]
        
        # 2) Update EMA stats only when requested
        if update_stats:
            with torch.no_grad():
                m = E.mean()          # E[E]
                m2 = (E * E).mean()   # E[E^2]
                
                # EMA of mean and second moment
                self.energy_mu.mul_(0.99).add_(0.01 * m)
                self.energy_m2.mul_(0.99).add_(0.01 * m2)
        
        # 3) Compute variance from moments: Var(E) = E[E^2] - (E[E])^2
        mu = self.energy_mu
        m2 = self.energy_m2
        var = (m2 - mu * mu).clamp_min(1e-6)
        std = var.sqrt().clamp(1e-3, 1e3)  # Bounded for safety
        
        # 4) Normalize
        E_canonical = (E - mu) / std
        return E_canonical


class EBTSystem(nn.Module):
    """
    Complete EBT system combining all components.
    """
    
    def __init__(self):
        super().__init__()
        self.encoder = GridEncoder()
        self.rule_heads = RuleHeads()
        decoder_type = getattr(config, 'DECODER_TYPE', 'conv')
        if decoder_type == 'transformer':
            from decoders_transformer import TransformerDecoder  # type: ignore
            self.decoder = TransformerDecoder()
        elif decoder_type == 'token':
            from token_decoder import TokenDecoder  # type: ignore
            self.decoder = TokenDecoder()
        else:
            self.decoder = Decoder()
        self.energy = Energy()
        
        # Neighbor affinity head (uses neighbor features only)
        if config.USE_NEIGHBOR_ENCODER and config.USE_NEIGHBOR_SOLVER:
            from neighbor_solver import NeighborAffinityHead  # type: ignore
            self.neighbor_affinity = NeighborAffinityHead(
                in_ch=config.NCE_DIM,
                dirs=config.NEIGHBOR_DIRS,
                hidden=32
            )
            # Simple color-only decoder head for neighbor solver
            # Takes color features and produces logits directly
            self.color_only_head = nn.Sequential(
                nn.Conv2d(config.D_COLOR, 128, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, config.NUM_COLORS, 1)
            )
        else:
            self.neighbor_affinity = None
            self.color_only_head = None
        
        # Spatial supervision heads (anchor decoder to canvas locations)
        if config.USE_SPATIAL_SUPERVISION and config.USE_MULTI_ITEM_CANVAS:
            from spatial_supervision import SpatialSupervisionHeads  # type: ignore
            # Note: Requires decoder to expose features before final logits
            # For conv decoder, d_hidden = DEC_CHANNELS
            # For transformer decoder, d_hidden = TRANSFORMER_D_MODEL
            if getattr(config, 'DECODER_TYPE', 'conv') == 'transformer':
                d_hidden = config.TRANSFORMER_D_MODEL
            else:
                d_hidden = config.DEC_CHANNELS
            self.spatial_heads = SpatialSupervisionHeads(
                d_hidden=d_hidden,
                fourier_freqs=config.SPATIAL_FOURIER_FREQS
            )
        else:
            self.spatial_heads = None
    
    def aggregate_rules(self, r1, r2, r_star):
        """
        Aggregate rule vectors.
        Args:
            r1: (B, d_rule) rule from pair 1
            r2: (B, d_rule) rule from pair 2
            r_star: (B, d_rule) rule from test input
        Returns:
            (B, d_rule) aggregated rule
        """
        rbar = config.ALPHA_RULE * r1 + config.ALPHA_RULE * r2 + config.BETA_RULE * r_star
        return l2norm(rbar)
    
    def encode_rule_shared(self, x1, y1, x2, y2, x_test):
        """
        Shared rule encoding for energy verifier.
        Returns: (B, d_rule) aggregated rule embedding
        """
        fx1 = self.encoder(x1)
        fy1 = self.encoder(y1)
        fx2 = self.encoder(x2)
        fy2 = self.encoder(y2)
        fx_star = self.encoder(x_test)
        
        r1 = self.rule_heads.pair_rule(fx1, fy1)
        r2 = self.rule_heads.pair_rule(fx2, fy2)
        r_star = self.rule_heads.inp_rule(fx_star)
        
        return self.aggregate_rules(r1, r2, r_star)
    
    def encode_input_shared(self, x_test):
        """
        Shared test input encoding for energy verifier.
        Returns: (B, d_feat) test input features
        """
        return self.encoder(x_test)
    
    def encode_candidate_shared(self, y_input, tau=1.0):
        """
        Shared candidate encoding for energy verifier.
        Can handle both logits (B,H,W,C) and discrete grids (B,H,W).
        Returns: (B, d_feat) candidate features
        """
        # Check if input is logits or discrete
        if y_input.dim() == 4:  # (B, H, W, C) logits
            y_int = y_input.argmax(dim=-1)  # (B, H, W)
        elif y_input.dim() == 3:  # (B, H, W) discrete
            y_int = y_input
        else:
            raise ValueError(f"encode_candidate_shared expects (B,H,W,C) logits or (B,H,W) discrete, got {y_input.shape}")

        return self.encoder(y_int)
    
    def forward(self, x1, y1, x2, y2, x_star):
        """
        Forward pass through the system.
        Args:
            x1, y1: (B, H, W) first training pair
            x2, y2: (B, H, W) second training pair
            x_star: (B, H, W) test input
        Returns:
            r1, r2, r_star, rbar, y_logits
        """
        # Encode all grids
        fx1 = self.encoder(x1)
        fy1 = self.encoder(y1)
        fx2 = self.encoder(x2)
        fy2 = self.encoder(y2)
        fx_star = self.encoder(x_star)
        
        # Compute rules
        r1 = self.rule_heads.pair_rule(fx1, fy1)
        r2 = self.rule_heads.pair_rule(fx2, fy2)
        r_star = self.rule_heads.inp_rule(fx_star)
        
        # Aggregate
        rbar = self.aggregate_rules(r1, r2, r_star)
        
        # Decode
        y_logits = self.decoder(x_star, rbar)
        
        return r1, r2, r_star, rbar, y_logits
