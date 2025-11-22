"""
Token-based decoder that predicts structured tokens and renders them into grids.

The decoder generates a fixed-length sequence of tokens, each with:
- presence probability
- type, color, and layer logits
- spatial parameters (center x/y, width, height)
- thickness/orientation controls for structured shapes

Tokens are rasterized with a differentiable renderer so gradients can flow
from pixel-level losses and the energy model back into the token parameters.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config

TOKEN_TYPE_VOCAB: Tuple[str, ...] = tuple(config.TOKEN_DECODER_TYPES)
TOKEN_TYPE_TO_ID = {name: idx for idx, name in enumerate(TOKEN_TYPE_VOCAB)}


class TokenDecoder(nn.Module):
    """
    Transformer-style decoder that emits token parameters and renders them back to logits.
    """

    def __init__(self, d_rule: Optional[int] = None, num_colors: Optional[int] = None):
        super().__init__()
        self.num_colors = num_colors or config.NUM_COLORS
        self.d_rule = d_rule or config.D_RULE
        self.max_tokens = config.TOKEN_DECODER_MAX_TOKENS
        self.hidden = config.TOKEN_DECODER_HIDDEN
        self.num_layers = config.TOKEN_DECODER_LAYERS
        self.type_vocab = TOKEN_TYPE_VOCAB
        self.edge_sharpness = config.TOKEN_RENDER_EDGE_SHARPNESS
        self.min_size = config.TOKEN_RENDER_MIN_SIZE
        self.layer_temp = config.TOKEN_RENDER_LAYER_TEMP
        self.presence_thresh = config.TOKEN_PRESENCE_THRESHOLD

        # Input embedding stack
        self.color_emb = nn.Embedding(self.num_colors, 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, self.hidden, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Rule conditioning
        self.rule_proj = nn.Linear(self.d_rule, self.hidden)
        self.context_proj = nn.Linear(self.hidden * 2, self.hidden)

        # Token queries
        self.token_queries = nn.Parameter(
            torch.randn(self.max_tokens, self.hidden) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=4,
            dim_feedforward=self.hidden * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Output heads
        self.presence_head = nn.Linear(self.hidden, 1)
        self.type_head = nn.Linear(self.hidden, len(self.type_vocab))
        self.color_head = nn.Linear(self.hidden, self.num_colors)
        self.layer_head = nn.Linear(self.hidden, 1)
        self.bbox_head = nn.Linear(self.hidden, 4)          # cx, cy, w, h (normalized)
        self.thickness_head = nn.Linear(self.hidden, 1)
        self.orientation_head = nn.Linear(self.hidden, 1)    # 0=horiz, 1=vert

        self.coord_cache: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_token_struct: Optional[Dict[str, torch.Tensor]] = None

    def _get_coords(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (H, W, device)
        if key not in self.coord_cache:
            yy, xx = torch.meshgrid(
                torch.linspace(0.0, float(H - 1), H, device=device),
                torch.linspace(0.0, float(W - 1), W, device=device),
                indexing='ij'
            )
            self.coord_cache[key] = (yy, xx)
        return self.coord_cache[key]

    def forward(self, x: torch.Tensor, r: torch.Tensor, return_tokens: bool = False):
        """
        Args:
            x: (B, H, W) integer grid
            r: (B, d_rule) rule embedding
            return_tokens: if True, also return token dictionary (for debugging)
        """
        B, H, W = x.shape

        z = self.color_emb(x).permute(0, 3, 1, 2)  # (B, 64, H, W)
        feat = self.encoder(z)                     # (B, hidden, H, W)
        pooled = self.pool(feat).flatten(1)        # (B, hidden)

        rule_feat = self.rule_proj(r)
        context = torch.tanh(self.context_proj(torch.cat([pooled, rule_feat], dim=-1)))

        queries = self.token_queries.unsqueeze(0).expand(B, -1, -1)
        token_inp = queries + context.unsqueeze(1)
        token_feat = self.transformer(token_inp)

        presence_logits = self.presence_head(token_feat)
        type_logits = self.type_head(token_feat)
        color_logits = self.color_head(token_feat)
        layer_logits = self.layer_head(token_feat)
        bbox_raw = self.bbox_head(token_feat)
        thickness_raw = self.thickness_head(token_feat)
        orientation_logits = self.orientation_head(token_feat)

        token_state = self._render_tokens(
            H=H,
            W=W,
            presence_logits=presence_logits,
            type_logits=type_logits,
            color_logits=color_logits,
            layer_logits=layer_logits,
            bbox_raw=bbox_raw,
            thickness_raw=thickness_raw,
            orientation_logits=orientation_logits,
        )

        self.last_token_struct = token_state
        self.last_token_struct['H'] = H
        self.last_token_struct['W'] = W

        if return_tokens:
            return token_state['logits'], token_state
        return token_state['logits']

    def _render_tokens(
        self,
        H: int,
        W: int,
        presence_logits: torch.Tensor,
        type_logits: torch.Tensor,
        color_logits: torch.Tensor,
        layer_logits: torch.Tensor,
        bbox_raw: torch.Tensor,
        thickness_raw: torch.Tensor,
        orientation_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert token parameters into logits via a differentiable renderer.
        """
        device = type_logits.device
        B, T, _ = type_logits.shape

        presence = torch.sigmoid(presence_logits)                # (B, T, 1)
        type_probs = F.softmax(type_logits, dim=-1)              # (B, T, K)
        color_probs = F.softmax(color_logits, dim=-1)            # (B, T, C)
        layer_scores = layer_logits.squeeze(-1)                  # (B, T)

        bbox_norm = torch.sigmoid(bbox_raw)                      # (B, T, 4)
        H_span = float(max(H - 1, 1))
        W_span = float(max(W - 1, 1))
        H_full = float(max(H, 1))
        W_full = float(max(W, 1))
        cx = bbox_norm[..., 0] * W_span                          # (B, T)
        cy = bbox_norm[..., 1] * H_span
        size_min = self.min_size
        width_frac = size_min + (1.0 - size_min) * bbox_norm[..., 2]
        height_frac = size_min + (1.0 - size_min) * bbox_norm[..., 3]
        width = width_frac * W_full
        height = height_frac * H_full

        thickness_frac = torch.sigmoid(thickness_raw).squeeze(-1)  # (B, T)
        orientation = torch.sigmoid(orientation_logits).squeeze(-1)

        yy, xx = self._get_coords(H, W, device)
        yy = yy.view(1, 1, H, W)
        xx = xx.view(1, 1, H, W)

        half_w = width / 2.0
        half_h = height / 2.0
        left = (xx - (cx - half_w).unsqueeze(-1).unsqueeze(-1))
        right = ((cx + half_w).unsqueeze(-1).unsqueeze(-1) - xx)
        top = (yy - (cy - half_h).unsqueeze(-1).unsqueeze(-1))
        bottom = ((cy + half_h).unsqueeze(-1).unsqueeze(-1) - yy)

        inside_x = torch.sigmoid(self.edge_sharpness * left) * torch.sigmoid(self.edge_sharpness * right)
        inside_y = torch.sigmoid(self.edge_sharpness * top) * torch.sigmoid(self.edge_sharpness * bottom)
        rect_mask = inside_x * inside_y  # (B, T, H, W)

        # Hollow rectangles / borders use inner subtract outer
        base_thickness = torch.min(width, height) * (0.45 * thickness_frac)
        inner_half_w = torch.clamp(half_w - base_thickness, min=1.0)
        inner_half_h = torch.clamp(half_h - base_thickness, min=1.0)
        inner_left = (xx - (cx - inner_half_w).unsqueeze(-1).unsqueeze(-1))
        inner_right = ((cx + inner_half_w).unsqueeze(-1).unsqueeze(-1) - xx)
        inner_top = (yy - (cy - inner_half_h).unsqueeze(-1).unsqueeze(-1))
        inner_bottom = ((cy + inner_half_h).unsqueeze(-1).unsqueeze(-1) - yy)
        inner_mask = (
            torch.sigmoid(self.edge_sharpness * inner_left)
            * torch.sigmoid(self.edge_sharpness * inner_right)
            * torch.sigmoid(self.edge_sharpness * inner_top)
            * torch.sigmoid(self.edge_sharpness * inner_bottom)
        )
        hollow_mask = torch.clamp(rect_mask - inner_mask, min=0.0)
        border_inner_half_w = torch.clamp(half_w - 0.5 * base_thickness, min=1.0)
        border_inner_half_h = torch.clamp(half_h - 0.5 * base_thickness, min=1.0)
        border_inner_left = (xx - (cx - border_inner_half_w).unsqueeze(-1).unsqueeze(-1))
        border_inner_right = ((cx + border_inner_half_w).unsqueeze(-1).unsqueeze(-1) - xx)
        border_inner_top = (yy - (cy - border_inner_half_h).unsqueeze(-1).unsqueeze(-1))
        border_inner_bottom = ((cy + border_inner_half_h).unsqueeze(-1).unsqueeze(-1) - yy)
        border_inner_mask = (
            torch.sigmoid(self.edge_sharpness * border_inner_left)
            * torch.sigmoid(self.edge_sharpness * border_inner_right)
            * torch.sigmoid(self.edge_sharpness * border_inner_top)
            * torch.sigmoid(self.edge_sharpness * border_inner_bottom)
        )
        border_mask = torch.clamp(rect_mask - border_inner_mask, min=0.0)

        # Lines: blend horizontal / vertical strips
        line_half_h = torch.clamp(half_h * torch.clamp(thickness_frac, min=0.05), min=1.0)
        line_half_w = torch.clamp(half_w * torch.clamp(thickness_frac, min=0.05), min=1.0)
        horiz_top = (yy - (cy - line_half_h).unsqueeze(-1).unsqueeze(-1))
        horiz_bottom = ((cy + line_half_h).unsqueeze(-1).unsqueeze(-1) - yy)
        horiz_mask = (
            torch.sigmoid(self.edge_sharpness * horiz_top)
            * torch.sigmoid(self.edge_sharpness * horiz_bottom)
            * inside_x
        )
        vert_left = (xx - (cx - line_half_w).unsqueeze(-1).unsqueeze(-1))
        vert_right = ((cx + line_half_w).unsqueeze(-1).unsqueeze(-1) - xx)
        vert_mask = (
            torch.sigmoid(self.edge_sharpness * vert_left)
            * torch.sigmoid(self.edge_sharpness * vert_right)
            * inside_y
        )
        line_mask = (1.0 - orientation).unsqueeze(-1).unsqueeze(-1) * horiz_mask + \
            orientation.unsqueeze(-1).unsqueeze(-1) * vert_mask

        mask_bank = {
            'RECT': rect_mask,
            'HOLLOW_RECT': hollow_mask,
            'LINE': line_mask,
            'BORDER': border_mask,
        }
        stack_masks = torch.stack(
            [mask_bank.get(name, rect_mask) for name in self.type_vocab],
            dim=-1
        )  # (B, T, H, W, num_types)

        type_probs_view = type_probs.view(B, T, 1, 1, -1)
        mask = (stack_masks * type_probs_view).sum(dim=-1)  # (B, T, H, W)
        mask = mask * presence.view(B, T, 1, 1)  # Broadcast presence correctly

        color_probs_view = color_probs.view(B, T, 1, 1, self.num_colors)
        token_grids = mask.unsqueeze(-1) * color_probs_view  # (B, T, H, W, C)

        layer_weights = torch.softmax(layer_scores / max(self.layer_temp, 1e-3), dim=1)
        layer_weights = layer_weights * presence.squeeze(-1)
        layer_weights = layer_weights / (layer_weights.sum(dim=1, keepdim=True) + 1e-6)
        layer_weights_expanded = layer_weights.view(B, T, 1, 1, 1)

        grid = (token_grids * layer_weights_expanded).sum(dim=1)  # (B, H, W, C)
        grid = torch.clamp(grid, 1e-4, 1.0)
        grid = grid / (grid.sum(dim=-1, keepdim=True) + 1e-6)
        logits = torch.logit(grid.clamp(1e-4, 1.0 - 1e-4))

        # Compute per-token visibility: how much each token contributes to final output
        # Visibility = (token_contribution / token_area) averaged across spatial dims
        token_contribution = (token_grids * layer_weights_expanded).sum(dim=-1)  # (B, T, H, W) - color-summed
        token_area = mask  # (B, T, H, W) - spatial mask for each token
        
        # Visibility ratio: what fraction of the token's area is actually visible in output
        visibility = (token_contribution * token_area).sum(dim=(2, 3))  # (B, T)
        token_size = token_area.sum(dim=(2, 3)) + 1e-6  # (B, T)
        visibility_ratio = visibility / token_size  # (B, T) in [0, 1]

        bbox_struct = torch.stack([bbox_norm[..., 0], bbox_norm[..., 1], width_frac, height_frac], dim=-1)

        return {
            'logits': logits,
            'prob_grid': grid,
            'presence_logits': presence_logits,
            'type_logits': type_logits,
            'color_logits': color_logits,
            'layer_logits': layer_logits,
            'bbox_struct': bbox_struct,
            'orientation_logits': orientation_logits,
            'thickness': thickness_frac.unsqueeze(-1),
            'token_grids': token_grids,  # (B, T, H, W, C) - per-token rendered grids
            'layer_weights': layer_weights,  # (B, T) - layer blending weights
            'visibility_ratio': visibility_ratio,  # (B, T) - how visible each token is
        }

    def export_token_lists(self, threshold: Optional[float] = None) -> Optional[List[List[Dict]]]:
        """
        Export the most recent token predictions as Python dicts for diagnostics / matching.
        """
        if self.last_token_struct is None:
            return None

        thresh = threshold if threshold is not None else self.presence_thresh
        H = self.last_token_struct.get('H')
        W = self.last_token_struct.get('W')

        presence = torch.sigmoid(self.last_token_struct['presence_logits']).detach().cpu()  # (B, T, 1)
        type_probs = F.softmax(self.last_token_struct['type_logits'], dim=-1).detach().cpu()
        color_probs = F.softmax(self.last_token_struct['color_logits'], dim=-1).detach().cpu()
        layer_values = torch.sigmoid(self.last_token_struct['layer_logits']).detach().cpu()
        bbox = self.last_token_struct['bbox_struct'].detach().cpu()
        orientation = torch.sigmoid(self.last_token_struct['orientation_logits']).detach().cpu()
        thickness = self.last_token_struct['thickness'].detach().cpu()

        tokens_per_batch: List[List[Dict]] = []
        B = presence.shape[0]
        for b in range(B):
            tokens: List[Dict] = []

            for t in range(self.max_tokens):
                if presence[b, t, 0].item() < thresh:
                    continue
                type_id = int(type_probs[b, t].argmax().item())
                color_id = int(color_probs[b, t].argmax().item())
                layer = float(layer_values[b, t, 0].item())
                cx = float(bbox[b, t, 0].item() * max(W - 1, 1))
                cy = float(bbox[b, t, 1].item() * max(H - 1, 1))
                w = float(bbox[b, t, 2].item() * max(W, 1))
                h = float(bbox[b, t, 3].item() * max(H, 1))
                ori = float(orientation[b, t, 0].item())
                thick = float(thickness[b, t, 0].item())

                token = {
                    'type': self.type_vocab[type_id],
                    'color': color_id,
                    'layer': layer,
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h,
                    'orientation': 'V' if ori > 0.5 else 'H',
                    'presence': presence[b, t, 0].item(),
                    'thickness': thick,
                    'decoder_index': t,
                }
                # convert to top-left for compatibility
                token['x'] = max(0.0, cx - w / 2.0)
                token['y'] = max(0.0, cy - h / 2.0)
                tokens.append(token)
            tokens_per_batch.append(tokens)
        return tokens_per_batch
