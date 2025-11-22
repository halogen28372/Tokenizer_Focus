"""
Hybrid Program-Renderer (HyPER) Loss
=====================================
Trains energy models with token-level program distance + optional shape-aware rendering.

Key components:
1. InfoNCE energy ranking with adaptive margin from program distance
2. Multi-scale render consistency (Dice + boundary + one-sided edge Chamfer)
3. Token set matching via greedy assignment with IoU/type/color/params
4. Group-balanced reweighting for rare shape types
5. Energy gradient regularizer for smooth landscapes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from arc_lago_tokenizer import ArcLagoTokenizer, TokenizerConfig
import numpy as np


# ============================================================================
# Program Distance (Token Set Matching)
# ============================================================================

class ProgramDistanceConfig:
    def __init__(self,
                 w_type=1.0, w_color=0.5, w_params=1.0, w_iou=1.0, w_layer=0.25,
                 w_missed=3.0, norm="mean"):
        self.w_type = w_type
        self.w_color = w_color
        self.w_params = w_params
        self.w_iou = w_iou
        self.w_layer = w_layer
        self.w_missed = w_missed
        self.norm = norm  # "mean" or "sum"


# Global tokenizer instance
_tokenizer = ArcLagoTokenizer(TokenizerConfig(
    max_layers=4, emit_symmetry_meta=True, emit_relations=True,
    assume_rect_occlusion_prior=True, merge_diagonal_lines=True
))


def _bbox_from_token(t):
    """Extract bounding box from token parameters."""
    ty = t.get("type")
    if ty in ("RECT", "HOLLOW_RECT", "BORDER", "CHECKER", "C_SHAPE"):
        x = int(t.get("x", t.get("cx", 0)))
        y = int(t.get("y", t.get("cy", 0)))
        w = int(t.get("w", t.get("spacing", 1)))
        h = int(t.get("h", t.get("num_layers", 1)))
        return (y, x, y + h - 1, x + w - 1)
    if ty == "CONCENTRIC_RECTS":
        cx = int(t.get("cx", 0))
        cy = int(t.get("cy", 0))
        spacing = int(t.get("spacing", 1))
        L = int(t.get("num_layers", 1))
        half = spacing * L
        return (cy - half, cx - half, cy + half, cx + half)
    if ty == "LINE":
        ori = t.get("orientation", "H")
        x, y = int(t.get("x", 0)), int(t.get("y", 0))
        w, h = int(t.get("w", 1)), int(t.get("h", 1))
        return (y, x, y, x + w - 1) if ori == "H" else (y, x, y + h - 1, x)
    if ty in ("DIAG_LINE", "DIAG_CROSS_X", "X_CROSS"):
        x, y = int(t.get("x", 0)), int(t.get("y", 0))
        length = int(t.get("length", max(t.get("w", 1), t.get("h", 1))))
        return (y, x, y + length - 1, x + length - 1)
    # Fallback for other types
    x, y = int(t.get("x", t.get("cx", 0))), int(t.get("y", t.get("cy", 0)))
    w, h = int(t.get("w", 1)), int(t.get("h", 1))
    return (y, x, y + h - 1, x + w - 1)


def _iou(bb1, bb2):
    """Compute IoU between two bboxes."""
    if bb1 is None or bb2 is None:
        return 0.0
    y1, x1, y2, x2 = bb1
    Y1, X1, Y2, X2 = bb2
    iy1, ix1 = max(y1, Y1), max(x1, X1)
    iy2, ix2 = min(y2, Y2), min(x2, X2)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (y2 - y1 + 1) * (x2 - x1 + 1)
    a2 = (Y2 - Y1 + 1) * (X2 - X1 + 1)
    return inter / float(a1 + a2 - inter + 1e-6)


def _kind_dist(tp, tg):
    return 0.0 if tp.get("type") == tg.get("type") else 1.0


def _color_dist(tp, tg):
    return 0.0 if int(tp.get("color", -1)) == int(tg.get("color", -1)) else 1.0


def _layer_dist(tp, tg):
    return min(1.0, abs(int(tp.get("layer", 0)) - int(tg.get("layer", 0))) / 3.0)


def _param_dist(tp, tg, H, W):
    """Normalized parameter distance."""
    keys = (set(tp.keys()) & set(tg.keys())) - {"type", "color", "layer", "id"}
    if not keys:
        return 0.0
    
    s, n = 0.0, 0
    for k in keys:
        vp, vg = tp[k], tg[k]
        # String parameters
        if isinstance(vp, str) or isinstance(vg, str):
            s += 0.0 if vp == vg else 1.0
            n += 1
            continue
        # Numeric parameters - normalize by appropriate scale
        if k in ("x", "cx", "w"):
            scale = max(1, W)
        elif k in ("y", "cy", "h"):
            scale = max(1, H)
        elif k in ("length", "radius", "spacing", "thickness", "border_thickness",
                   "num_layers", "arm_length", "cell_size", "spoke_length",
                   "amplitude", "wavelength", "num_cycles", "coil_count", "order"):
            scale = max(H, W)
        elif "angle" in k:
            scale = 180.0
        else:
            scale = max(H, W)
        s += abs(float(vp) - float(vg)) / float(scale)
        n += 1
    
    return s / max(1, n)


def tokens_distance(tokens_pred, tokens_gt, H, W, cfg, kind_weights=None):
    """
    Greedy token set matching with costs.
    
    Returns:
        distance: scalar program distance
        info: dict with match statistics
    """
    Np, Ng = len(tokens_pred), len(tokens_gt)
    if Np == 0 and Ng == 0:
        return 0.0, {'matches': 0, 'Np': 0, 'Ng': 0}
    
    used = [False] * Ng
    total_cost, matches = 0.0, 0
    
    # Sort predicted tokens by bbox area (largest first for greedy matching)
    def _area(bb):
        if bb is None:
            return 0
        return (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
    
    bbs_pred = [_bbox_from_token(t) for t in tokens_pred]
    order = list(range(Np))
    try:
        order.sort(key=lambda i: _area(bbs_pred[i]), reverse=True)
    except:
        pass
    
    # Greedy matching
    for pi in order:
        tp = tokens_pred[pi]
        best_j, best_cost = -1, 1e9
        
        for j, tg in enumerate(tokens_gt):
            if used[j]:
                continue
            
            # Compute matching cost
            wt = kind_weights.get(tg.get("type", ""), 1.0) if kind_weights else 1.0
            
            cost = cfg.w_type * _kind_dist(tp, tg)
            cost += cfg.w_color * _color_dist(tp, tg)
            cost += cfg.w_params * _param_dist(tp, tg, H, W)
            
            bbp = _bbox_from_token(tp)
            bbg = _bbox_from_token(tg)
            if bbp is not None and bbg is not None:
                cost += cfg.w_iou * (1.0 - _iou(bbp, bbg))
            
            cost += cfg.w_layer * _layer_dist(tp, tg)
            cost *= wt  # Group-balanced weighting
            
            if cost < best_cost:
                best_cost, best_j = cost, j
        
        # Accept match if cost < miss penalty
        if best_j != -1 and best_cost < cfg.w_missed:
            total_cost += best_cost
            used[best_j] = True
            matches += 1
        else:
            total_cost += cfg.w_missed
    
    # Penalize unmatched GT tokens
    total_cost += cfg.w_missed * (Ng - matches)
    
    # Normalize
    if cfg.norm == "mean":
        distance = total_cost / max(1, Np + Ng)
    else:
        distance = total_cost
    
    return distance, {'matches': matches, 'Np': Np, 'Ng': Ng}


def tokenize_grid_tensor(grid_hw):
    """Tokenize a (H, W) integer grid."""
    grid_list = grid_hw.detach().cpu().tolist()
    result = _tokenizer.tokenize(grid_list)
    return result["shapes"]


def program_distance_batch(y_logits, y_star, cfg, kind_weights=None):
    """
    Compute program distance for a batch.
    
    Args:
        y_logits: (B, H, W, C) predicted logits
        y_star: (B, H, W) target grids
        cfg: ProgramDistanceConfig
        kind_weights: dict mapping shape type to weight
    
    Returns:
        distances: (B,) tensor of program distances
        infos: list of info dicts
    """
    B, H, W, C = y_logits.shape
    pred_grids = y_logits.argmax(dim=-1)  # (B, H, W)
    
    distances, infos = [], []
    for b in range(B):
        tokens_pred = tokenize_grid_tensor(pred_grids[b])
        tokens_gt = tokenize_grid_tensor(y_star[b])
        dist, info = tokens_distance(tokens_pred, tokens_gt, H, W, cfg, kind_weights)
        distances.append(dist)
        infos.append(info)
    
    return torch.tensor(distances, device=y_logits.device, dtype=torch.float32), infos


# ============================================================================
# Group-Balanced Token Kind Weighting
# ============================================================================

class KindCounterEMA:
    """Track token kind frequencies and compute inverse-frequency weights."""
    
    def __init__(self, decay=0.99):
        self.counts = {}  # kind -> EMA count
        self.decay = decay
    
    def update(self, shapes_list):
        """Update counts from a list of shape tokens."""
        batch_counts = {}
        for t in shapes_list:
            k = t.get("type", "UNK")
            batch_counts[k] = batch_counts.get(k, 0) + 1
        
        for k, v in batch_counts.items():
            old = self.counts.get(k, 0.0)
            self.counts[k] = self.decay * old + (1.0 - self.decay) * float(v)
    
    def get_weights(self):
        """Get inverse-frequency weights normalized to mean=1."""
        if not self.counts:
            return {}
        
        vals = list(self.counts.values())
        min_val, max_val = min(vals), max(vals)
        
        weights = {}
        for k, v in self.counts.items():
            # Normalized frequency (0=rare, 1=common)
            if max_val == min_val:
                freq_norm = 0.5
            else:
                freq_norm = (v - min_val) / (max_val - min_val)
            # Inverse weight (rare shapes get higher weight)
            weights[k] = 1.0 - freq_norm
        
        # Normalize to mean 1
        mean_w = sum(weights.values()) / max(1, len(weights))
        return {k: (w / (mean_w + 1e-8)) for k, w in weights.items()}


# ============================================================================
# Render Consistency (Shape-Aware, No Per-Cell CE)
# ============================================================================

def render_consistency_loss(y_logits, y_star, scales=(1, 2)):
    """
    Multi-scale shape consistency without per-cell CE.
    
    Components:
    - Multi-scale soft Dice (coarse shape agreement)
    - Boundary alignment (Sobel magnitude)
    - One-sided edge Chamfer (predicted edges to GT edges)
    
    Args:
        y_logits: (B, H, W, C) predicted logits
        y_star: (B, H, W) target grid
        scales: pooling scales for Dice
    
    Returns:
        scalar loss
    """
    B, H, W, C = y_logits.shape
    device = y_logits.device
    
    # Soft probabilities from predictions
    P = F.softmax(y_logits.permute(0, 3, 1, 2), dim=1)  # (B, C, H, W)
    
    # One-hot from GT
    G = F.one_hot(y_star, num_classes=C).permute(0, 3, 1, 2).float()  # (B, C, H, W)
    
    # (1) Multi-scale Dice
    dice_loss = 0.0
    valid_scales = 0
    for s in scales:
        if s > min(H, W):  # Skip if scale too large
            continue
        
        if s > 1:
            P_s = F.avg_pool2d(P, kernel_size=s, stride=s)
            G_s = F.avg_pool2d(G, kernel_size=s, stride=s)
        else:
            P_s, G_s = P, G
        
        intersection = (P_s * G_s).sum(dim=(1, 2, 3))
        denominator = (P_s.pow(2) + G_s.pow(2)).sum(dim=(1, 2, 3)) + 1e-6
        dice_loss += (1.0 - 2.0 * intersection / denominator).mean()
        valid_scales += 1
    
    if valid_scales > 0:
        dice_loss /= valid_scales
    else:
        dice_loss = torch.tensor(0.0, device=device)
    
    # (2) Boundary alignment (Sobel)
    if H >= 3 and W >= 3:
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                               device=device, dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               device=device, dtype=torch.float32).view(1, 1, 3, 3)
        
        # Predicted edges (differentiable)
        P_sum = P.sum(1, keepdim=True)  # (B, 1, H, W)
        Px = F.conv2d(P_sum, sobel_x, padding=1)
        Py = F.conv2d(P_sum, sobel_y, padding=1)
        P_edge = torch.sqrt(Px**2 + Py**2 + 1e-6)
        
        # GT edges (constant)
        G_sum = G.sum(1, keepdim=True)
        Gx = F.conv2d(G_sum, sobel_x, padding=1)
        Gy = F.conv2d(G_sum, sobel_y, padding=1)
        G_edge = torch.sqrt(Gx**2 + Gy**2 + 1e-6)
        
        boundary_loss = F.l1_loss(P_edge, G_edge)
    else:
        boundary_loss = torch.tensor(0.0, device=device)
    
    # (3) One-sided edge Chamfer: predicted edges → nearest GT edge
    # Compute distance transform of GT edges (approximation using max-pooling)
    if H >= 3 and W >= 3:
        # GT edge mask (threshold)
        G_edge_mask = (G_edge > 0.1).float()  # (B, 1, H, W)
        
        # Simple distance approximation: use negative max-pooling
        # For each predicted edge, penalize if far from any GT edge
        # Simplified: just use L1 between edge maps
        chamfer_loss = (P_edge * (1.0 - G_edge_mask)).mean()
    else:
        chamfer_loss = torch.tensor(0.0, device=device)
    
    # Combine components
    total = dice_loss + 0.25 * boundary_loss + 0.15 * chamfer_loss
    
    return total


# ============================================================================
# Energy Ranking with InfoNCE
# ============================================================================

def energy_rank_infonce(E_pos, E_neg_list, margins, temperature=1.0):
    """
    InfoNCE-style ranking loss on energies.
    
    We want E_pos to be LOWER than all E_neg values.
    
    Args:
        E_pos: (B,) positive energies E(x, y*)
        E_neg_list: list of (B,) tensors, negatives E(x, y_k)
        margins: (B,) adaptive margins based on program distance
        temperature: temperature for softmax
    
    Returns:
        scalar loss
    """
    # Stack all energies: [E_pos, E_neg_0, E_neg_1, ...]
    # We want to maximize exp(-E_pos) / [exp(-E_pos) + sum exp(-E_neg - margin)]
    
    # Numerator: exp(-E_pos / tau)
    logits_pos = -E_pos / temperature  # (B,)
    
    # Denominator: exp(-E_pos / tau) + sum exp(-(E_neg + margin) / tau)
    # For numerical stability, use logsumexp
    all_logits = [logits_pos]
    for E_neg in E_neg_list:
        # Add margin to make negatives "easier" (higher energy harder to beat)
        all_logits.append(-(E_neg + margins) / temperature)
    
    all_logits = torch.stack(all_logits, dim=1)  # (B, 1 + num_negatives)
    
    # InfoNCE: -log(exp(pos) / sum(exp(all)))
    loss = -F.log_softmax(all_logits, dim=1)[:, 0].mean()
    
    return loss


# ============================================================================
# Energy Gradient Regularizer
# ============================================================================

def gradient_norm_energy(model, x_test, rbar, y_logits):
    """
    Regularize ||∇_y E||^2 for smooth energy landscapes.
    
    Args:
        model: EBTSystem with energy function
        x_test: (B, H, W) test input
        rbar: (B, d_rule) aggregated rule
        y_logits: (B, H, W, C) candidate logits
    
    Returns:
        scalar loss
    """
    # Enable gradient tracking on y_logits
    y_logits_copy = y_logits.detach().requires_grad_(True)
    
    # Compute energy
    E = model.energy(x_test, rbar, y_logits_copy, canonical=False).mean()
    
    # Compute gradient
    grad_y, = torch.autograd.grad(E, y_logits_copy, create_graph=True)
    
    # L2 norm
    grad_norm_sq = (grad_y ** 2).mean()
    
    return grad_norm_sq


# ============================================================================
# HyPER: Hybrid Program-Renderer Loss
# ============================================================================

def hyper_program_energy_loss(model, x_test, rbar, y_logits, y_star,
                               pd_cfg, kind_ema=None,
                               w_rank=1.0, w_render=0.5, w_reg=0.05,
                               base_margin=0.1, margin_scale=0.6,
                               temperature=1.0, negatives=None):
    """
    Hybrid Program-Renderer (HyPER) loss for token-aware energy training.
    
    Args:
        model: EBTSystem with energy function
        x_test: (B, H, W) test input grid
        rbar: (B, d_rule) aggregated rule
        y_logits: (B, H, W, C) predicted output logits (after K optimization steps)
        y_star: (B, H, W) ground truth output grid
        pd_cfg: ProgramDistanceConfig for token matching
        kind_ema: KindCounterEMA for group-balanced weighting (optional)
        w_rank: weight for ranking loss
        w_render: weight for render consistency (set 0 for no grid supervision)
        w_reg: weight for energy gradient regularizer
        base_margin: base margin for ranking
        margin_scale: scaling factor for adaptive margin from program distance
        temperature: temperature for InfoNCE
        negatives: list of negative samples (optional, for replay buffer)
    
    Returns:
        loss: scalar loss
        metrics: dict of diagnostic metrics
    """
    B, H, W, C = y_logits.shape
    device = y_logits.device
    
    # (1) Compute program distance and update EMA
    with torch.no_grad():
        # Collect all GT tokens for kind counter
        all_gt_shapes = []
        for b in range(B):
            all_gt_shapes += tokenize_grid_tensor(y_star[b])
        
        if kind_ema is not None:
            kind_ema.update(all_gt_shapes)
        
        kind_weights = kind_ema.get_weights() if kind_ema else None
    
    # Compute program distances
    d_prog, prog_infos = program_distance_batch(y_logits, y_star, pd_cfg, kind_weights)
    
    # (2) Energy ranking with adaptive margin
    E_pos = model.energy(x_test, rbar, y_star, canonical=False)  # (B,)
    E_neg = model.energy(x_test, rbar, y_logits, canonical=False)  # (B,)
    
    # Adaptive margin from program distance
    margins = base_margin + margin_scale * d_prog  # (B,)
    
    # Collect negatives
    neg_list = [E_neg]
    if negatives is not None:
        for y_neg in negatives:
            E_neg_k = model.energy(x_test, rbar, y_neg, canonical=False)
            neg_list.append(E_neg_k)
    
    # InfoNCE ranking loss
    rank_loss = energy_rank_infonce(E_pos, neg_list, margins, temperature)
    
    # (3) Render consistency (optional)
    if w_render > 0.0:
        render_loss = render_consistency_loss(y_logits, y_star, scales=(1, 2))
    else:
        render_loss = torch.tensor(0.0, device=device)
    
    # (4) Energy gradient regularizer
    if w_reg > 0.0:
        reg_loss = gradient_norm_energy(model, x_test, rbar, y_logits)
    else:
        reg_loss = torch.tensor(0.0, device=device)
    
    # Total loss
    loss = w_rank * rank_loss + w_render * render_loss + w_reg * reg_loss
    
    # Metrics
    with torch.no_grad():
        pred = y_logits.argmax(dim=-1)
        pixel_acc = (pred == y_star).float().mean()
        energy_gap = (E_neg - E_pos).mean()  # Should be positive (E_neg > E_pos)
    
    metrics = {
        'loss_rank': rank_loss.item(),
        'loss_render': render_loss.item() if w_render > 0 else 0.0,
        'loss_reg': reg_loss.item() if w_reg > 0 else 0.0,
        'prog_dist': d_prog.mean().item(),
        'energy_pos': E_pos.mean().item(),
        'energy_neg': E_neg.mean().item(),
        'energy_gap': energy_gap.item(),
        'margin_mean': margins.mean().item(),
        'pixel_acc': pixel_acc.item(),
        'token_matches': np.mean([info['matches'] for info in prog_infos]),
        'tokens_pred': np.mean([info['Np'] for info in prog_infos]),
        'tokens_gt': np.mean([info['Ng'] for info in prog_infos]),
    }
    
    return loss, metrics

