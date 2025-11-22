"""
Permutation-Invariant Loss for ARC
===================================
Set-based matching with Hungarian algorithm + color invariance.
Now includes differentiable token rendering for proper token-level supervision.
Includes occlusion penalty to prevent wasted tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from arc_lago_tokenizer import ArcLagoTokenizer, TokenizerConfig
from token_renderer import render_tokens_to_grid, tokens_to_logits
from token_occlusion_penalty import compute_token_occlusion_penalty
from config import config

try:
    from token_decoder import TOKEN_TYPE_VOCAB, TOKEN_TYPE_TO_ID
except ImportError:
    TOKEN_TYPE_VOCAB = ()
    TOKEN_TYPE_TO_ID = {}


# Global tokenizer
_tokenizer = ArcLagoTokenizer(TokenizerConfig(
    max_layers=4, emit_symmetry_meta=True, emit_relations=True,
    assume_rect_occlusion_prior=True, merge_diagonal_lines=True
))


def tokenize_grid(grid_hw):
    """Tokenize a (H, W) integer grid."""
    grid_list = grid_hw.detach().cpu().tolist()
    result = _tokenizer.tokenize(grid_list)
    return result["shapes"]


def bbox_from_token(t):
    """Extract bounding box from token."""
    ty = t.get("type")
    if ty in ("RECT", "HOLLOW_RECT", "BORDER", "CHECKER", "C_SHAPE"):
        x = int(t.get("x", t.get("cx", 0)))
        y = int(t.get("y", t.get("cy", 0)))
        w = int(t.get("w", 1))
        h = int(t.get("h", 1))
        return (y, x, y + h - 1, x + w - 1)
    if ty == "LINE":
        ori = t.get("orientation", "H")
        x, y = int(t.get("x", 0)), int(t.get("y", 0))
        w, h = int(t.get("w", 1)), int(t.get("h", 1))
        return (y, x, y, x + w - 1) if ori == "H" else (y, x, y + h - 1, x)
    # Default fallback
    x = int(t.get("x", t.get("cx", 0)))
    y = int(t.get("y", t.get("cy", 0)))
    w = int(t.get("w", 1))
    h = int(t.get("h", 1))
    return (y, x, y + h - 1, x + w - 1)


def iou_bbox(bb1, bb2):
    """Compute IoU between two bboxes."""
    if bb1 is None or bb2 is None:
        return 0.0
    y1, x1, y2, x2 = bb1
    Y1, X1, Y2, X2 = bb2
    iy1, ix1 = max(y1, Y1), max(x1, X1)
    iy2, ix2 = min(y2, Y2), min(x2, X2)
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    if inter <= 0:
        return 0.0
    a1 = (y2 - y1 + 1) * (x2 - x1 + 1)
    a2 = (Y2 - Y1 + 1) * (X2 - X1 + 1)
    return inter / float(a1 + a2 - inter + 1e-6)


def token_cost(tp, tg, H, W, color_perm=None):
    """
    Compute matching cost between two tokens.
    
    Args:
        tp: predicted token dict
        tg: ground truth token dict
        H, W: grid dimensions for normalization
        color_perm: optional color permutation dict {pred_color: gt_color}
    
    Returns:
        cost: scalar cost (0 = perfect match)
    """
    cost = 0.0
    
    # Type cost
    if tp.get("type") != tg.get("type"):
        cost += 2.0  # High penalty for type mismatch
    
    # Color cost (with optional permutation)
    pred_color = int(tp.get("color", -1))
    gt_color = int(tg.get("color", -1))
    if color_perm is not None and pred_color in color_perm:
        pred_color = color_perm[pred_color]
    if pred_color != gt_color:
        cost += 1.0
    
    # Layer cost
    pred_layer = int(tp.get("layer", 0))
    gt_layer = int(tg.get("layer", 0))
    cost += 0.25 * abs(pred_layer - gt_layer)
    
    # Spatial parameters (normalized by grid size)
    spatial_keys = ["x", "y", "w", "h", "cx", "cy"]
    for key in spatial_keys:
        if key in tp and key in tg:
            scale = W if key in ["x", "cx", "w"] else H
            cost += abs(float(tp[key]) - float(tg[key])) / max(1, scale)
    
    # Other parameters
    param_keys = ["thickness", "border_thickness", "scale", "length", 
                  "orientation", "num_layers", "spacing"]
    for key in param_keys:
        if key in tp and key in tg:
            vp, vg = tp[key], tg[key]
            if isinstance(vp, str):
                cost += 0.0 if vp == vg else 0.25
            else:
                cost += abs(float(vp) - float(vg)) / max(H, W)
    
    # Bounding box IoU cost
    bbp = bbox_from_token(tp)
    bbg = bbox_from_token(tg)
    if bbp is not None and bbg is not None:
        iou = iou_bbox(bbp, bbg)
        cost += 1.0 * (1.0 - iou)
    
    return cost


CANONICAL_TYPE_MAP = {
    'RECTANGLE': 'RECT',
    'SQUARE': 'RECT',
    'REGION': 'RECT',
    'LINE': 'LINE',
    'DIAG_LINE': 'LINE',
    'DIAG_CROSS_X': 'LINE',
    'BORDER': 'BORDER',
    'FRAME': 'BORDER',
    'HOLLOW_RECT': 'HOLLOW_RECT',
    'CHECKER': 'HOLLOW_RECT',  # Pattern → hollow
    'CONCENTRIC_RECTS': 'HOLLOW_RECT',
    'CROSSHATCH': 'LINE',  # Grid pattern → lines
    'RADIAL': 'LINE',  # Radial spokes → lines
    'ZIGZAG': 'LINE',  # Zigzag → line
    'SPIRAL': 'HOLLOW_RECT',  # Spiral → hollow
    'TETROMINO': 'RECT',  # Small shapes → rect
    'PENTOMINO': 'RECT',
    'SPARSE_DOTS': 'RECT',  # Dots → small rects
    'C_SHAPE': 'HOLLOW_RECT',  # C-shape → hollow
}


def canonical_decoder_type(name: str) -> str:
    """
    Map ARC tokenizer types to the decoder's limited vocabulary.
    """
    if not name:
        return 'RECT'
    key = name.upper()
    return CANONICAL_TYPE_MAP.get(key, 'RECT')


def build_gt_token_targets(tokens_gt, H: int, W: int):
    """
    Convert GT tokens into normalized targets for supervising the token decoder.
    """
    targets = []
    W_span = max(W, 1)
    H_span = max(H, 1)
    rect_id = TOKEN_TYPE_TO_ID.get('RECT', 0)
    for token in tokens_gt:
        bb = bbox_from_token(token)
        if bb is None:
            continue
        y0, x0, y1, x1 = bb
        w = max(1.0, float(x1 - x0 + 1))
        h = max(1.0, float(y1 - y0 + 1))
        cx = (x0 + x1 + 1.0) / 2.0
        cy = (y0 + y1 + 1.0) / 2.0
        cx_norm = min(1.0, max(0.0, cx / W_span))
        cy_norm = min(1.0, max(0.0, cy / H_span))
        w_norm = min(1.0, w / W_span)
        h_norm = min(1.0, h / H_span)
        canon = canonical_decoder_type(token.get('type', 'RECT'))
        type_id = TOKEN_TYPE_TO_ID.get(canon, rect_id)
        color_id = int(token.get('color', 0)) % config.NUM_COLORS
        layer = float(token.get('layer', 0)) / 3.0
        orientation = 1.0 if h_norm > w_norm else 0.0
        targets.append({
            'type_id': type_id,
            'color_id': color_id,
            'bbox': (cx_norm, cy_norm, w_norm, h_norm),
            'layer': max(0.0, min(1.0, layer)),
            'orientation': orientation,
        })
    return targets


def compute_token_param_loss(struct_b, decoder_tokens, assignment, unmatched_pred, gt_targets, device):
    """
    Compute supervised losses on token parameters using Hungarian matches.
    """
    total_loss = struct_b['type_logits'].new_tensor(0.0, device=device)
    matched = 0
    if assignment:
        decoder_indices = torch.tensor(
            [decoder_tokens[i]['decoder_index'] for i, _ in assignment],
            dtype=torch.long,
            device=device
        )
        gt_indices = [j for _, j in assignment]
        type_targets = torch.tensor(
            [gt_targets[j]['type_id'] for j in gt_indices],
            dtype=torch.long,
            device=device
        )
        color_targets = torch.tensor(
            [gt_targets[j]['color_id'] for j in gt_indices],
            dtype=torch.long,
            device=device
        )
        bbox_targets = torch.tensor(
            [gt_targets[j]['bbox'] for j in gt_indices],
            dtype=torch.float32,
            device=device
        )
        orient_targets = torch.tensor(
            [gt_targets[j]['orientation'] for j in gt_indices],
            dtype=torch.float32,
            device=device
        ).unsqueeze(-1)
        layer_targets = torch.tensor(
            [gt_targets[j]['layer'] for j in gt_indices],
            dtype=torch.float32,
            device=device
        ).unsqueeze(-1)

        total_loss = total_loss + F.cross_entropy(struct_b['type_logits'][decoder_indices], type_targets)
        total_loss = total_loss + F.cross_entropy(struct_b['color_logits'][decoder_indices], color_targets)
        total_loss = total_loss + F.smooth_l1_loss(struct_b['bbox_struct'][decoder_indices], bbox_targets)
        orient_pred = torch.sigmoid(struct_b['orientation_logits'][decoder_indices])
        layer_pred = torch.sigmoid(struct_b['layer_logits'][decoder_indices])
        total_loss = total_loss + F.mse_loss(orient_pred, orient_targets)
        total_loss = total_loss + F.mse_loss(layer_pred, layer_targets)
        ones = torch.ones_like(struct_b['presence_logits'][decoder_indices])
        total_loss = total_loss + F.binary_cross_entropy_with_logits(struct_b['presence_logits'][decoder_indices], ones)
        matched = len(gt_indices)

    if unmatched_pred:
        decoder_indices = torch.tensor(
            [decoder_tokens[i]['decoder_index'] for i in unmatched_pred],
            dtype=torch.long,
            device=device
        )
        zeros = torch.zeros_like(struct_b['presence_logits'][decoder_indices])
        total_loss = total_loss + F.binary_cross_entropy_with_logits(struct_b['presence_logits'][decoder_indices], zeros)

    return total_loss, matched


def hungarian_token_matching(tokens_pred, tokens_gt, H, W, color_perm=None, miss_penalty=3.0):
    """
    Hungarian matching between predicted and GT token sets.
    
    Args:
        tokens_pred: list of predicted token dicts
        tokens_gt: list of GT token dicts
        H, W: grid dimensions
        color_perm: optional color permutation
        miss_penalty: cost for unmatched tokens
    
    Returns:
        total_cost: average matching cost
        assignment: list of (pred_idx, gt_idx) pairs
        unmatched_pred: list of unmatched pred indices
        unmatched_gt: list of unmatched gt indices
    """
    Np, Ng = len(tokens_pred), len(tokens_gt)
    
    if Np == 0 and Ng == 0:
        return 0.0, [], [], []
    
    # Compute cost matrix (Np x Ng)
    cost_matrix = np.zeros((Np, Ng), dtype=np.float32)
    for i in range(Np):
        for j in range(Ng):
            cost_matrix[i, j] = token_cost(tokens_pred[i], tokens_gt[j], H, W, color_perm)
    
    # Solve Hungarian matching on the rectangular matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assignment = []
    total_cost = 0.0
    matched_pred = set()
    matched_gt = set()
    
    for i, j in zip(row_ind, col_ind):
        c = cost_matrix[i, j]
        # Always accept match to force gradient flow
        assignment.append((i, j))
        total_cost += c
        matched_pred.add(i)
        matched_gt.add(j)
            
    # Collect unmatched
    unmatched_pred = [i for i in range(Np) if i not in matched_pred]
    unmatched_gt = [j for j in range(Ng) if j not in matched_gt]
    
    # Add penalty for unmatched
    total_cost += len(unmatched_pred) * miss_penalty
    total_cost += len(unmatched_gt) * miss_penalty
    
    # Average cost
    avg_cost = total_cost / max(1, Np + Ng)
    
    return avg_cost, assignment, unmatched_pred, unmatched_gt


def find_best_color_permutation(pred_grid, gt_grid, num_colors=10):
    """
    Find best color permutation using IoU-based Hungarian matching.
    
    Args:
        pred_grid: (H, W) predicted grid
        gt_grid: (H, W) ground truth grid
        num_colors: number of colors
    
    Returns:
        color_perm: dict {pred_color: gt_color}
    """
    H, W = pred_grid.shape
    
    # Compute IoU matrix between all color pairs
    iou_matrix = np.zeros((num_colors, num_colors), dtype=np.float32)
    
    for c_pred in range(num_colors):
        mask_pred = (pred_grid == c_pred).float()
        area_pred = mask_pred.sum()
        
        for c_gt in range(num_colors):
            mask_gt = (gt_grid == c_gt).float()
            area_gt = mask_gt.sum()
            
            if area_pred == 0 and area_gt == 0:
                continue
            
            intersection = (mask_pred * mask_gt).sum()
            union = area_pred + area_gt - intersection
            
            if union > 0:
                iou_matrix[c_pred, c_gt] = intersection / union
    
    # Hungarian matching (maximize IoU = minimize -IoU)
    cost_matrix = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Build permutation dict
    color_perm = {}
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] > 0.01:  # Only map if there's some overlap
            color_perm[i] = j
    
    return color_perm


def apply_color_permutation(grid, color_perm):
    """Apply color permutation to a grid."""
    result = grid.clone()
    for pred_c, gt_c in color_perm.items():
        result[grid == pred_c] = gt_c
    return result


def permutation_invariant_token_loss(model, x_test, rbar, y_logits, y_star,
                                     w_match=1.0, w_unmatched=3.0,
                                     w_render=0.3, w_energy_rank=1.0,
                                     base_margin=0.2):
    """
    Permutation-invariant loss with Hungarian token matching + color invariance.
    
    Args:
        model: EBTSystem with energy function
        x_test: (B, H, W) test input
        rbar: (B, d_rule) aggregated rule
        y_logits: (B, H, W, C) predicted logits
        y_star: (B, H, W) ground truth grid
        w_match: weight for matched token cost
        w_unmatched: penalty for unmatched tokens
        w_render: weight for render consistency
        w_energy_rank: weight for energy ranking
        base_margin: base margin for energy ranking
    
    Returns:
        loss: scalar loss
        metrics: dict of metrics
    """
    B, H, W, C = y_logits.shape
    device = y_logits.device
    
    # Convert predictions to hard labels
    pred_grid = y_logits.argmax(dim=-1)  # (B, H, W)
    
    total_loss = 0.0
    all_metrics = {
        'match_cost': [],
        'color_perm_iou': [],
        'token_precision': [],
        'token_recall': [],
        'energy_gap': [],
    }
    
    for b in range(B):
        # (1) Find best color permutation
        color_perm = find_best_color_permutation(pred_grid[b], y_star[b], num_colors=C)
        
        # Apply color permutation to predicted grid
        pred_grid_aligned = apply_color_permutation(pred_grid[b], color_perm)
        
        # Compute color alignment quality
        color_iou = (pred_grid_aligned == y_star[b]).float().mean().item()
        all_metrics['color_perm_iou'].append(color_iou)
        
        # (2) Tokenize both grids
        try:
            tokens_pred = tokenize_grid(pred_grid_aligned)
            tokens_gt = tokenize_grid(y_star[b])
        except:
            # If tokenization fails, skip this sample
            continue
        
        # (3) Hungarian token matching
        match_cost, assignment, unmatched_pred, unmatched_gt = hungarian_token_matching(
            tokens_pred, tokens_gt, H, W, color_perm=None, miss_penalty=w_unmatched
        )
        
        # Token precision/recall
        if len(tokens_pred) > 0:
            precision = len(assignment) / len(tokens_pred)
        else:
            precision = 0.0
        
        if len(tokens_gt) > 0:
            recall = len(assignment) / len(tokens_gt)
        else:
            recall = 1.0 if len(tokens_pred) == 0 else 0.0
        
        all_metrics['match_cost'].append(match_cost)
        all_metrics['token_precision'].append(precision)
        all_metrics['token_recall'].append(recall)
        
        # Token matching loss
        sample_loss = w_match * match_cost
        
        # (4) Render consistency (optional, shape-aware)
        if w_render > 0.0:
            # Multi-scale Dice on color-aligned prediction
            P = F.softmax(y_logits[b:b+1].permute(0, 3, 1, 2), dim=1)  # (1, C, H, W)
            
            # Permute prediction probabilities according to color mapping
            P_aligned = torch.zeros_like(P)
            for pred_c, gt_c in color_perm.items():
                P_aligned[:, gt_c] += P[:, pred_c]
            # Add unpermuted colors (those not in mapping)
            for c in range(C):
                if c not in color_perm:
                    P_aligned[:, c] += P[:, c]
            
            G = F.one_hot(y_star[b:b+1], num_classes=C).permute(0, 3, 1, 2).float()
            
            # Dice loss
            intersection = (P_aligned * G).sum()
            denominator = (P_aligned.pow(2) + G.pow(2)).sum() + 1e-6
            dice_loss = 1.0 - 2.0 * intersection / denominator
            
            sample_loss += w_render * dice_loss.item()
        
        total_loss += sample_loss
    
    # (5) Energy ranking (use shared encoder pathway for consistency!)
    if w_energy_rank > 0.0:
        h_x_shared = model.encode_input_shared(x_test)
        E_pos = model.energy.energy_with_shared_input(h_x_shared, rbar, y_star, canonical=False)  # (B,)
        E_neg = model.energy.energy_with_shared_input(h_x_shared, rbar, y_logits, canonical=False)  # (B,)
        
        # Adaptive margin based on token match quality
        avg_match_cost = np.mean(all_metrics['match_cost']) if all_metrics['match_cost'] else 1.0
        margin = base_margin + 0.5 * avg_match_cost
        
        # Ranking: E_pos + margin < E_neg
        rank_loss = F.relu(E_pos + margin - E_neg).mean()
        
        energy_gap = (E_neg - E_pos).mean().item()
        all_metrics['energy_gap'].append(energy_gap)
        
        total_loss += w_energy_rank * rank_loss.item()
    
    # Convert to tensor for backprop
    loss_tensor = torch.tensor(total_loss / max(1, B), device=device, requires_grad=False)
    
    # Aggregate metrics
    metrics = {
        'token_match_cost': np.mean(all_metrics['match_cost']) if all_metrics['match_cost'] else 0.0,
        'color_alignment': np.mean(all_metrics['color_perm_iou']) if all_metrics['color_perm_iou'] else 0.0,
        'token_precision': np.mean(all_metrics['token_precision']) if all_metrics['token_precision'] else 0.0,
        'token_recall': np.mean(all_metrics['token_recall']) if all_metrics['token_recall'] else 0.0,
        'energy_gap': np.mean(all_metrics['energy_gap']) if all_metrics['energy_gap'] else 0.0,
        'token_matches': token_matches_total,
        'tokens_pred': tokens_pred_total,
        'tokens_gt': tokens_gt_total,
    }

    return loss_tensor, metrics


def differentiable_hungarian_loss(model, x_test, rbar, y_logits, y_star,
                                  w_match=1.0, w_unmatched=3.0, w_render=0.3,
                                  w_energy=1.0, base_margin=0.2, temperature=1.0,
                                  w_token=1.0, w_occlusion=0.05):
    """
    Fully differentiable version with token-level supervision.
    
    Key innovation: Tokenize GT grid → render tokens → supervise predicted grid to match rendered tokens.
    This ensures the model learns to generate tokenizable structures.
    """
    B, H, W, C = y_logits.shape
    device = y_logits.device
    
    P = F.softmax(y_logits.permute(0, 3, 1, 2), dim=1)  # (B, C, H, W)
    
    decoder_token_lists = None
    token_struct = None
    if hasattr(model, 'decoder'):
        decoder = model.decoder
        if hasattr(decoder, 'export_token_lists'):
            # EXPOSE ALL TOKENS for DETR-style matching (including low-confidence ones)
            # This allows the matcher to assign GT targets to the best available slots
            # even if they are currently "inactive".
            decoder_token_lists = decoder.export_token_lists(threshold=0.01)
        token_struct = getattr(decoder, 'last_token_struct', None)
    
    # (1) TOKENIZE GT GRID → RENDER TOKENS → SUPERVISE PREDICTIONS
    token_render_loss = y_logits.new_tensor(0.0)
    token_match_metrics = []
    token_param_loss = y_logits.new_tensor(0.0)
    supervised_tokens = 0
    token_matches_total = 0
    tokens_pred_total = 0
    tokens_gt_total = 0
    
    for b in range(B):
        gt_tokens = []
        try:
            gt_tokens = tokenize_grid(y_star[b])
        except Exception:
            gt_tokens = []
        gt_targets = build_gt_token_targets(gt_tokens, H, W)
        
        if gt_tokens:
            # Render GT tokens to grid (DIFFERENTIABLE)
            gt_rendered = render_tokens_to_grid(gt_tokens, H, W, C, device=device)  # (H, W, C)
            gt_rendered = gt_rendered.unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
            
            # Compare predicted grid to rendered GT tokens via multi-scale Dice
            # Use MACRO-averaged Dice to handle class imbalance (rare colors matter!)
            dice_b = y_logits.new_tensor(0.0)
            for scale in [1, 2]:
                if scale > min(H, W):
                    continue
                if scale > 1:
                    P_s = F.avg_pool2d(P[b:b+1], kernel_size=scale, stride=scale)
                    G_s = F.avg_pool2d(gt_rendered, kernel_size=scale, stride=scale)
                else:
                    P_s, G_s = P[b:b+1], gt_rendered
                
                # Per-class Dice (C channels)
                # P_s, G_s: (1, C, H, W)
                intersection = (P_s * G_s).sum(dim=(2, 3))  # (1, C)
                denominator = (P_s.pow(2) + G_s.pow(2)).sum(dim=(2, 3)) + 1e-6  # (1, C)
                dice_c = 1.0 - 2.0 * intersection / denominator  # (1, C)
                
                # Weight classes: Background (0) gets less weight, others get more
                # Simple heuristic: Mean of all classes (Macro average)
                dice_b += dice_c.mean()
            
            token_render_loss += dice_b / 2.0
        
        decoder_tokens_b = None
        if decoder_token_lists is not None and b < len(decoder_token_lists):
            decoder_tokens_b = decoder_token_lists[b]
            # DEBUG: Check why tokens might be empty
            # if len(decoder_tokens_b) == 0:
            #     print(f"DEBUG: Batch {b} decoder produced 0 tokens > threshold")

        match_tokens = decoder_tokens_b
        if match_tokens is None:
            with torch.no_grad():
                pred_hard_b = y_logits[b].argmax(dim=-1).detach().cpu()
                try:
                    match_tokens = tokenize_grid(pred_hard_b)
                except Exception:
                    match_tokens = []
        else:
            match_tokens = match_tokens or []

        match_cost, assignment, unmatched_pred, unmatched_gt = hungarian_token_matching(
            match_tokens, gt_tokens, H, W, miss_penalty=w_unmatched
        )
        token_match_metrics.append(match_cost)
        token_matches_total += len(assignment)
        tokens_pred_total += len(match_tokens)
        tokens_gt_total += len(gt_tokens)
        
        if decoder_tokens_b is not None and token_struct is not None and gt_targets:
            struct_b = {
                'type_logits': token_struct['type_logits'][b],
                'color_logits': token_struct['color_logits'][b],
                'bbox_struct': token_struct['bbox_struct'][b],
                'orientation_logits': token_struct['orientation_logits'][b],
                'layer_logits': token_struct['layer_logits'][b],
                'presence_logits': token_struct['presence_logits'][b],
            }
            sample_loss, matched = compute_token_param_loss(
                struct_b,
                decoder_tokens_b,
                assignment,
                unmatched_pred,
                gt_targets,
                device
            )
            token_param_loss = token_param_loss + sample_loss
            supervised_tokens += matched
    
    if B > 0:
        token_render_loss = token_render_loss / B
    if supervised_tokens > 0:
        token_param_loss = token_param_loss / supervised_tokens
    else:
        token_param_loss = token_param_loss * 0.0
    
    # (2) COLOR PERMUTATION + PIXEL-LEVEL ALIGNMENT (fallback for non-tokenizable regions)
    G = F.one_hot(y_star, num_classes=C).permute(0, 3, 1, 2).float()  # (B, C, H, W)
    
    # Compute IoU matrix for color alignment
    iou_matrix = torch.zeros(B, C, C, device=device)
    for c_pred in range(C):
        for c_gt in range(C):
            intersection = (P[:, c_pred:c_pred+1] * G[:, c_gt:c_gt+1]).sum(dim=(2, 3))
            union = (P[:, c_pred:c_pred+1].pow(2) + G[:, c_gt:c_gt+1].pow(2)).sum(dim=(2, 3))
            iou_matrix[:, c_pred, c_gt] = 2.0 * intersection / (union + 1e-6)
    
    # Soft color permutation
    logits_perm = iou_matrix * 10.0
    perm_soft = F.gumbel_softmax(logits_perm, tau=temperature, hard=False, dim=2)  # (B, C, C)
    P_aligned = torch.einsum('bcp,bchw->bphw', perm_soft, P)  # (B, C, H, W)
    
    # Pixel-level Dice loss (complementary to token render loss)
    # Use MACRO-averaged Dice here too
    pixel_dice_loss = 0.0
    for scale in [1, 2]:
        if scale > min(H, W):
            continue
        if scale > 1:
            P_s = F.avg_pool2d(P_aligned, kernel_size=scale, stride=scale)
            G_s = F.avg_pool2d(G, kernel_size=scale, stride=scale)
        else:
            P_s, G_s = P_aligned, G
        
        # Per-class Dice
        # P_s, G_s: (B, C, H, W)
        intersection = (P_s * G_s).sum(dim=(2, 3))  # (B, C)
        denominator = (P_s.pow(2) + G_s.pow(2)).sum(dim=(2, 3)) + 1e-6
        dice_c = 1.0 - 2.0 * intersection / denominator  # (B, C)
        
        pixel_dice_loss += dice_c.mean(dim=1).mean(dim=0)  # Mean over classes, then batch
    
    pixel_dice_loss /= 2.0
    
    # (3) Energy ranking (use shared encoder pathway for consistency!)
    h_x_shared = model.encode_input_shared(x_test)
    E_pos = model.energy.energy_with_shared_input(h_x_shared, rbar, y_star, canonical=False)
    E_neg = model.energy.energy_with_shared_input(h_x_shared, rbar, y_logits, canonical=False)
    
    alignment_quality = iou_matrix.max(dim=2).values.mean(dim=1)  # (B,)
    margin = base_margin + 0.5 * (1.0 - alignment_quality)
    
    rank_loss = F.relu(E_pos + margin - E_neg).mean()
    
    # (4) Token occlusion penalty - penalize tokens that are 100% hidden
    occlusion_loss = torch.tensor(0.0, device=device)
    occlusion_metrics = {}
    
    if token_struct is not None:
        occlusion_loss, occlusion_metrics = compute_token_occlusion_penalty(
            token_struct, 
            w_occlusion=1.0,  # Raw loss, will be scaled by w_occlusion parameter
            visibility_threshold=0.05,
            presence_threshold=0.3
        )
    
    # Total loss: render + energy + token params + occlusion
    loss = (
        w_render * (token_render_loss + pixel_dice_loss)
        + w_energy * rank_loss
        + w_token * token_param_loss
        + w_occlusion * occlusion_loss
    )
    
    # Metrics
    with torch.no_grad():
        pred_hard = y_logits.argmax(dim=-1)
        pixel_acc = (pred_hard == y_star).float().mean()
        energy_gap = (E_neg - E_pos).mean()
        token_match_cost = np.mean(token_match_metrics) if token_match_metrics else 0.0
    
    metrics = {
        'loss_render': (token_render_loss + pixel_dice_loss).item(),
        'loss_token_render': token_render_loss.item(),
        'loss_pixel_dice': pixel_dice_loss.item(),
        'loss_rank': rank_loss.item(),
        'loss_token_params': token_param_loss.item(),
        'loss_occlusion': occlusion_loss.item() if isinstance(occlusion_loss, torch.Tensor) else 0.0,
        'pixel_acc': pixel_acc.item(),
        'energy_gap': energy_gap.item(),
        'color_alignment': alignment_quality.mean().item(),
        'margin_mean': margin.mean().item(),
        'token_match_cost': token_match_cost,
        'token_matches': float(token_matches_total),
        'tokens_pred': float(tokens_pred_total),
        'tokens_gt': float(tokens_gt_total),
        'num_active_tokens': occlusion_metrics.get('num_active_tokens', 0.0),
        'num_occluded_tokens': occlusion_metrics.get('num_occluded_tokens', 0.0),
        'avg_token_visibility': occlusion_metrics.get('avg_visibility', 1.0),
    }
    
    return loss, metrics
