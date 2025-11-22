"""
Token Occlusion Penalty
========================
Penalizes tokens that are 100% hidden by other tokens (wasted capacity).
"""

import torch
import torch.nn.functional as F


def compute_token_occlusion_penalty(decoder_state, w_occlusion=0.1, visibility_threshold=0.05, presence_threshold=0.3):
    """
    Compute a penalty for tokens that are fully occluded (not visible in final output).
    
    Uses actual visibility_ratio computed by the decoder to penalize tokens that are
    present but contribute nothing to the final output (100% occluded).
    
    Args:
        decoder_state: dict from TokenDecoder._render_tokens with:
            - presence_logits: (B, T, 1)
            - visibility_ratio: (B, T) - fraction of token visible in output [0, 1]
        w_occlusion: weight for occlusion penalty
        visibility_threshold: tokens below this visibility are considered occluded
        presence_threshold: tokens above this presence are considered "active"
    
    Returns:
        occlusion_loss: scalar loss
        metrics: dict with occlusion statistics
    """
    if decoder_state is None:
        return torch.tensor(0.0), {}
    
    presence_logits = decoder_state.get('presence_logits')
    visibility_ratio = decoder_state.get('visibility_ratio')
    
    if presence_logits is None or visibility_ratio is None:
        return torch.tensor(0.0), {}
    
    # Get presence probabilities
    presence = torch.sigmoid(presence_logits).squeeze(-1)  # (B, T)
    
    # Verify shapes match
    if presence.shape != visibility_ratio.shape:
        # Shape mismatch, return zero loss
        return torch.tensor(0.0), {}
    
    # Occlusion penalty: high presence + low visibility = wasted token
    # We want to penalize: presence * (1 - visibility) for active tokens
    
    # Only count tokens that are "active" (high presence)
    active_mask = (presence > presence_threshold).float()
    
    # Compute occlusion: how much presence is wasted on invisible tokens
    occlusion = presence * (1.0 - visibility_ratio)  # (B, T)
    
    # Only penalize active tokens that are occluded
    occluded_mask = (visibility_ratio < visibility_threshold).float()
    wasted_presence = occlusion * active_mask * occluded_mask
    
    # Average loss
    occlusion_loss = wasted_presence.mean()
    
    # Metrics
    with torch.no_grad():
        num_active = active_mask.sum(dim=1).mean().item()
        num_occluded = (active_mask * occluded_mask).sum(dim=1).mean().item()
        avg_visibility = (visibility_ratio * active_mask).sum(dim=1).mean().item()
        avg_wasted = wasted_presence.sum(dim=1).mean().item()
    
    metrics = {
        'occlusion_loss': occlusion_loss.item(),
        'num_active_tokens': num_active,
        'num_occluded_tokens': num_occluded,
        'avg_visibility': avg_visibility,
        'avg_wasted_presence': avg_wasted,
    }
    
    return w_occlusion * occlusion_loss, metrics


def compute_token_visibility(decoder_state, visibility_threshold=0.01):
    """
    Compute per-token visibility in the final rendered output (more accurate).
    
    This requires rendering each token individually and comparing to the final output.
    More expensive but more accurate than the heuristic above.
    
    Args:
        decoder_state: dict from TokenDecoder with full rendering info
        visibility_threshold: minimum contribution ratio to consider visible
    
    Returns:
        visibility: (B, T) tensor with visibility ratio [0, 1] for each token
    """
    # This would require the decoder to save per-token rendered grids
    # which is memory-intensive but more accurate
    
    # Implementation would:
    # 1. For each token i:
    #    - Render only that token
    #    - Compare to final output (IoU or L1 distance)
    #    - visibility[i] = contribution of token i to final output
    # 2. Penalize tokens where visibility[i] < threshold but presence[i] > threshold
    
    raise NotImplementedError("Requires per-token rendering in decoder")


def add_token_efficiency_penalty(loss_dict, decoder_state, w_occlusion=0.1):
    """
    Add token efficiency penalties to existing loss dict.
    
    Args:
        loss_dict: dict with 'loss' and other metrics
        decoder_state: dict from TokenDecoder
        w_occlusion: weight for occlusion penalty
    
    Returns:
        Updated loss_dict with occlusion penalty added
    """
    occlusion_loss, occlusion_metrics = compute_token_occlusion_penalty(
        decoder_state, w_occlusion=w_occlusion
    )
    
    # Add to total loss
    loss_dict['loss'] = loss_dict['loss'] + occlusion_loss
    
    # Add metrics
    loss_dict['occlusion_loss'] = occlusion_metrics['occlusion_loss']
    loss_dict['num_active_tokens'] = occlusion_metrics['num_active_tokens']
    loss_dict['avg_occlusion_risk'] = occlusion_metrics['avg_occlusion_risk']
    
    return loss_dict

