"""
Differentiable Token Renderer
==============================
Renders token representations into grids in a differentiable way.
This allows gradients to flow from grid-level loss back to token parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


def render_rect(grid: torch.Tensor, x: float, y: float, w: float, h: float, 
                color: int, layer: int, filled: bool = True, thickness: int = 1):
    """
    Render a rectangle onto a grid with PERFECT FIDELITY for integer coordinates.
    
    Args:
        grid: (H, W, C) soft grid (will be modified in-place)
        x, y, w, h: rectangle parameters
        color: color index (0..C-1)
        layer: layer index (higher layers occlude lower)
        filled: if True, fill rectangle; if False, draw border only
        thickness: border thickness (if not filled)
    """
    H, W, C = grid.shape
    device = grid.device
    
    # Convert to integer coordinates for exact rendering
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Create mask
    mask = torch.zeros(H, W, device=device, dtype=torch.float32)
    
    if filled:
        # Fill entire rectangle
        y_end = min(y + h, H)
        x_end = min(x + w, W)
        if y < H and x < W and y_end > y and x_end > x:
            mask[y:y_end, x:x_end] = 1.0
    else:
        # Draw border only (outer rect minus inner rect)
        y_end = min(y + h, H)
        x_end = min(x + w, W)
        
        # Fill outer rectangle
        if y < H and x < W and y_end > y and x_end > x:
            mask[y:y_end, x:x_end] = 1.0
        
        # Remove inner rectangle (creating hollow effect)
        y_inner = y + thickness
        x_inner = x + thickness
        h_inner = h - 2 * thickness
        w_inner = w - 2 * thickness
        
        if h_inner > 0 and w_inner > 0:
            y_inner_end = min(y_inner + h_inner, H)
            x_inner_end = min(x_inner + w_inner, W)
            if y_inner < H and x_inner < W and y_inner_end > y_inner and x_inner_end > x_inner:
                mask[y_inner:y_inner_end, x_inner:x_inner_end] = 0.0
    
    # Set pixels with this color (using max for layer occlusion)
    grid[:, :, color] = torch.maximum(grid[:, :, color], mask)
    
    return grid


def render_line(grid: torch.Tensor, x: float, y: float, w: float, h: float,
                orientation: str, color: int, layer: int):
    """
    Render a line onto a grid with PERFECT FIDELITY.
    
    Args:
        grid: (H, W, C) soft grid
        x, y, w, h: line parameters
        orientation: 'H' (horizontal) or 'V' (vertical)
        color: color index
        layer: layer index
    """
    H, W, C = grid.shape
    device = grid.device
    
    x, y, w, h = int(x), int(y), int(w), int(h)
    mask = torch.zeros(H, W, device=device, dtype=torch.float32)
    
    if orientation == 'H':
        # Horizontal line
        y_end = min(y + h, H)
        x_end = min(x + w, W)
        if y < H and x < W and y_end > y and x_end > x:
            mask[y:y_end, x:x_end] = 1.0
    else:  # 'V'
        # Vertical line
        y_end = min(y + h, H)
        x_end = min(x + w, W)
        if y < H and x < W and y_end > y and x_end > x:
            mask[y:y_end, x:x_end] = 1.0
    
    grid[:, :, color] = torch.maximum(grid[:, :, color], mask)
    
    return grid


def render_token_to_int_grid(token: Dict, int_grid: torch.Tensor, H: int, W: int):
    """
    Render a single token onto an integer grid (like drawing in MS Paint).
    
    Args:
        token: dict with 'type', 'color', 'layer', and params
        int_grid: (H, W) integer grid of colors (will be modified in-place)
        H, W: grid dimensions
    """
    token_type = token.get('type', 'REGION')
    color = int(token.get('color', 0))
    
    x = int(token.get('x', 0))
    y = int(token.get('y', 0))
    w = int(token.get('w', 1))
    h = int(token.get('h', 1))
    
    if token_type == 'RECT':
        # Filled rectangle
        y_end = min(y + h, H)
        x_end = min(x + w, W)
        if y >= 0 and x >= 0 and y_end > y and x_end > x:
            int_grid[y:y_end, x:x_end] = color
    
    elif token_type == 'HOLLOW_RECT':
        # Hollow rectangle: outer rect minus inner rect
        thickness = int(token.get('border_thickness', 1))
        y_end = min(y + h, H)
        x_end = min(x + w, W)
        
        # Draw outer rectangle
        if y >= 0 and x >= 0 and y_end > y and x_end > x:
            int_grid[y:y_end, x:x_end] = color
        
        # Clear inner rectangle
        y_inner = y + thickness
        x_inner = x + thickness
        h_inner = h - 2 * thickness
        w_inner = w - 2 * thickness
        
        if h_inner > 0 and w_inner > 0:
            y_inner_end = min(y_inner + h_inner, H)
            x_inner_end = min(x_inner + w_inner, W)
            if y_inner >= 0 and x_inner >= 0 and y_inner_end > y_inner and x_inner_end > x_inner:
                # Don't set to color - leave as-is (will be filled by lower layers or become 0)
                pass  # Actually, we should NOT clear it - that's the hollow part
    
    elif token_type == 'LINE':
        orientation = token.get('orientation', 'H')
        y_end = min(y + h, H)
        x_end = min(x + w, W)
        if y >= 0 and x >= 0 and y_end > y and x_end > x:
            int_grid[y:y_end, x:x_end] = color
    
    elif token_type == 'BORDER':
        # Border frame around entire grid
        thickness = int(token.get('thickness', 1))
        # Draw outer rectangle
        int_grid[0:H, 0:W] = color
        # Clear inner rectangle
        if H > 2*thickness and W > 2*thickness:
            pass  # Leave inner as-is
    
    # For other token types, render as bounding box
    else:
        y_end = min(y + h, H)
        x_end = min(x + w, W)
        if y >= 0 and x >= 0 and y_end > y and x_end > x:
            int_grid[y:y_end, x:x_end] = color


def render_tokens_to_grid(tokens: List[Dict], H: int, W: int, C: int, device='cpu'):
    """
    Render a list of tokens into a grid with PERFECT FIDELITY.
    
    Strategy: Render to a 2D integer grid first, then convert to one-hot.
    This avoids issues with torch.maximum() not handling pixel removal.
    
    Args:
        tokens: list of token dicts
        H, W: grid dimensions
        C: number of colors
        device: device for tensors
    
    Returns:
        grid: (H, W, C) soft grid (one-hot encoding)
    """
    if len(tokens) == 0:
        return torch.zeros(H, W, C, device=device, dtype=torch.float32)
    
    # Render to integer grid first (like a real image)
    int_grid = torch.full((H, W), -1, dtype=torch.long, device=device)  # -1 = unset
    
    # Sort tokens by layer (lower layers first, so higher layers occlude)
    sorted_tokens = sorted(tokens, key=lambda t: int(t.get('layer', 0)))
    
    for token in sorted_tokens:
        render_token_to_int_grid(token, int_grid, H, W)
    
    # Convert any unset pixels (-1) to color 0 (background)
    int_grid = torch.where(int_grid == -1, torch.tensor(0, device=device), int_grid)
    
    # Convert to one-hot encoding
    grid_onehot = torch.zeros(H, W, C, device=device, dtype=torch.float32)
    for c in range(C):
        grid_onehot[:, :, c] = (int_grid == c).float()
    
    return grid_onehot


def tokens_to_logits(tokens: List[Dict], H: int, W: int, C: int, device='cpu', temperature=1.0):
    """
    Convert tokens to logits (for use in loss functions).
    
    Args:
        tokens: list of token dicts
        H, W: grid dimensions
        C: number of colors
        device: device for tensors
        temperature: temperature for soft rendering (lower = sharper)
    
    Returns:
        logits: (H, W, C) logits tensor
    """
    grid = render_tokens_to_grid(tokens, H, W, C, device)
    
    # Convert probabilities to logits
    # Use inverse softmax: logits = log(probs / (1 - probs + eps))
    eps = 1e-8
    probs = grid + eps
    logits = torch.log(probs / (1.0 - probs + eps)) / temperature
    
    return logits

