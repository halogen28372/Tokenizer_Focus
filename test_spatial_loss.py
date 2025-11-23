import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from spatial_loss import SpatialAttractorLoss

def test_spatial_loss():
    print("Testing SpatialAttractorLoss...")
    
    # 1. Setup Dimensions
    B, C, H, W = 1, 10, 10, 10  # 10x10 grid, 10 colors
    
    # 2. Create Dummy Ground Truth
    # Let's place a single red pixel (color 2) at (5, 5)
    targets = torch.zeros((B, H, W), dtype=torch.long)
    targets[0, 5, 5] = 2
    
    # 3. Initialize Loss
    loss_fn = SpatialAttractorLoss(tau=2.0, ignore_index=0)
    
    # 4. Create Predictions (Logits)
    # Case A: Perfect Prediction
    logits_perfect = torch.zeros((B, C, H, W))
    logits_perfect[0, 0, :, :] = 10.0 # Default predict background everywhere
    logits_perfect[0, 2, 5, 5] = 20.0 # Predict red at correct spot
    
    # Case B: Near Miss (predict red at 5, 6 - 1 pixel away)
    logits_near = torch.zeros((B, C, H, W))
    logits_near[0, 0, :, :] = 10.0
    logits_near[0, 2, 5, 6] = 20.0 # 1 pixel right
    
    # Case C: Far Miss (predict red at 0, 0 - 7 pixels away)
    logits_far = torch.zeros((B, C, H, W))
    logits_far[0, 0, :, :] = 10.0
    logits_far[0, 2, 0, 0] = 20.0
    
    # Case D: No Prediction (Background only)
    logits_bg = torch.zeros((B, C, H, W))
    logits_bg[0, 0, :, :] = 20.0
    
    # 5. Compute Losses
    loss_perfect = loss_fn(logits_perfect, targets)
    loss_near = loss_fn(logits_near, targets)
    loss_far = loss_fn(logits_far, targets)
    loss_bg = loss_fn(logits_bg, targets)
    
    print(f"Loss Perfect: {loss_perfect.item():.4f} (Should be lowest)")
    print(f"Loss Near:    {loss_near.item():.4f}   (Should be low)")
    print(f"Loss Far:     {loss_far.item():.4f}   (Should be higher)")
    print(f"Loss None:    {loss_bg.item():.4f}   (Should be highest or close to Far)")
    
    # 6. Verify Gradients
    logits_near.requires_grad = True
    loss = loss_fn(logits_near, targets)
    loss.backward()
    
    # Check gradient at the target location (5,5) for class 2
    # The loss wants to INCREASE prob at (5,5), so gradient of loss w.r.t. logit should be negative?
    # Loss = - sum(prob * reward)
    # dLoss/dLogit ... 
    # If we increase logit at (5,5), prob increases, reward is 1.0 there, so Loss becomes more negative (smaller).
    # So gradient should be negative.
    grad_at_target = logits_near.grad[0, 2, 5, 5]
    print(f"Gradient at target (5,5) for class 2: {grad_at_target.item():.4f}")
    
    if grad_at_target < 0:
        print("✓ Gradient direction correct (pushing to increase target probability)")
    else:
        print("⚠ Gradient direction unexpected")

    # 7. Visualize Reward Field
    rewards = loss_fn.compute_reward_fields(targets, C)
    # rewards is [B, C, H, W]
    # Look at class 2 reward map
    r_map = rewards[0, 2].cpu().numpy()
    
    # Verify peak is at 5,5
    peak_val = r_map[5, 5]
    print(f"Reward at target (5,5): {peak_val:.4f} (Should be 1.0)")
    
    # Verify decay
    dist_1 = r_map[5, 6]
    expected_decay = np.exp(-1.0 / 2.0) # tau=2.0, dist=1.0
    print(f"Reward at dist 1 (5,6): {dist_1:.4f} (Expected ~{expected_decay:.4f})")
    
    return True

if __name__ == "__main__":
    test_spatial_loss()

