import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttractorLoss(nn.Module):
    """
    Computes a spatial attraction loss using a GPU-native Euclidean Distance Transform.
    Perfect for ARC 'soft spatial' exploration.
    """
    def __init__(self, tau=1.5, ignore_index=0, reduction='mean'):
        super().__init__()
        self.tau = tau
        self.ignore_index = ignore_index
        self.reduction = reduction

    def compute_reward_fields(self, targets, num_classes):
        """
        Computes soft distance reward maps entirely on GPU using vectorized ops.
        targets: [B, H, W]
        Returns: [B, C, H, W] (Values 0 to 1, where 1 is 'on top of GT')
        """
        B, H, W = targets.shape
        device = targets.device
        
        # 1. Create Coordinate Grid [B, H*W, 2]
        # We flatten spatial dims to treat pixels as a point cloud
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device), 
                                        indexing='ij')
        # [B, H*W, 2]
        flat_coords = torch.stack([y_grid, x_grid], dim=-1).float().view(1, -1, 2).expand(B, -1, -1)
        
        reward_maps = []
        
        for c in range(num_classes):
            # SKIP BACKGROUND: If this is the background color, return zero field
            # (unless you specifically want background gravity, but usually you don't)
            if c == self.ignore_index:
                reward_maps.append(torch.zeros((B, H, W), device=device))
                continue
            # Binary mask for class c [B, H, W]
            mask = (targets == c)
            
            class_rewards = []
            
            # Iterate over batch (Batch size is usually small, this loop is negligible compared to pixel loops)
            for b in range(B):
                target_pixels = torch.nonzero(mask[b], as_tuple=False).float() # [N_targets, 2]
                
                if target_pixels.shape[0] == 0:
                    # Color not present in this GT sample
                    class_rewards.append(torch.zeros((H, W), device=device))
                    continue
                
                # 2. Vectorized Distance Calculation (The Magic Step)
                # computes dist between every grid pixel and every target pixel
                # dists: [H*W, N_targets]
                dists = torch.cdist(flat_coords[b], target_pixels)
                
                # 3. Nearest Neighbor Distance (EDT equivalent)
                # min_dist: [H*W]
                min_dist, _ = dists.min(dim=1)
                
                # 4. Convert to Heatmap (Exp decay)
                # reward: [H, W]
                reward = torch.exp(-min_dist.view(H, W) / self.tau)
                class_rewards.append(reward)
            
            reward_maps.append(torch.stack(class_rewards))
            
        return torch.stack(reward_maps, dim=1) # [B, C, H, W]

    def forward(self, logits, targets):
        # logits: [B, C, H, W]
        # targets: [B, H, W]
        num_classes = logits.shape[1]
        
        # Probabilities
        probs = F.softmax(logits, dim=1)
        
        # Generate Fields (Ground Truth "Gravity")
        with torch.no_grad():
            rewards = self.compute_reward_fields(targets, num_classes)
        
        # Loss = Negative Expected Reward
        # We want p=1 where reward=1.
        loss_map = -(probs * rewards).sum(dim=1) # Sum over classes
        
        if self.reduction == 'mean':
            return loss_map.mean()
        elif self.reduction == 'sum':
            return loss_map.sum()
        return loss_map

