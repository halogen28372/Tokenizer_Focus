import torch
import numpy as np
import random

class TTTAugmenter:
    """
    Test-Time Training Augmenter.
    Generates auxiliary support pairs for S2 optimization using rigid transformations
    and color permutations.
    """
    def __init__(self, num_augmentations=50, device='cpu'):
        self.num_augmentations = num_augmentations
        self.device = device
        
    def augment(self, x_train_list, y_train_list):
        """
        Generate augmented pairs from the original support set.
        
        Args:
            x_train_list: List of (1, H, W) input tensors
            y_train_list: List of (1, H, W) output tensors
            
        Returns:
            augmented_x_list: List of augmented (1, H, W) input tensors
            augmented_y_list: List of augmented (1, H, W) output tensors
        """
        aug_x = []
        aug_y = []
        
        # Always include originals
        aug_x.extend(x_train_list)
        aug_y.extend(y_train_list)
        
        if not x_train_list:
            return aug_x, aug_y
            
        # Calculate how many augmentations per example
        # We want roughly num_augmentations total new pairs
        num_orig = len(x_train_list)
        augs_per_ex = max(1, self.num_augmentations // num_orig)
        
        for idx in range(num_orig):
            x = x_train_list[idx] # (1, H, W)
            y = y_train_list[idx] # (1, H, W)
            
            for _ in range(augs_per_ex):
                x_aug, y_aug = self._apply_random_transform(x, y)
                aug_x.append(x_aug)
                aug_y.append(y_aug)
                
        return aug_x, aug_y
    
    def _apply_random_transform(self, x, y):
        """Apply a random rigid transformation + color permutation to a pair."""
        # Copy tensors
        x_aug = x.clone()
        y_aug = y.clone()
        
        # 1. Rigid Transformations (Dihedral Group D4)
        # 8 symmetries: Identity, Rot90, Rot180, Rot270, and their horizontal flips
        
        # Random Flip
        if random.random() > 0.5:
            x_aug = torch.flip(x_aug, [-1]) # Flip width
            y_aug = torch.flip(y_aug, [-1])
            
        if random.random() > 0.5:
            x_aug = torch.flip(x_aug, [-2]) # Flip height
            y_aug = torch.flip(y_aug, [-2])
            
        # Random Rotation (0, 1, 2, 3 times 90 degrees)
        k = random.randint(0, 3)
        if k > 0:
            x_aug = torch.rot90(x_aug, k, [-2, -1])
            y_aug = torch.rot90(y_aug, k, [-2, -1])
            
        # 2. Color Permutation
        # Map colors 0-9 to a random permutation of 0-9
        # Note: We preserve 0 (background) usually, but the paper suggests aggressive permutation.
        # Let's try full permutation for now, but maybe keep 0 fixed with 50% prob?
        # ARC tasks often treat 0 as special (background).
        # Let's do a safe permutation: permute 1-9, keep 0 fixed.
        # Or just full permutation. Let's try full permutation.
        
        perm = torch.randperm(10).to(self.device)
        
        # Apply permutation
        # x_aug and y_aug are long tensors with indices. We can use simple indexing or gather.
        # Since values are 0-9, we can just do x_aug = perm[x_aug]
        x_aug = perm[x_aug]
        y_aug = perm[y_aug]
        
        return x_aug, y_aug

