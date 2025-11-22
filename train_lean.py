#!/usr/bin/env python3
"""
S2-EBT Training (Lean Baseline): Multi-Step Optimization with Object-Augmented Transformers
=======================================================================================

This script trains the "Lean" EBT system:
1. Object-Augmented ViT Encoder
2. Cross-Attention Rule Inductor
3. Patch-based Transformer Decoder

Loss:
- Primary: Standard Cross-Entropy on pixels (Teacher Forcing / MLE).
- Secondary (Test Time): Energy-based Re-Ranking (S2 inference).

Note: The complex differentiable renderer and token losses are REMOVED.
We focus on simple, robust pixel prediction first.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
import time
import os
import sys
import importlib
import random
from collections import deque

from dataloader_arc import F8a8fe49ARCDataset, collate_fn
from torch.utils.data import DataLoader

# Load Lean Model
from lean_model import LeanEBTSystem

config = None

def train_epoch(model, train_loader, optimizer, epoch, device, cfg):
    """Train one epoch with standard MLE loss."""
    model.train()
    
    losses = []
    accuracies = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Lean EBT]")
    for batch_list in pbar:
        # Process all items in the batch (batch_size items)
        batch_losses = []
        batch_accs = []
        
        for batch_dict in batch_list:
            # 1. Unpack Data
            # In ARC training, we typically use one pair as 'test' and others as 'train'
            # even if they are all training examples.
            # Our data loader gives pairs.
            pairs = batch_dict['train_pairs']
            test_pair = batch_dict['test_pair']
            
            # We can augment by rotating who is "test"
            # For simplicity, use standard split from loader
            x_test = test_pair[0].unsqueeze(0).to(device)
            y_test = test_pair[1].unsqueeze(0).to(device)
            
            x_train_list = [p[0].unsqueeze(0).to(device) for p in pairs]
            y_train_list = [p[1].unsqueeze(0).to(device) for p in pairs]
            
            # Skip if target size mismatches input size AND we rely on input size prediction
            # (For the baseline, we assume H_out = H_in, or we cheat and provide target shape)
            # To be fair, let's provide target shape during training (teacher forcing shape)
            target_shape = (y_test.shape[1], y_test.shape[2])
            
            # 2. Forward Pass
            # (B, H, W, C) logits
            logits = model(x_test, x_train_list, y_train_list, target_shape=target_shape)
            
            # 3. Loss (Cross Entropy)
            # logits: (B, H, W, C) -> (B, C, H, W)
            logits_perm = logits.permute(0, 3, 1, 2)
            loss = F.cross_entropy(logits_perm, y_test.long())
            
            # 4. Metrics
            pred = logits.argmax(dim=-1)
            acc = (pred == y_test).float().mean().item()
            
            batch_losses.append(loss)
            batch_accs.append(acc)
        
        # Aggregate losses and backprop
        if batch_losses:
            optimizer.zero_grad()
            total_loss = sum(batch_losses) / len(batch_losses)  # Average loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            avg_loss = total_loss.item()
            avg_acc = np.mean(batch_accs)
            
            losses.append(avg_loss)
            accuracies.append(avg_acc)
            
            pbar.set_postfix({
                'Loss': f"{avg_loss:.3f}",
                'Acc': f"{avg_acc:.3f}"
            })
        
    return {
        'loss': np.mean(losses) if losses else 0.0,
        'accuracy': np.mean(accuracies) if accuracies else 0.0
    }

def validate(model, val_loader, device):
    """Validate on held-out tasks."""
    model.eval()
    
    correct = 0
    total = 0
    exact_match = 0
    skipped = 0
    
    pbar = tqdm(val_loader, desc="Validation")
    for batch_list in pbar:
        batch_dict = batch_list[0]
        
        pairs = batch_dict['train_pairs']
        test_pair = batch_dict['test_pair']
        
        x_test = test_pair[0].unsqueeze(0).to(device)
        y_test = test_pair[1].unsqueeze(0).to(device)
        
        x_train_list = [p[0].unsqueeze(0).to(device) for p in pairs]
        y_train_list = [p[1].unsqueeze(0).to(device) for p in pairs]
        
        # Inference: We don't know target shape!
        # Heuristic: Assume output size = input size (common in ARC)
        # OR try to deduce from examples?
        # For baseline, we assume same size.
        target_shape = (x_test.shape[1], x_test.shape[2])
        
        # If GT size != Input size, we will fail or have mismatch.
        # Check if this task preserves size
        if (y_test.shape[1], y_test.shape[2]) != target_shape:
            # We skip evaluation on size-changing tasks for this specific baseline
            # unless we implement a size predictor.
            skipped += 1
            continue
            
        with torch.no_grad():
            logits = model(x_test, x_train_list, y_train_list, target_shape=target_shape)
            pred = logits.argmax(dim=-1)
            
        # Metrics
        correct += (pred == y_test).sum().item()
        total += y_test.numel()
        if (pred == y_test).all():
            exact_match += 1
            
    acc = correct / total if total > 0 else 0.0
    return {
        'pixel_acc': acc,
        'exact_match': exact_match,
        'skipped': skipped
    }

def main():
    parser = argparse.ArgumentParser(description='Lean EBT Training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_lean')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Lean EBT Training: Object-ViT Baseline")
    print(f"{'='*60}")
    print(f"Batch Size: {args.batch_size}")
    
    # Datasets
    train_dataset = F8a8fe49ARCDataset(
        task_id='f8a8fe49', split='train',
        num_problems_per_epoch=500, num_train_pairs=2,
        rearc_path='re-arc/re_arc/tasks'
    )
    
    val_dataset = F8a8fe49ARCDataset(
        task_id='f8a8fe49', split='val',
        num_problems_per_epoch=100, num_train_pairs=2,
        rearc_path='re-arc/re_arc/tasks'
    )
    
    # Datasets
    train_dataset = F8a8fe49ARCDataset(
        task_id='f8a8fe49', split='train',
        num_problems_per_epoch=500, num_train_pairs=2,
        rearc_path='re-arc/re_arc/tasks'
    )
    
    val_dataset = F8a8fe49ARCDataset(
        task_id='f8a8fe49', split='val',
        num_problems_per_epoch=100, num_train_pairs=2,
        rearc_path='re-arc/re_arc/tasks'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = LeanEBTSystem().to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, args.device, {})
        val_metrics = validate(model, val_loader, args.device)
        
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Acc:  {train_metrics['accuracy']:.4f}")
        print(f"  Val Acc:    {val_metrics['pixel_acc']:.4f}")
        print(f"  Val Exact:  {val_metrics['exact_match']}/100")
        print(f"  Skipped:    {val_metrics['skipped']}")
        
        # Save
        if val_metrics['pixel_acc'] > best_acc:
            best_acc = val_metrics['pixel_acc']
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best.pt'))
            print("  ✓ New best model saved!")
            
    print(f"\nTraining Complete. Best Val Acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()

