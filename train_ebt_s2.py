#!/usr/bin/env python3
"""
S2-EBT Training: Multi-Step Optimization with Landscape Regularization
======================================================================

Phase B: Add S2 tricks to improve energy landscape:
1. Multi-step optimization (K ∈ {3,4,5}, random per batch)
2. Random alpha per example (×[0.5, 1.5] jitter)
3. Adaptive step size (target 0.57 acceptance rate)
4. Scaled Langevin noise (σ = sqrt(2*α) * 0.5)
5. MIXED INITIALIZATION (prevents mode collapse):
   - 70% from decoder output (refinement)
   - 15% from replay buffer (persistent chains)
   - 15% from random noise (exploration)
6. Loss ONLY on final step (encourages K-step thinking)
7. Full TBPTT (no detach between steps)
8. Macro-averaged Dice loss (class-balanced)

Expected: Better energy landscape → stronger T/BoN gains
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
from hybrid_program_loss import (
    hyper_program_energy_loss,
    ProgramDistanceConfig,
    KindCounterEMA
)
from invariant_loss import (
    differentiable_hungarian_loss,
    permutation_invariant_token_loss
)

# Config and models loaded dynamically
config = None


class ReplayBuffer:
    """Simple ring buffer for y replays"""
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, y_tensor):
        """Add y (detached) to buffer"""
        self.buffer.append(y_tensor.detach().clone())
    
    def sample_like(self, y_template):
        """Sample a y with same shape as template (or return noise if no match)"""
        if len(self.buffer) == 0:
            return torch.randn_like(y_template)
        
        # Find samples with matching shape
        target_shape = y_template.shape
        matching = [y for y in self.buffer if y.shape == target_shape]
        
        if len(matching) == 0:
            # No matching size in buffer, return random noise
            return torch.randn_like(y_template)
        
        # Sample from matching entries
        idx = random.randint(0, len(matching) - 1)
        return matching[idx].to(y_template.device)
    
    def __len__(self):
        return len(self.buffer)


def s2_ebt_loss(model, batch, cfg, replay_buf=None, class_counts_ema=None):
    """
    S2-EBT training loss: multi-step optimize-then-supervise
    
    Args:
        model: EBTSystem
        batch: (x1, y1, x2, y2, x_test, y_star)
        cfg: config dict with alpha_base, K_choices, langevin_sigma, replay_prob
        replay_buf: ReplayBuffer instance (optional)
        class_counts_ema: EMA counts for class balancing
    
    Returns:
        loss, metrics
    """
    x1, y1, x2, y2, x_test, y_star = batch
    
    # CRITICAL: Check if x_test and y_star have matching sizes
    # In ARC, input and output can have different sizes - skip those
    if x_test.shape[1:] != y_star.shape[1:]:
        # Skip this batch - return dummy loss
        return torch.tensor(0.0, device=x_test.device, requires_grad=True), {
            'loss_ce': 0.0,
            'energy_init': 0.0,
            'energy_final': 0.0,
            'energy_delta': 0.0,
            'accuracy': 0.0,
            'K': 0,
            'alpha_mean': 0.0,
        }
    
    # Store shapes for verification
    B_orig, H_orig, W_orig = y_star.shape
    B_test, H_test, W_test = x_test.shape
    assert (B_orig, H_orig, W_orig) == (B_test, H_test, W_test), \
        f"Shapes still don't match after check: y_star={y_star.shape}, x_test={x_test.shape}"
    
    B, H, W = y_star.shape
    C = cfg['C']
    
    # 1. ENCODE context AND get decoder's initial prediction
    r1, r2, r_star, rbar, y_decoder_output = model(x1, y1, x2, y2, x_test)
    
    # 2. MIXED INITIALIZATION: decoder / replay / noise
    # This prevents mode collapse and trains the energy function to handle diverse inputs
    init_strategy = random.random()
    replay_prob = cfg.get('replay_prob', 0.2)
    noise_prob = cfg.get('noise_prob', 0.2)  # New: explicit noise initialization
    
    if init_strategy < (1.0 - replay_prob - noise_prob):
        # Strategy A: Start from decoder output (refinement)
        y = y_decoder_output.clone()
    elif init_strategy < (1.0 - noise_prob) and replay_buf is not None and len(replay_buf) > 0:
        # Strategy B: Start from replay buffer (persistent chains)
        y = replay_buf.sample_like(y_decoder_output)
    else:
        # Strategy C: Start from random noise (exploration)
        # Use small noise around the "uniform" logit value (log(1/C) ≈ -2.3 for C=10)
        uniform_logit = -torch.log(torch.tensor(float(C)))
        y = uniform_logit + torch.randn(B, H, W, C, device=x_test.device) * 0.5
    
    y.requires_grad_(True)
    
    # Verify y was initialized correctly
    assert y.shape == (B, H, W, C), f"Y initialization failed: got {y.shape}, expected ({B},{H},{W},{C})"
    
    # 3. RANDOM K and per-example alpha
    K = random.choice(cfg['K_choices'])
    alpha_factors = torch.empty(B, 1, 1, 1, device=y.device).uniform_(0.5, 1.5)
    alpha = cfg['alpha_base'] * alpha_factors  # (B,1,1,1)
    
    # CORRECT LANGEVIN NOISE SCALING: sigma = sqrt(2 * alpha)
    # We dampen it slightly (0.5 factor) to prefer optimization over pure sampling,
    # but it must scale with alpha to be effective.
    sigma = torch.sqrt(2 * alpha) * 0.5
    
    # 4. MULTI-STEP OPTIMIZATION with accept-only-if-energy-drops
    energy_trajectory = []
    accepted_steps = 0
    rejected_steps = 0
    first_try_successes = 0

    # Pre-compute shared encoder features for x_test
    h_x = model.encode_input_shared(x_test)  # (B, d_feat)

    for t in range(K):
        # Compute energy using shared encoder for x, energy's own differentiable encoding for y
        E = model.energy.energy_with_shared_input(h_x, rbar, y, canonical=False)  # (B,) raw energy
        energy_trajectory.append(E.mean().item())

        grad_y, = torch.autograd.grad(E.mean(), y, create_graph=True)

        # Langevin dynamics with backtracking
        noise = sigma * torch.randn_like(y)
        step_alpha = alpha.clone()
        max_retries = 2

        for retry in range(max_retries + 1):
            y_new = y - step_alpha * grad_y + noise

            # Verify shape
            assert y_new.shape == y.shape, f"Shape changed: {y.shape} -> {y_new.shape}"

            # Accept only if energy drops (or on last retry)
            with torch.no_grad():
                E_new = model.energy.energy_with_shared_input(h_x, rbar, y_new, canonical=False).mean().item()

            if E_new < E.mean().item() or retry == max_retries:
                y = y_new
                if E_new < E.mean().item():
                    accepted_steps += 1
                    if retry == 0:
                        first_try_successes += 1
                else:
                    rejected_steps += 1
                break
            else:
                # Reduce step size and retry
                step_alpha = step_alpha * 0.5
                rejected_steps += 1

        # Keep gradient graph (NO detach!)
        if t < K - 1:
            y.requires_grad_(True)
    
    # 5. SUPERVISE with Permutation-Invariant Hungarian Loss
    # Set-based matching + color invariance
    assert y.shape == (B, H, W, C), f"Final y has wrong shape: {y.shape} vs expected ({B}, {H}, {W}, {C})"
    
    # Use differentiable Hungarian loss with color permutation + occlusion penalty
    loss_ce, inv_metrics = differentiable_hungarian_loss(
        model=model,
        x_test=x_test,
        rbar=rbar,
        y_logits=y,
        y_star=y_star,
        w_match=1.0,         # Matched token cost
        w_unmatched=3.0,     # Unmatched penalty
        w_render=0.3,        # Render consistency (shape-aware)
        w_energy=1.0,        # Energy ranking
        base_margin=0.2,     # Base margin for ranking
        temperature=0.5,     # Gumbel-Softmax temperature for color permutation
        w_token=1.0,         # Token parameter supervision weight
        w_occlusion=0.1      # Occlusion penalty for 100% hidden tokens
    )
    
    # Convert non-diff metrics for compatibility
    hyper_metrics = {
        'pixel_acc': inv_metrics['pixel_acc'],
        'loss_rank': inv_metrics['loss_rank'],
        'loss_render': inv_metrics['loss_render'],
        'loss_reg': 0.0,  # Not used in invariant loss
        'prog_dist': inv_metrics.get('token_match_cost', 0.0),
        'energy_gap': inv_metrics['energy_gap'],
        'token_matches': inv_metrics.get('token_matches', 0.0),
        'tokens_pred': inv_metrics.get('tokens_pred', 0.0),
        'tokens_gt': inv_metrics.get('tokens_gt', 0.0),
        'color_alignment': inv_metrics['color_alignment'],
    }
    pde_metrics = hyper_metrics
    
    # 6. UPDATE replay buffer with final y
    if replay_buf is not None:
        with torch.no_grad():
            replay_buf.add(y.detach())
    
    # Combine metrics from HyPER loss with energy trajectory
    metrics = {
        'loss_ce': loss_ce.item(),
        'energy_init': energy_trajectory[0],
        'energy_final': energy_trajectory[-1],
        'energy_delta': energy_trajectory[-1] - energy_trajectory[0],
        'accuracy': hyper_metrics['pixel_acc'],
        'K': K,
        'accepted_steps': accepted_steps,
        'rejected_steps': rejected_steps,
        'first_try_rate': first_try_successes / K if K > 0 else 0.0,
        'alpha_mean': alpha.mean().item(),
        # HyPER-specific metrics
        'prog_dist': hyper_metrics['prog_dist'],
        'loss_rank': hyper_metrics['loss_rank'],
        'loss_render': hyper_metrics['loss_render'],
        'loss_token_render': inv_metrics.get('loss_token_render', 0.0),
        'loss_pixel_dice': inv_metrics.get('loss_pixel_dice', 0.0),
        'loss_reg': hyper_metrics['loss_reg'],
        'energy_gap': hyper_metrics['energy_gap'],
        'token_matches': hyper_metrics['token_matches'],
        'tokens_pred': hyper_metrics['tokens_pred'],
        'tokens_gt': hyper_metrics['tokens_gt'],
        'token_match_cost': inv_metrics.get('token_match_cost', 0.0),
        'loss_occlusion': inv_metrics.get('loss_occlusion', 0.0),
        'num_active_tokens': inv_metrics.get('num_active_tokens', 0.0),
        'num_occluded_tokens': inv_metrics.get('num_occluded_tokens', 0.0),
        'avg_token_visibility': inv_metrics.get('avg_token_visibility', 1.0),
    }
    
    return loss_ce, metrics


def train_epoch_s2(model, train_loader, optimizer, epoch, device, cfg, replay_buf):
    """Train one epoch with S2-EBT + Program-Distilled Energy loss"""
    model.train()
    
    losses = []
    metrics_list = {
        'energy_init': [],
        'energy_final': [],
        'energy_delta': [],
        'accuracy': [],
        'K': [],
        'accepted_steps': [],
        'rejected_steps': [],
        'alpha_mean': [],
        'grad_norm_theta': [],
        'prog_dist': [],
        'loss_rank': [],
        'loss_render': [],
        'loss_token_render': [],
        'loss_pixel_dice': [],
        'loss_reg': [],
        'energy_gap': [],
        'token_matches': [],
        'tokens_pred': [],
        'tokens_gt': [],
        'color_alignment': [],
        'token_match_cost': [],
        'loss_occlusion': [],
        'num_active_tokens': [],
        'num_occluded_tokens': [],
        'avg_token_visibility': [],
    }
    
    # Initialize token kind counter for group-balanced weighting
    class_counts_ema = KindCounterEMA(decay=0.999)
    
    skipped_count = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [S2-EBT]")
    for batch_list in pbar:
        batch_dict = batch_list[0]
        
        x1 = batch_dict['train_pairs'][0][0].unsqueeze(0).to(device)
        y1 = batch_dict['train_pairs'][0][1].unsqueeze(0).to(device)
        x2 = batch_dict['train_pairs'][1][0].unsqueeze(0).to(device)
        y2 = batch_dict['train_pairs'][1][1].unsqueeze(0).to(device)
        x_test = batch_dict['test_pair'][0].unsqueeze(0).to(device)
        y_star = batch_dict['test_pair'][1].unsqueeze(0).to(device)
        
        batch_data = (x1, y1, x2, y2, x_test, y_star)
        
        optimizer.zero_grad()
        loss, batch_metrics = s2_ebt_loss(model, batch_data, cfg, replay_buf, class_counts_ema)
        
        # Skip tracking if this was a skipped batch (K=0 indicates skip)
        if batch_metrics['K'] == 0:
            skipped_count += 1
            pbar.set_postfix({
                'Status': 'SKIP',
                'Skipped': skipped_count,
            })
            continue
        
        loss.backward()
        
        # Check energy param gradients
        energy_grad_norm = sum(p.grad.norm().item() for n, p in model.energy.named_parameters() 
                              if p.grad is not None)
        batch_metrics['grad_norm_theta'] = energy_grad_norm
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Track
        losses.append(loss.item())
        for k, v in batch_metrics.items():
            if k != 'loss_ce' and k in metrics_list:
                metrics_list[k].append(v)
        
        # Adaptive Step Size Tuning (Target 0.57 acceptance)
        # Heuristic: If acceptance < 0.57, step is too big (rejecting often) -> decrease alpha
        #            If acceptance > 0.57, step is too small (accepting too easily) -> increase alpha
        acc_rate = batch_metrics.get('first_try_rate', 0.57)
        target_rate = 0.57
        step_adjustment = 0.01 * (acc_rate - target_rate)  # Conservative adjustment
        cfg['alpha_base'] = cfg['alpha_base'] * (1.0 + step_adjustment)
        cfg['alpha_base'] = max(0.001, min(1.0, cfg['alpha_base']))

        pbar.set_postfix({
            'Loss': f"{loss.item():.3f}",
            'Acc': f"{batch_metrics['accuracy']:.3f}",
            'K': batch_metrics['K'],
            'ProgD': f"{batch_metrics.get('prog_dist', 0):.2f}",
            'α': f"{cfg['alpha_base']:.3f}",
            'AR': f"{acc_rate:.2f}"
        })
    
    avg_metrics = {
        'loss': np.mean(losses) if losses else 0.0,
        'energy_init': np.mean(metrics_list['energy_init']) if metrics_list['energy_init'] else 0,
        'energy_final': np.mean(metrics_list['energy_final']) if metrics_list['energy_final'] else 0,
        'energy_delta': np.mean(metrics_list['energy_delta']) if metrics_list['energy_delta'] else 0,
        'accuracy': np.mean(metrics_list['accuracy']) if metrics_list['accuracy'] else 0,
        'K_avg': np.mean(metrics_list['K']) if metrics_list['K'] else 0,
        'accepted_steps': np.mean(metrics_list['accepted_steps']) if metrics_list['accepted_steps'] else 0,
        'rejected_steps': np.mean(metrics_list['rejected_steps']) if metrics_list['rejected_steps'] else 0,
        'alpha_mean': np.mean(metrics_list['alpha_mean']) if metrics_list['alpha_mean'] else 0,
        'grad_norm_theta': np.mean(metrics_list['grad_norm_theta']) if metrics_list['grad_norm_theta'] else 0,
        'prog_dist': np.mean(metrics_list['prog_dist']) if metrics_list['prog_dist'] else 0,
        'loss_rank': np.mean(metrics_list['loss_rank']) if metrics_list['loss_rank'] else 0,
        'loss_render': np.mean(metrics_list['loss_render']) if metrics_list['loss_render'] else 0,
        'loss_token_render': np.mean(metrics_list['loss_token_render']) if metrics_list['loss_token_render'] else 0,
        'loss_pixel_dice': np.mean(metrics_list['loss_pixel_dice']) if metrics_list['loss_pixel_dice'] else 0,
        'loss_reg': np.mean(metrics_list['loss_reg']) if metrics_list['loss_reg'] else 0,
        'energy_gap': np.mean(metrics_list['energy_gap']) if metrics_list['energy_gap'] else 0,
        'token_matches': np.mean(metrics_list['token_matches']) if metrics_list['token_matches'] else 0,
        'tokens_pred': np.mean(metrics_list['tokens_pred']) if metrics_list['tokens_pred'] else 0,
        'tokens_gt': np.mean(metrics_list['tokens_gt']) if metrics_list['tokens_gt'] else 0,
        'color_alignment': np.mean(metrics_list['color_alignment']) if metrics_list['color_alignment'] else 0,
        'token_match_cost': np.mean(metrics_list['token_match_cost']) if metrics_list['token_match_cost'] else 0,
        'loss_occlusion': np.mean(metrics_list['loss_occlusion']) if metrics_list['loss_occlusion'] else 0,
        'num_active_tokens': np.mean(metrics_list['num_active_tokens']) if metrics_list['num_active_tokens'] else 0,
        'num_occluded_tokens': np.mean(metrics_list['num_occluded_tokens']) if metrics_list['num_occluded_tokens'] else 0,
        'avg_token_visibility': np.mean(metrics_list['avg_token_visibility']) if metrics_list['avg_token_visibility'] else 1.0,
        'replay_size': len(replay_buf) if replay_buf else 0,
        'skipped': skipped_count,
        'valid_batches': len(losses),
    }
    
    return avg_metrics


def validate(model, val_loader, device, T=20, alpha=0.05, N=2):
    """Validate with S2 inference (T steps, best-of-N)"""
    model.eval()
    
    correct = 0
    total = 0
    exact_match = 0
    skipped = 0
    
    C = 10
    
    pbar = tqdm(val_loader, desc=f"Val [S2: T={T}, N={N}]")
    for batch_list in pbar:
        batch_dict = batch_list[0]
        
        x1 = batch_dict['train_pairs'][0][0].unsqueeze(0).to(device)
        y1 = batch_dict['train_pairs'][0][1].unsqueeze(0).to(device)
        x2 = batch_dict['train_pairs'][1][0].unsqueeze(0).to(device)
        y2 = batch_dict['train_pairs'][1][1].unsqueeze(0).to(device)
        x_test = batch_dict['test_pair'][0].unsqueeze(0).to(device)
        y_star = batch_dict['test_pair'][1].unsqueeze(0).to(device)
        
        # Skip if sizes don't match (same as training)
        if x_test.shape[1:] != y_star.shape[1:]:
            skipped += 1
            continue
        
        B, H, W = y_star.shape
        
        # Encode AND get decoder prediction
        with torch.no_grad():
            _, _, _, rbar, y_decoder_output = model(x1, y1, x2, y2, x_test)
            h_x = model.encode_input_shared(x_test)  # (B, d_feat)

        # Best-of-N
        best_pred = None
        best_E = float('inf')

        for n in range(N):
            # Initialize from decoder output (same as training)
            y = y_decoder_output.clone()

            with torch.enable_grad():
                for t in range(T):
                    y.requires_grad_(True)
                    E = model.energy.energy_with_shared_input(h_x, rbar, y, canonical=False).mean()
                    grad_y, = torch.autograd.grad(E, y)
                    y = (y - alpha * grad_y).detach()

            with torch.no_grad():
                pred = y.argmax(dim=-1)
                y.requires_grad_(True)
                final_E = model.energy.energy_with_shared_input(h_x, rbar, y, canonical=False).mean().item()

            if final_E < best_E:
                best_E = final_E
                best_pred = pred
        
        correct += (best_pred == y_star).sum().item()
        total += y_star.numel()
        exact_match += (best_pred == y_star).all().sum().item()
        
        pbar.set_postfix({'Acc': f"{correct/total:.3f}", 'Exact': exact_match})
    
    return {
        'pixel_acc': correct / total if total > 0 else 0.0,
        'exact_match': exact_match,
        'skipped': skipped,
    }


def main():
    parser = argparse.ArgumentParser(description='S2-EBT Training')
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (increased from 1e-4)')
    parser.add_argument('--alpha_base', type=float, default=0.1, help='Base step size (increased from 0.05)')
    parser.add_argument('--K_min', type=int, default=3, help='Min optimization steps (increased from 2)')
    parser.add_argument('--K_max', type=int, default=5, help='Max optimization steps (increased from 3)')
    parser.add_argument('--langevin_sigma', type=float, default=0.01, help='Langevin noise scale (reduced from 0.05)')
    parser.add_argument('--replay_prob', type=float, default=0.15, help='Prob of starting from replay buffer')
    parser.add_argument('--noise_prob', type=float, default=0.15, help='Prob of starting from random noise')
    parser.add_argument('--replay_capacity', type=int, default=1000, help='Replay buffer size')
    parser.add_argument('--T_infer', type=int, default=20, help='Inference steps')
    parser.add_argument('--N_infer', type=int, default=2, help='Best-of-N for inference')
    parser.add_argument('--alpha_infer', type=float, default=0.05, help='Inference step size')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_ebt_s2')
    parser.add_argument('--resume', type=str, default=None, help='S1 checkpoint to fine-tune from')
    
    args = parser.parse_args()
    
    # Load config
    config_module = importlib.import_module(args.config)
    sys.modules['config'] = config_module
    global config
    config = config_module.config
    
    from models import EBTSystem
    
    print(f"\n{'='*60}")
    print("S2-EBT Training: Multi-Step Optimization")
    print(f"{'='*60}")
    print(f"Config: {args.config}.py")
    print(f"Training:")
    print(f"  K ∈ [{args.K_min}, {args.K_max}] (random per batch)")
    print(f"  alpha_base: {args.alpha_base} × Uniform(0.5, 1.5) (adaptive)")
    print(f"  Langevin σ: sqrt(2*α) * 0.5 (scaled)")
    print(f"  Initialization:")
    decoder_prob = 1.0 - args.replay_prob - args.noise_prob
    print(f"    {decoder_prob:.0%} decoder output (refinement)")
    print(f"    {args.replay_prob:.0%} replay buffer (persistent chains)")
    print(f"    {args.noise_prob:.0%} random noise (exploration)")
    print(f"Inference:")
    print(f"  T={args.T_infer}, N={args.N_infer}, alpha={args.alpha_infer}")
    print(f"{'='*60}\n")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = EBTSystem().to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
    
    # Resume from S1 checkpoint if specified
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        print(f"\nFine-tuning from S1 checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint.get('s2_pixel_acc', 0.0)
        print(f"S1 baseline: {best_acc:.4f}\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Replay buffer
    replay_buf = ReplayBuffer(capacity=args.replay_capacity)
    
    # S2 config
    cfg = {
        'C': 10,
        'alpha_base': args.alpha_base,
        'K_choices': list(range(args.K_min, args.K_max + 1)),
        'langevin_sigma': args.langevin_sigma,
        'replay_prob': args.replay_prob,
        'noise_prob': args.noise_prob,
    }
    
    # Training
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Replay buffer: {len(replay_buf)} samples")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        # Train
        train_metrics = train_epoch_s2(model, train_loader, optimizer, epoch, 
                                       args.device, cfg, replay_buf)
        
        # Validate
        val_metrics = validate(model, val_loader, args.device, 
                              T=args.T_infer, alpha=args.alpha_infer, N=args.N_infer)
        
        elapsed = time.time() - t0
        
        # Summary
        print(f"\nEpoch {epoch} Summary ({elapsed:.1f}s):")
        print(f"  Batches:        {train_metrics['valid_batches']}/400 (skipped: {train_metrics['skipped']})")
        print(f"  Train Loss:     {train_metrics['loss']:.4f}")
        print(f"  Train Acc:      {train_metrics['accuracy']:.4f}")
        print(f"  Color Alignment: {train_metrics['color_alignment']:.4f} ← PERMUTATION-INVARIANT")
        print(f"  Token Render Loss: {train_metrics.get('loss_token_render', 0):.4f} ← TOKEN-LEVEL SUPERVISION")
        print(f"  Token Match Cost: {train_metrics.get('token_match_cost', 0):.4f} ← TOKEN MATCHING")
        print(f"  Occlusion Penalty: {train_metrics.get('loss_occlusion', 0):.4f} (active: {train_metrics.get('num_active_tokens', 0):.1f}, occluded: {train_metrics.get('num_occluded_tokens', 0):.1f})")
        print(f"  Token Visibility: {train_metrics.get('avg_token_visibility', 1.0):.2f} ← Higher is better")
        print(f"  Program Dist:   {train_metrics['prog_dist']:.4f} ← TOKEN-LEVEL")
        print(f"  Token Matching: {train_metrics['token_matches']:.1f}/{train_metrics['tokens_gt']:.1f} matches")
        print(f"  Optimization:")
        print(f"    K (avg):      {train_metrics['K_avg']:.1f}")
        print(f"    Steps Accepted: {train_metrics['accepted_steps']:.1f}")
        print(f"    Steps Rejected: {train_metrics['rejected_steps']:.1f}")
        print(f"    Alpha (avg):  {train_metrics['alpha_mean']:.3f}")
        print(f"  Val Acc:        {val_metrics['pixel_acc']:.4f} ← MAIN METRIC")
        print(f"  Val Exact:      {val_metrics['exact_match']}/100")
        print(f"  Val Skipped:    {val_metrics.get('skipped', 0)}/100")
        print(f"  Energy:")
        print(f"    Init → Final: {train_metrics['energy_init']:.2f} → {train_metrics['energy_final']:.2f}")
        print(f"    Δ (trajectory): {train_metrics['energy_delta']:.2f}  ← Should be NEGATIVE!")
        print(f"    Gap (E- - E+):  {train_metrics['energy_gap']:.2f}  ← Should be POSITIVE!")
        print(f"  HyPER Components:")
        print(f"    Rank Loss:    {train_metrics['loss_rank']:.4f}")
        print(f"    Render Loss:  {train_metrics['loss_render']:.4f} ← Shape consistency")
        print(f"    Reg Loss:     {train_metrics['loss_reg']:.4f}")
        print(f"  Grad(θ):        {train_metrics['grad_norm_theta']:.1f}")
        print(f"  Replay size:    {train_metrics['replay_size']}")
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            's2_pixel_acc': val_metrics['pixel_acc'],
            'config': cfg,
        }
        
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt'))
        
        if val_metrics['pixel_acc'] > best_acc:
            best_acc = val_metrics['pixel_acc']
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pt'))
            print(f"  ✓ New best! Saved to {args.checkpoint_dir}/best.pt")
    
    print(f"\n{'='*60}")
    print("S2 Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Checkpoints in: {args.checkpoint_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
