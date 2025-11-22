"""
Standalone visualizer for comparing model predictions against ARC ground truth.

Usage:
    python visualize_checkpoint.py \
        --checkpoint checkpoints_ebt_s2/best.pt \
        --data-dir Data/ARC \
        --split evaluation \
        --num-tasks 8 \
        --device cpu \
        --output-dir visualizations/best_run
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from config import config
from dataloader_arc import ARCDataset
from models import EBTSystem


# ======== ARC COLOR MAP ===========
ARC_COLORS = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: sky
    '#870C25',  # 9: maroon
]
cmap = ListedColormap(ARC_COLORS)


# ======== MODEL HELPERS ===========
def load_model(checkpoint_path: str, device: str = 'cpu') -> EBTSystem:
    model = EBTSystem().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def run_s2_inference(model: EBTSystem,
                     x1: torch.Tensor,
                     y1: torch.Tensor,
                     x2: torch.Tensor,
                     y2: torch.Tensor,
                     x_test: torch.Tensor,
                     device: str = 'cpu',
                     T: int = 0,
                     alpha: float = 0.05,
                     sigma: float = 0.0,
                     N: int = 1,
                     use_backtracking: bool = True):
    """Multi-step S2 inference loop with Best-of-N. If T=0, return direct decoder output."""
    r1, r2, r_star, rbar, y_decoder_output = model(x1, y1, x2, y2, x_test)

    # If T=0, return decoder output directly (no optimization)
    if T == 0:
        pred = y_decoder_output.argmax(dim=-1)
        return pred, rbar

    # Best-of-N: run N times and pick the one with lowest energy
    h_x = model.encode_input_shared(x_test)
    best_pred = None
    best_E = float('inf')
    
    # For BoN, add initialization noise to create diverse starting points
    init_noise_scale = 0.1 if N > 1 else 0.0

    for n in range(N):
        # Initialize from decoder output with unique noise for each run
        if N > 1:
            # Add initialization noise to create diverse starting points
            init_noise = init_noise_scale * torch.randn_like(y_decoder_output)
            y_logits = y_decoder_output.clone() + init_noise
        else:
            y_logits = y_decoder_output.clone()

        with torch.enable_grad():
            for t in range(T):
                y_logits.requires_grad_(True)
                E = model.energy.energy_with_shared_input(h_x, rbar, y_logits, canonical=False)
                grad_y, = torch.autograd.grad(E.mean(), y_logits, create_graph=False)
                
                if use_backtracking:
                    # Backtracking line search (like training)
                    step_alpha = alpha
                    max_retries = 2
                    E_current = E.mean().item()
                    
                    for retry in range(max_retries + 1):
                        with torch.no_grad():
                            noise = sigma * torch.randn_like(y_logits) if sigma > 0 else 0.0
                            y_new = y_logits - step_alpha * grad_y + noise
                            y_new.requires_grad_(True)
                            E_new = model.energy.energy_with_shared_input(h_x, rbar, y_new, canonical=False).mean().item()
                        
                        # Accept if energy drops or on last retry
                        if E_new < E_current or retry == max_retries:
                            y_logits = y_new.detach()
                            break
                        else:
                            # Reduce step size and retry
                            step_alpha = step_alpha * 0.5
                else:
                    # Blind gradient descent (original implementation)
                    with torch.no_grad():
                        noise = sigma * torch.randn_like(y_logits) if sigma > 0 else 0.0
                        y_logits = (y_logits - alpha * grad_y + noise).detach()

        # Evaluate final energy and keep best
        with torch.no_grad():
            pred = y_logits.argmax(dim=-1)
            final_E = model.energy.energy_with_shared_input(h_x, rbar, y_logits, canonical=False).mean().item()
            
            if final_E < best_E:
                best_E = final_E
                best_pred = pred

    return best_pred, rbar


def extract_tokens(model: EBTSystem,
                   x_test: torch.Tensor,
                   rbar: torch.Tensor,
                   threshold: float = 0.3):
    """Return token predictions from the token decoder if available."""
    decoder = getattr(model, 'decoder', None)
    if decoder is None or not hasattr(decoder, 'export_token_lists'):
        return []

    with torch.no_grad():
        _ = decoder(x_test, rbar)
        tokens = decoder.export_token_lists(threshold=threshold) or []

    return tokens[0] if tokens else []


# ======== VISUALIZATION ===========
def add_grid(ax, grid):
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax.axis('off')
    ax.grid(True, which='both', color='gray', linewidth=0.4, alpha=0.3)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))


def format_tokens(tokens: List[dict], max_tokens: int = 9) -> str:
    if not tokens:
        return "No tokens emitted (decoder inactive or below threshold)."

    lines = [f"Predicted Tokens (showing up to {max_tokens}):", ""]
    for idx, token in enumerate(tokens[:max_tokens]):
        line = (
            f"[{idx+1:02d}] {token.get('type','?')} | "
            f"color={token.get('color','?')} | "
            f"pos=({token.get('x',0):.1f},{token.get('y',0):.1f}) "
            f"size=({token.get('w',0):.1f}×{token.get('h',0):.1f}) "
            f"presence={token.get('presence',0):.2f}"
        )
        lines.append(line)

    if len(tokens) > max_tokens:
        lines.append("")
        lines.append(f"... {len(tokens) - max_tokens} additional tokens hidden ...")

    return "\n".join(lines)


def visualize_example(model: EBTSystem,
                      sample: dict,
                      device: str,
                      args,
                      save_path: Optional[Path] = None,
                      T: int = 0,
                      alpha: float = 0.05,
                      use_backtracking: bool = True):
    """Create a side-by-side visualization for a single ARC task."""
    x1, y1 = sample['train_pairs'][0]
    x2, y2 = sample['train_pairs'][1]
    x_test, y_gt = sample['test_pair']

    # Handle variable sizes
    if x_test.shape != y_gt.shape:
        print(f"⚠ Skipping {sample['task_id']} due to size mismatch "
              f"{x_test.shape} vs {y_gt.shape}")
        return None

    # Add batch dimension / move to device
    x1 = x1.unsqueeze(0).to(device)
    y1 = y1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)
    y2 = y2.unsqueeze(0).to(device)
    x_test = x_test.unsqueeze(0).to(device)
    y_gt = y_gt.unsqueeze(0).to(device)

    pred_grid, rbar = run_s2_inference(model, x1, y1, x2, y2, x_test, device, T=T, alpha=alpha, sigma=args.sigma, N=args.N, use_backtracking=use_backtracking)
    tokens = extract_tokens(model, x_test, rbar)

    pred_np = pred_grid[0].cpu().numpy()
    gt_np = y_gt[0].cpu().numpy()
    acc = (pred_np == gt_np).mean()

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4, hspace=0.25, wspace=0.3)

    axes_info = [
        (gs[0, 0], x1[0].cpu().numpy(), 'Train Input 1'),
        (gs[0, 1], y1[0].cpu().numpy(), 'Train Output 1'),
        (gs[0, 2], x2[0].cpu().numpy(), 'Train Input 2'),
        (gs[0, 3], y2[0].cpu().numpy(), 'Train Output 2'),
        (gs[1, 1], x_test[0].cpu().numpy(), 'Test Input'),
        (gs[1, 2], pred_np, f'Prediction (Acc: {acc:.1%})'),
        (gs[1, 3], gt_np, 'Ground Truth'),
    ]

    for spec, grid, title in axes_info:
        ax = fig.add_subplot(spec)
        add_grid(ax, grid)
        ax.set_title(title, fontsize=10, fontweight='bold')

    # Token summary
    ax_tokens = fig.add_subplot(gs[2, :])
    ax_tokens.axis('off')
    ax_tokens.text(0.02, 0.5, format_tokens(tokens), fontsize=9,
                   family='monospace', va='center', ha='left')

    # Color legend
    legend_elements = [mpatches.Patch(facecolor=ARC_COLORS[i], label=str(i))
                       for i in range(len(ARC_COLORS))]
    fig.legend(handles=legend_elements, loc='upper right',
               title='ARC Colors', fontsize=8, title_fontsize=9)

    fig.suptitle(f"Task: {sample['task_id']} — Accuracy: {acc:.1%}",
                 fontsize=14, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✔ Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close(fig)

    return acc


# ======== CLI ENTRY ===========
def parse_args():
    parser = argparse.ArgumentParser(description="Standalone ARC visualizer")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best.pt)')
    parser.add_argument('--data-dir', type=str, default='Data/ARC',
                        help='Path to ARC dataset root')
    parser.add_argument('--split', type=str, default='evaluation',
                        choices=['training', 'evaluation'],
                        help='ARC split to visualize')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--num-tasks', type=int, default=8,
                        help='Number of tasks to visualize')
    parser.add_argument('--task-ids', type=str, nargs='*', default=None,
                        help='Specific ARC task IDs to visualize (overrides num-tasks)')
    parser.add_argument('--output-dir', type=str, default='visualizations/checkpoint_view',
                        help='Directory to save generated figures')
    parser.add_argument('--max-train-pairs', type=int, default=2,
                        help='How many training pairs per task to use')
    parser.add_argument('--token-threshold', type=float, default=0.3,
                        help='Presence threshold when displaying tokens')
    parser.add_argument('--T', type=int, default=0,
                        help='S2 optimization steps (0=direct decoder, 20=default S2)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='S2 optimization step size')
    parser.add_argument('--sigma', type=float, default=0.0,
                        help='Langevin noise scale')
    parser.add_argument('--N', type=int, default=1,
                        help='Best-of-N: run inference N times and pick lowest energy')
    parser.add_argument('--no-backtracking', action='store_true',
                        help='Disable backtracking line search (use blind gradient descent)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)
    print("✓ Model loaded")

    dataset = ARCDataset(
        data_dir=args.data_dir,
        split=args.split,
        num_train_pairs=args.max_train_pairs,
        max_samples=None
    )

    # Determine which tasks to visualize
    if args.task_ids:
        indices = []
        id_set = set(args.task_ids)
        for idx, tf in enumerate(dataset.task_files):
            if tf.stem in id_set:
                indices.append(idx)
        if not indices:
            raise ValueError(f"Could not find any tasks matching IDs: {args.task_ids}")
    else:
        total = len(dataset)
        indices = list(range(min(args.num_tasks, total)))

    print(f"Visualizing {len(indices)} tasks from split '{args.split}'")

    accuracies = []
    for count, idx in enumerate(indices, start=1):
        sample = dataset[idx]
        save_path = Path(args.output_dir) / f"{count:03d}_{sample['task_id']}.png"
        print(f"\n[{count}/{len(indices)}] Task: {sample['task_id']}")
        acc = visualize_example(model, sample, device=device, args=args, save_path=save_path, 
                               T=args.T, alpha=args.alpha, use_backtracking=not args.no_backtracking)
        if acc is not None:
            accuracies.append(acc)
            print(f"    Accuracy: {acc:.1%}")

    if accuracies:
        print("\n" + "=" * 60)
        print(f"Visualized {len(accuracies)} tasks")
        print(f"Average pixel accuracy: {np.mean(accuracies):.2%}")
        print(f"Figures saved to: {args.output_dir}")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

