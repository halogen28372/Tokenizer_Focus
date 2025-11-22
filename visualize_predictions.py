"""
Visualize model predictions vs ground truth on ARC tasks.
Shows input grids, predicted output, ground truth, and generated tokens.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

from models import EBTSystem
from dataloader_arc import F8a8fe49ARCDataset, collate_fn
from config import config

# ARC color palette
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


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    model = EBTSystem().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def run_s2_inference(model, x1, y1, x2, y2, x_test, device='cpu', T=20, alpha=0.05):
    """Run S2-EBT inference (multi-step optimization)."""
    # Encode rule
    with torch.no_grad():
        r1, r2, r_star, rbar, _ = model(x1, y1, x2, y2, x_test)
        h_x = model.encode_input_shared(x_test)  # (B, d_feat)

    # Initialize output
    B, H, W = x_test.shape
    C = 10
    y = torch.randn(B, H, W, C, device=device, requires_grad=True)

    # Multi-step optimization
    for t in range(T):
        E = model.energy.energy_with_shared_input(h_x, rbar, y, canonical=False)
        grad_y, = torch.autograd.grad(E.mean(), y, create_graph=False)

        with torch.no_grad():
            y = y - alpha * grad_y
            y = y.detach()
            if t < T - 1:
                y.requires_grad_(True)
    
    # Get final prediction
    with torch.no_grad():
        pred_grid = y.argmax(dim=-1)
    
    return pred_grid, rbar


def extract_tokens_from_decoder(model, x_test, rbar):
    """Extract token predictions from the decoder."""
    if not hasattr(model.decoder, 'export_token_lists'):
        return None
    
    with torch.no_grad():
        # Run decoder to get tokens
        _ = model.decoder(x_test, rbar)
        tokens_batch = model.decoder.export_token_lists(threshold=0.3)
        
    return tokens_batch[0] if tokens_batch else []


def visualize_task(model, batch_dict, device='cpu', save_path=None):
    """Visualize a single task: inputs, prediction, ground truth, and tokens."""
    # Prepare data
    x1 = batch_dict['train_pairs'][0][0].unsqueeze(0).to(device)
    y1 = batch_dict['train_pairs'][0][1].unsqueeze(0).to(device)
    x2 = batch_dict['train_pairs'][1][0].unsqueeze(0).to(device)
    y2 = batch_dict['train_pairs'][1][1].unsqueeze(0).to(device)
    x_test = batch_dict['test_pair'][0].unsqueeze(0).to(device)
    y_star = batch_dict['test_pair'][1].unsqueeze(0).to(device)
    
    # Check if sizes match
    if x_test.shape[1:] != y_star.shape[1:]:
        print(f"⚠ Skipping task: size mismatch (input: {x_test.shape[1:]}, output: {y_star.shape[1:]})")
        return None
    
    # Run inference
    pred_grid, rbar = run_s2_inference(model, x1, y1, x2, y2, x_test, device, T=20, alpha=0.05)
    
    # Extract tokens
    tokens = extract_tokens_from_decoder(model, x_test, rbar)
    
    # Convert to numpy
    x1_np = x1[0].cpu().numpy()
    y1_np = y1[0].cpu().numpy()
    x2_np = x2[0].cpu().numpy()
    y2_np = y2[0].cpu().numpy()
    x_test_np = x_test[0].cpu().numpy()
    pred_np = pred_grid[0].cpu().numpy()
    gt_np = y_star[0].cpu().numpy()
    
    # Compute accuracy
    correct = (pred_np == gt_np).sum()
    total = pred_np.size
    accuracy = correct / total
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)
    
    # Row 1: Training examples
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(x1_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax1.set_title('Train Input 1', fontsize=10, fontweight='bold')
    ax1.axis('off')
    ax1.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax1.set_xticks(np.arange(-0.5, x1_np.shape[1], 1))
    ax1.set_yticks(np.arange(-0.5, x1_np.shape[0], 1))
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(y1_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax2.set_title('Train Output 1', fontsize=10, fontweight='bold')
    ax2.axis('off')
    ax2.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax2.set_xticks(np.arange(-0.5, y1_np.shape[1], 1))
    ax2.set_yticks(np.arange(-0.5, y1_np.shape[0], 1))
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(x2_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax3.set_title('Train Input 2', fontsize=10, fontweight='bold')
    ax3.axis('off')
    ax3.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax3.set_xticks(np.arange(-0.5, x2_np.shape[1], 1))
    ax3.set_yticks(np.arange(-0.5, x2_np.shape[0], 1))
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(y2_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax4.set_title('Train Output 2', fontsize=10, fontweight='bold')
    ax4.axis('off')
    ax4.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax4.set_xticks(np.arange(-0.5, y2_np.shape[1], 1))
    ax4.set_yticks(np.arange(-0.5, y2_np.shape[0], 1))
    
    # Row 2: Test input, prediction, ground truth
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(x_test_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax5.set_title('Test Input', fontsize=10, fontweight='bold')
    ax5.axis('off')
    ax5.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax5.set_xticks(np.arange(-0.5, x_test_np.shape[1], 1))
    ax5.set_yticks(np.arange(-0.5, x_test_np.shape[0], 1))
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(pred_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax6.set_title(f'Model Prediction\n(Acc: {accuracy:.1%})', fontsize=10, fontweight='bold')
    ax6.axis('off')
    ax6.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax6.set_xticks(np.arange(-0.5, pred_np.shape[1], 1))
    ax6.set_yticks(np.arange(-0.5, pred_np.shape[0], 1))
    
    ax7 = fig.add_subplot(gs[1, 3])
    ax7.imshow(gt_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax7.set_title('Ground Truth', fontsize=10, fontweight='bold')
    ax7.axis('off')
    ax7.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax7.set_xticks(np.arange(-0.5, gt_np.shape[1], 1))
    ax7.set_yticks(np.arange(-0.5, gt_np.shape[0], 1))
    
    # Row 3: Token information
    ax_tokens = fig.add_subplot(gs[2, :])
    ax_tokens.axis('off')
    
    if tokens:
        token_text = f"Generated Tokens ({len(tokens)} total):\n\n"
        for i, tok in enumerate(tokens[:8]):  # Show first 8 tokens
            token_text += f"Token {i+1}: {tok['type']}, "
            token_text += f"color={tok['color']}, "
            token_text += f"pos=({tok['x']:.1f}, {tok['y']:.1f}), "
            token_text += f"size=({tok['w']:.1f}×{tok['h']:.1f}), "
            token_text += f"presence={tok['presence']:.2f}\n"
        if len(tokens) > 8:
            token_text += f"\n... and {len(tokens) - 8} more tokens"
    else:
        token_text = "Token decoder not available or no tokens generated."
    
    ax_tokens.text(0.05, 0.5, token_text, fontsize=9, family='monospace',
                   verticalalignment='center', wrap=True)
    
    # Add color legend
    legend_elements = [mpatches.Patch(facecolor=ARC_COLORS[i], label=str(i)) 
                      for i in range(10)]
    fig.legend(handles=legend_elements, loc='upper right', ncol=1, 
              title='ARC Colors', fontsize=8, title_fontsize=9)
    
    plt.suptitle(f'ARC Task Visualization - Accuracy: {accuracy:.1%}', 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return accuracy


def main():
    """Visualize predictions on multiple tasks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize ARC predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_ebt_s2/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--num-tasks', type=int, default=5, help='Number of tasks to visualize')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                       help='Dataset split to use')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    print(f"Model loaded successfully!")
    print(f"Decoder type: {type(model.decoder).__name__}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = F8a8fe49ARCDataset(
        task_id='f8a8fe49',
        split=args.split,
        num_problems_per_epoch=args.num_tasks,
        num_train_pairs=2,
        rearc_path='re-arc/re_arc/tasks'
    )
    
    # Visualize tasks
    print(f"\nVisualizing {args.num_tasks} tasks...\n")
    accuracies = []
    
    for i in range(min(args.num_tasks, len(dataset))):
        print(f"Task {i+1}/{args.num_tasks}:")
        batch_dict = dataset[i]
        
        save_path = os.path.join(args.output_dir, f'task_{i+1:03d}.png')
        acc = visualize_task(model, batch_dict, device=args.device, save_path=save_path)
        
        if acc is not None:
            accuracies.append(acc)
            print(f"  ✓ Accuracy: {acc:.1%}\n")
    
    # Summary
    if accuracies:
        avg_acc = np.mean(accuracies)
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Visualized: {len(accuracies)} tasks")
        print(f"  Average Accuracy: {avg_acc:.1%}")
        print(f"  Saved to: {args.output_dir}/")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

