"""
Visualize Lean EBT model predictions on ARC tasks.
Shows training examples, test input, prediction, and ground truth.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import argparse
import os

from lean_model import LeanEBTSystem
from dataloader_arc import F8a8fe49ARCDataset, collate_fn
from torch.utils.data import DataLoader

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


def load_lean_model(checkpoint_path, device='cpu'):
    """Load trained Lean EBT model from checkpoint."""
    model = LeanEBTSystem().to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded model from {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found at {checkpoint_path}, using untrained model")
    model.eval()
    return model


def visualize_task(model, batch_dict, device='cpu', save_path=None):
    """Visualize a single task: inputs, prediction, ground truth."""
    # Prepare data
    pairs = batch_dict['train_pairs']
    test_pair = batch_dict['test_pair']
    
    x1 = pairs[0][0].unsqueeze(0).to(device)
    y1 = pairs[0][1].unsqueeze(0).to(device)
    x2 = pairs[1][0].unsqueeze(0).to(device)
    y2 = pairs[1][1].unsqueeze(0).to(device)
    x_test = test_pair[0].unsqueeze(0).to(device)
    y_star = test_pair[1].unsqueeze(0).to(device)
    
    # Check if sizes match
    if x_test.shape[1:] != y_star.shape[1:]:
        print(f"⚠ Skipping task: size mismatch (input: {x_test.shape[1:]}, output: {y_star.shape[1:]})")
        return None
    
    # Prepare training pairs
    x_train_list = [x1, x2]
    y_train_list = [y1, y2]
    target_shape = (y_star.shape[1], y_star.shape[2])
    
    # Run inference
    with torch.no_grad():
        logits = model(x_test, x_train_list, y_train_list, target_shape=target_shape)
        pred = logits.argmax(dim=-1)  # (B, H, W)
    
    # Convert to numpy
    x1_np = x1[0].cpu().numpy()
    y1_np = y1[0].cpu().numpy()
    x2_np = x2[0].cpu().numpy()
    y2_np = y2[0].cpu().numpy()
    x_test_np = x_test[0].cpu().numpy()
    pred_np = pred[0].cpu().numpy()
    gt_np = y_star[0].cpu().numpy()
    
    # Compute accuracy
    correct = (pred_np == gt_np).sum()
    total = pred_np.size
    accuracy = correct / total
    exact_match = (pred_np == gt_np).all()
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Training examples
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(x1_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax1.set_title('Train Input 1', fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax1.set_xticks(np.arange(-0.5, x1_np.shape[1], 1))
    ax1.set_yticks(np.arange(-0.5, x1_np.shape[0], 1))
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(y1_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax2.set_title('Train Output 1', fontsize=12, fontweight='bold')
    ax2.axis('off')
    ax2.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax2.set_xticks(np.arange(-0.5, y1_np.shape[1], 1))
    ax2.set_yticks(np.arange(-0.5, y1_np.shape[0], 1))
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(x2_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax3.set_title('Train Input 2', fontsize=12, fontweight='bold')
    ax3.axis('off')
    ax3.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax3.set_xticks(np.arange(-0.5, x2_np.shape[1], 1))
    ax3.set_yticks(np.arange(-0.5, x2_np.shape[0], 1))
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(y2_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax4.set_title('Train Output 2', fontsize=12, fontweight='bold')
    ax4.axis('off')
    ax4.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax4.set_xticks(np.arange(-0.5, y2_np.shape[1], 1))
    ax4.set_yticks(np.arange(-0.5, y2_np.shape[0], 1))
    
    # Row 2: Test input, prediction, ground truth, difference
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(x_test_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax5.set_title('Test Input', fontsize=12, fontweight='bold')
    ax5.axis('off')
    ax5.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax5.set_xticks(np.arange(-0.5, x_test_np.shape[1], 1))
    ax5.set_yticks(np.arange(-0.5, x_test_np.shape[0], 1))
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(pred_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    match_status = "✓ EXACT MATCH" if exact_match else ""
    ax6.set_title(f'Model Prediction\n(Acc: {accuracy:.1%}) {match_status}', 
                  fontsize=12, fontweight='bold',
                  color='green' if exact_match else 'black')
    ax6.axis('off')
    ax6.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax6.set_xticks(np.arange(-0.5, pred_np.shape[1], 1))
    ax6.set_yticks(np.arange(-0.5, pred_np.shape[0], 1))
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(gt_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')
    ax7.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax7.axis('off')
    ax7.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax7.set_xticks(np.arange(-0.5, gt_np.shape[1], 1))
    ax7.set_yticks(np.arange(-0.5, gt_np.shape[0], 1))
    
    # Difference map
    ax8 = fig.add_subplot(gs[1, 3])
    diff = (pred_np != gt_np).astype(float)
    ax8.imshow(diff, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
    ax8.set_title(f'Errors ({diff.sum():.0f} pixels)', fontsize=12, fontweight='bold')
    ax8.axis('off')
    ax8.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax8.set_xticks(np.arange(-0.5, diff.shape[1], 1))
    ax8.set_yticks(np.arange(-0.5, diff.shape[0], 1))
    
    # Add color legend
    legend_elements = [mpatches.Patch(facecolor=ARC_COLORS[i], label=str(i)) 
                      for i in range(10)]
    fig.legend(handles=legend_elements, loc='upper right', ncol=1, 
              title='ARC Colors', fontsize=9, title_fontsize=10)
    
    plt.suptitle(f'Lean EBT Model Prediction - Accuracy: {accuracy:.1%} | Exact Match: {exact_match}', 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return accuracy, exact_match


def main():
    parser = argparse.ArgumentParser(description='Visualize Lean EBT predictions')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_lean/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to visualize')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='visualizations/lean',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_lean_model(args.checkpoint, device=args.device)
    
    # Load dataset
    val_dataset = F8a8fe49ARCDataset(
        task_id='f8a8fe49', split='val',
        num_problems_per_epoch=100, num_train_pairs=2,
        rearc_path='re-arc/re_arc/tasks'
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Visualize examples
    accuracies = []
    exact_matches = 0
    
    print(f"\nVisualizing {args.num_examples} examples...")
    for i, batch_list in enumerate(val_loader):
        if i >= args.num_examples:
            break
            
        batch_dict = batch_list[0]
        
        # Skip if size mismatch
        test_pair = batch_dict['test_pair']
        if test_pair[0].shape != test_pair[1].shape:
            continue
        
        save_path = os.path.join(args.save_dir, f'example_{i+1}.png')
        acc, exact = visualize_task(model, batch_dict, device=args.device, save_path=save_path)
        
        if acc is not None:
            accuracies.append(acc)
            if exact:
                exact_matches += 1
            print(f"Example {i+1}: Accuracy = {acc:.1%}, Exact Match = {exact}")
    
    if accuracies:
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Average Accuracy: {np.mean(accuracies):.1%}")
        print(f"  Exact Matches: {exact_matches}/{len(accuracies)}")
        print(f"  Visualizations saved to: {args.save_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()

