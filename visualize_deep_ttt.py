"""
Visualize Deep TTT (Test-Time Training) results.
Updates model weights + task embeddings on augmented data at test time.
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
from deep_ttt import DeepTTTEngine, DeepTTTConfig

# ARC color palette
ARC_COLORS = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
]
cmap = ListedColormap(ARC_COLORS)

def load_lean_model(checkpoint_path, device='cpu'):
    model = LeanEBTSystem().to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded model from {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found at {checkpoint_path}, using untrained model")
    model.eval()
    return model

def visualize_task(model, batch_dict, device='cpu', save_path=None, config=None):
    """Visualize a single task with Deep TTT."""
    pairs = batch_dict['train_pairs']
    test_pair = batch_dict['test_pair']
    
    x1 = pairs[0][0].unsqueeze(0).to(device)
    y1 = pairs[0][1].unsqueeze(0).to(device)
    x2 = pairs[1][0].unsqueeze(0).to(device)
    y2 = pairs[1][1].unsqueeze(0).to(device)
    x_test = test_pair[0].unsqueeze(0).to(device)
    y_star = test_pair[1].unsqueeze(0).to(device)
    
    if x_test.shape[1:] != y_star.shape[1:]:
        print(f"⚠ Skipping task: size mismatch")
        return None

    x_train_list = [x1, x2]
    y_train_list = [y1, y2]
    target_shape = (y_star.shape[1], y_star.shape[2])
    
    # Initialize Deep TTT Engine
    engine = DeepTTTEngine(model, config, device=device)
    
    # Run Deep TTT Inference
    # This handles the full optimization loop and state restoration
    pred, loss = engine.run_inference(x_train_list, y_train_list, x_test, target_shape)
    
    # Convert to numpy for plotting
    x1_np = x1[0].cpu().numpy()
    y1_np = y1[0].cpu().numpy()
    x_test_np = x_test[0].cpu().numpy()
    pred_np = pred[0].cpu().numpy()
    gt_np = y_star[0].cpu().numpy()
    
    correct = (pred_np == gt_np).sum()
    accuracy = correct / pred_np.size
    exact_match = (pred_np == gt_np).all()
    
    # Visualization (Same as before)
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 4, wspace=0.3)
    
    # Plot simplified view
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(x1_np, cmap=cmap, vmin=0, vmax=9)
    ax1.set_title('Train Input 1')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(y1_np, cmap=cmap, vmin=0, vmax=9)
    ax2.set_title('Train Output 1')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred_np, cmap=cmap, vmin=0, vmax=9)
    ax3.set_title(f'Deep TTT Prediction\nAcc: {accuracy:.1%}')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(gt_np, cmap=cmap, vmin=0, vmax=9)
    ax4.set_title('Ground Truth')
    ax4.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        
    plt.close()
    return accuracy, exact_match

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints_lean/best.pt')
    parser.add_argument('--num_examples', type=int, default=5)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--lr_weights', type=float, default=1e-4)
    parser.add_argument('--lr_z', type=float, default=0.05)
    parser.add_argument('--augmentations', type=int, default=50)
    parser.add_argument('--save_dir', default='visualizations/deep_ttt')
    args = parser.parse_args()
    
    device = 'cpu' # Force CPU for safety/compatibility
    
    config = DeepTTTConfig(
        steps=args.steps,
        lr_weights=args.lr_weights,
        lr_z=args.lr_z,
        num_augmentations=args.augmentations,
        update_refiner=True,
        update_encoder=False,
        update_decoder=False,
        use_per_aug_z=True
    )
    
    model = load_lean_model(args.checkpoint, device)
    
    # Load dataset
    val_dataset = F8a8fe49ARCDataset(
        task_id='f8a8fe49', split='val',
        num_problems_per_epoch=100, num_train_pairs=2,
        rearc_path='re-arc/re_arc/tasks'
    )
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Same deterministic seed
    torch.manual_seed(42)
    np.random.seed(42)
    indices = list(range(args.num_examples))
    
    accuracies = []
    
    for i in indices:
        batch_dict = val_dataset[i]
        batch_dict['train_pairs'] = [ (p[0].unsqueeze(0), p[1].unsqueeze(0)) for p in batch_dict['train_pairs'] ]
        batch_dict['test_pair'] = (batch_dict['test_pair'][0].unsqueeze(0), batch_dict['test_pair'][1].unsqueeze(0))
        
        # Fix dimension hack for visualization function which expects batch-dim already
        # Actually my visualize_task adds batch dim again?
        # My visualize_task above expects already batched tensors in the list
        # But F8a8fe49ARCDataset returns tensors without batch dim.
        # Let's fix manually.
        
        # Clean batch_dict construction
        raw_sample = val_dataset[i]
        # ARCDataset returns (H, W) tensors. visualize_task expects this and adds batch dim.
        clean_batch = raw_sample
        
        save_path = os.path.join(args.save_dir, f'example_{i+1}.png')
        res = visualize_task(model, clean_batch, device, save_path, config)
        if res is None:
            continue
            
        acc, exact = res
        
        if acc is not None:
            accuracies.append(acc)
            print(f"Example {i+1}: Acc={acc:.1%}")
            
    print(f"Average Deep TTT Accuracy: {np.mean(accuracies):.1%}")

if __name__ == '__main__':
    main()

