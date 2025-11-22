
import torch
import matplotlib.pyplot as plt
import numpy as np
from models import EBTSystem
from dataloader_arc import ARCDataset

def inspect_predictions():
    # Load model
    model = EBTSystem()
    checkpoint = torch.load('checkpoints_ebt_s2/best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load dataset
    dataset = ARCDataset(data_dir='Data/ARC', split='evaluation')
    
    # Find task 03560426
    task_idx = -1
    for i in range(len(dataset)):
        if dataset[i]['task_id'] == '03560426':
            task_idx = i
            break
            
    if task_idx == -1:
        print("Task not found")
        return

    sample = dataset[task_idx]
    print(f"Inspecting Task: {sample['task_id']}")
    
    x1, y1 = sample['train_pairs'][0]
    x2, y2 = sample['train_pairs'][1]
    x_test, y_gt = sample['test_pair']
    
    # Add batch dim
    x1, y1 = x1.unsqueeze(0), y1.unsqueeze(0)
    x2, y2 = x2.unsqueeze(0), y2.unsqueeze(0)
    x_test = x_test.unsqueeze(0)
    
    # Run S1
    with torch.no_grad():
        r1, r2, r_star, rbar, y_decoder_output = model(x1, y1, x2, y2, x_test)
        pred_s1 = y_decoder_output.argmax(dim=-1).squeeze(0)
    
    # Run S2 (approximate)
    y = y_decoder_output.clone()
    h_x = model.encode_input_shared(x_test)
    # One gradient step to see direction
    y.requires_grad_(True)
    E = model.energy.energy_with_shared_input(h_x, rbar, y, canonical=False)
    grad_y, = torch.autograd.grad(E.mean(), y)
    y_s2 = (y - 0.05 * grad_y).detach() # Single step S2
    pred_s2 = y_s2.argmax(dim=-1).squeeze(0)

    print("\n=== S1 Prediction Stats ===")
    vals, counts = pred_s1.unique(return_counts=True)
    for v, c in zip(vals, counts):
        print(f"Color {v.item()}: {c.item()} pixels")
        
    print("\n=== S2 Prediction Stats (1 step) ===")
    vals, counts = pred_s2.unique(return_counts=True)
    for v, c in zip(vals, counts):
        print(f"Color {v.item()}: {c.item()} pixels")

    print("\n=== Ground Truth Stats ===")
    vals, counts = y_gt.unique(return_counts=True)
    for v, c in zip(vals, counts):
        print(f"Color {v.item()}: {c.item()} pixels")

if __name__ == "__main__":
    inspect_predictions()

