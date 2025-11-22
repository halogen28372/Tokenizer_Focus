#!/usr/bin/env python3
"""
Simple test script to verify the tokenizer-based training pipeline works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Import our modules
from config import config
from models import EBTSystem, GridEncoder
from tokenizer_encoder import TokenizerEncoder, HybridGridEncoder
from dataloader_arc import SimpleARCDataset, ARCDataset, collate_fn
from torch.utils.data import DataLoader
from arc_lago_tokenizer import ArcLagoTokenizer


def test_tokenizer():
    """Test that the tokenizer works on a sample grid."""
    print("="*60)
    print("Testing ARC/LAGO Tokenizer")
    print("="*60)
    
    # Create a simple test grid
    grid = [
        [0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,0],
        [0,1,0,0,0,0,1,0],
        [0,1,0,2,2,0,1,0],
        [0,1,0,2,2,0,1,0],
        [0,1,0,0,0,0,1,0],
        [0,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0],
    ]
    
    tokenizer = ArcLagoTokenizer()
    result = tokenizer.tokenize(grid)
    
    print(f"Number of shapes: {len(result['shapes'])}")
    print(f"Number of relations: {len(result['relations'])}")
    print(f"Number of meta tokens: {len(result['meta'])}")
    
    for shape in result['shapes'][:3]:  # Show first 3 shapes
        print(f"  - {shape['type']} (color={shape['color']}, layer={shape['layer']})")
    
    print("✓ Tokenizer test passed\n")
    return True


def test_tokenizer_encoder():
    """Test the tokenizer-based encoder."""
    print("="*60)
    print("Testing Tokenizer Encoder")
    print("="*60)
    
    encoder = TokenizerEncoder(num_colors=10, d_feat=512)
    
    # Create a batch of test grids
    B, H, W = 2, 8, 8
    x = torch.randint(0, 10, (B, H, W))
    
    # Forward pass
    features = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected shape: ({B}, 512)")
    
    assert features.shape == (B, 512), f"Shape mismatch: {features.shape} != ({B}, 512)"
    print("✓ Tokenizer encoder test passed\n")
    return True


def test_hybrid_encoder():
    """Test the hybrid encoder that combines tokenizer and CNN."""
    print("="*60)
    print("Testing Hybrid Encoder")
    print("="*60)
    
    encoder = HybridGridEncoder(num_colors=10, d_feat=512, use_cnn=True, use_tokenizer=True)
    
    # Create a batch of test grids
    B, H, W = 2, 8, 8
    x = torch.randint(0, 10, (B, H, W))
    
    # Forward pass
    features = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected shape: ({B}, 512)")
    
    assert features.shape == (B, 512), f"Shape mismatch: {features.shape} != ({B}, 512)"
    print("✓ Hybrid encoder test passed\n")
    return True


def test_grid_encoder_with_tokenizer():
    """Test the modified GridEncoder with tokenizer support."""
    print("="*60)
    print("Testing GridEncoder with Tokenizer")
    print("="*60)
    
    # Force tokenizer mode
    config.USE_TOKENIZER = True
    config.USE_NEIGHBOR_ENCODER = False
    
    encoder = GridEncoder(num_colors=10, d_feat=512, use_tokenizer=True)
    
    # Create a batch of test grids
    B, H, W = 2, 8, 8
    x = torch.randint(0, 10, (B, H, W))
    
    # Forward pass
    features = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected shape: ({B}, 512)")
    
    assert features.shape == (B, 512), f"Shape mismatch: {features.shape} != ({B}, 512)"
    print("✓ GridEncoder with tokenizer test passed\n")
    return True


def test_ebt_system():
    """Test the complete EBT system with tokenizer."""
    print("="*60)
    print("Testing Complete EBT System")
    print("="*60)
    
    # Configure for tokenizer
    config.USE_TOKENIZER = True
    config.USE_NEIGHBOR_ENCODER = False
    config.USE_NEIGHBOR_SOLVER = False
    
    model = EBTSystem()
    
    # Create test data
    B, H, W = 1, 8, 8
    x1 = torch.randint(0, 10, (B, H, W))
    y1 = torch.randint(0, 10, (B, H, W))
    x2 = torch.randint(0, 10, (B, H, W))
    y2 = torch.randint(0, 10, (B, H, W))
    x_test = torch.randint(0, 10, (B, H, W))
    
    # Forward pass
    r1, r2, r_star, rbar, y_logits = model(x1, y1, x2, y2, x_test)
    
    print(f"Rule 1 shape: {r1.shape}")
    print(f"Rule 2 shape: {r2.shape}")
    print(f"Rule star shape: {r_star.shape}")
    print(f"Aggregated rule shape: {rbar.shape}")
    print(f"Output logits shape: {y_logits.shape}")
    print(f"Expected output shape: ({B}, {H}, {W}, 10)")
    
    assert y_logits.shape == (B, H, W, 10), f"Output shape mismatch"
    print("✓ EBT system test passed\n")
    return True


def test_dataloader():
    """Test the ARC dataloader."""
    print("="*60)
    print("Testing ARC DataLoader")
    print("="*60)
    
    # Check if ARC data exists
    data_path = Path('Data/ARC/training')
    if not data_path.exists():
        print("⚠ ARC data not found. Using synthetic dataset for testing.")
        dataset = SimpleARCDataset(num_samples=5, grid_size=(8, 8))
    else:
        print("✓ Found ARC data directory")
        dataset = ARCDataset(data_dir='Data/ARC', split='training', max_samples=2)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Train pairs: {len(sample['train_pairs'])}")
    print(f"Train pair 1 input shape: {sample['train_pairs'][0][0].shape}")
    print(f"Test input shape: {sample['test_pair'][0].shape}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    for i, batch in enumerate(loader):
        if i == 0:
            batch_dict = batch[0]
            print(f"Batch loaded successfully")
            break
    
    print("✓ DataLoader test passed\n")
    return True


def test_training_step():
    """Test a single training step."""
    print("="*60)
    print("Testing Training Step")
    print("="*60)
    
    # Configure
    config.USE_TOKENIZER = True
    config.USE_NEIGHBOR_ENCODER = False
    config.USE_NEIGHBOR_SOLVER = False
    
    model = EBTSystem()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Use synthetic data for testing
    dataset = SimpleARCDataset(num_samples=1, grid_size=(8, 8))
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    # Get one batch
    batch_list = next(iter(loader))
    batch_dict = batch_list[0]
    
    # Prepare data
    device = 'cpu'
    x1 = batch_dict['train_pairs'][0][0].unsqueeze(0).to(device)
    y1 = batch_dict['train_pairs'][0][1].unsqueeze(0).to(device)
    x2 = batch_dict['train_pairs'][1][0].unsqueeze(0).to(device)
    y2 = batch_dict['train_pairs'][1][1].unsqueeze(0).to(device)
    x_test = batch_dict['test_pair'][0].unsqueeze(0).to(device)
    y_test = batch_dict['test_pair'][1].unsqueeze(0).to(device)
    
    # Forward pass
    r1, r2, r_star, rbar, y_logits = model(x1, y1, x2, y2, x_test)
    
    # Compute simple loss
    B, H, W = y_test.shape
    C = 10
    loss = F.cross_entropy(
        y_logits.reshape(B * H * W, C),
        y_test.reshape(B * H * W).long()
    )
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check gradients
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item()
    
    print(f"Total gradient norm: {total_grad_norm:.4f}")
    print("✓ Training step test passed\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TOKENIZER INTEGRATION TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Tokenizer", test_tokenizer),
        ("Tokenizer Encoder", test_tokenizer_encoder),
        ("Hybrid Encoder", test_hybrid_encoder),
        ("GridEncoder with Tokenizer", test_grid_encoder_with_tokenizer),
        ("EBT System", test_ebt_system),
        ("DataLoader", test_dataloader),
        ("Training Step", test_training_step),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} test failed")
        except Exception as e:
            failed += 1
            print(f"✗ {name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The tokenizer integration is working.")
        print("\nYou can now run the training with:")
        print("  python train_ebt_s2.py --config config --device cpu")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
