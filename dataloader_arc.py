"""
Simple ARC dataloader for training with the tokenizer-based encoder.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import os
from pathlib import Path


class ARCDataset(Dataset):
    """
    Simple ARC dataset loader that loads tasks from the Data/ARC directory.
    """
    
    def __init__(self, 
                 data_dir='Data/ARC',
                 split='training',
                 num_train_pairs=2,
                 max_samples=None):
        """
        Args:
            data_dir: Path to ARC data directory
            split: 'training' or 'evaluation'
            num_train_pairs: Number of train pairs to use per task
            max_samples: Maximum number of samples to load (None for all)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_train_pairs = num_train_pairs
        
        # Load all task files
        task_dir = self.data_dir / split
        self.task_files = sorted(list(task_dir.glob('*.json')))
        
        if max_samples:
            self.task_files = self.task_files[:max_samples]
        
        self.tasks = []
        for task_file in self.task_files:
            with open(task_file, 'r') as f:
                task = json.load(f)
                self.tasks.append(task)
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Get training examples
        train_examples = task['train']
        test_examples = task['test']
        
        # Select training pairs
        if len(train_examples) >= self.num_train_pairs:
            selected_train = random.sample(train_examples, self.num_train_pairs)
        else:
            # If not enough examples, repeat some
            selected_train = train_examples
            while len(selected_train) < self.num_train_pairs:
                selected_train.append(random.choice(train_examples))
        
        # Select a test example (usually just one in ARC)
        test_example = test_examples[0]
        
        # Convert to tensors
        train_pairs = []
        for ex in selected_train:
            x = torch.tensor(ex['input'], dtype=torch.long)
            y = torch.tensor(ex['output'], dtype=torch.long)
            train_pairs.append((x, y))
        
        x_test = torch.tensor(test_example['input'], dtype=torch.long)
        
        # For training, use output if available, otherwise use input as target
        if 'output' in test_example and len(test_example['output']) > 0:
            y_test = torch.tensor(test_example['output'], dtype=torch.long)
        else:
            # During evaluation, we don't have the output
            y_test = x_test.clone()  # Placeholder
        
        return {
            'train_pairs': train_pairs,
            'test_pair': (x_test, y_test),
            'task_id': self.task_files[idx].stem
        }


def collate_fn(batch):
    """
    Custom collate function for ARC dataset.
    Since ARC grids have variable sizes, we keep them as lists.
    """
    return batch


class SimpleARCDataset(Dataset):
    """
    Simplified ARC dataset that returns fixed-size examples for initial testing.
    """
    
    def __init__(self, num_samples=100, grid_size=(8, 8), num_colors=10):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.num_colors = num_colors
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        H, W = self.grid_size
        
        # Generate random grids for testing
        # Train pairs: simple transformations
        x1 = torch.randint(0, self.num_colors, (H, W))
        y1 = x1.clone()
        # Simple transformation: shift colors
        y1 = (y1 + 1) % self.num_colors
        
        x2 = torch.randint(0, self.num_colors, (H, W))
        y2 = x2.clone()
        y2 = (y2 + 1) % self.num_colors
        
        # Test pair
        x_test = torch.randint(0, self.num_colors, (H, W))
        y_test = (x_test + 1) % self.num_colors
        
        return {
            'train_pairs': [(x1, y1), (x2, y2)],
            'test_pair': (x_test, y_test),
            'task_id': f'synthetic_{idx}'
        }


# For compatibility with existing training script
class F8a8fe49ARCDataset(ARCDataset):
    """
    Compatibility wrapper for the specific task dataset used in training script.
    This loads general ARC tasks instead of a specific task.
    """
    
    def __init__(self, task_id='', split='train', 
                 num_problems_per_epoch=500, num_train_pairs=2,
                 rearc_path=None):
        # Map split names
        if split == 'train':
            arc_split = 'training'
        elif split == 'val':
            arc_split = 'evaluation'
        else:
            arc_split = split
        
        # Initialize with ARC data
        super().__init__(
            data_dir='Data/ARC',
            split=arc_split,
            num_train_pairs=num_train_pairs,
            max_samples=num_problems_per_epoch
        )


if __name__ == '__main__':
    # Test the dataloader
    dataset = ARCDataset(data_dir='Data/ARC', split='training', max_samples=5)
    print(f"Loaded {len(dataset)} tasks")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Task ID: {sample['task_id']}")
    print(f"Number of train pairs: {len(sample['train_pairs'])}")
    print(f"Train pair 1 input shape: {sample['train_pairs'][0][0].shape}")
    print(f"Train pair 1 output shape: {sample['train_pairs'][0][1].shape}")
    print(f"Test input shape: {sample['test_pair'][0].shape}")
    print(f"Test output shape: {sample['test_pair'][1].shape}")
