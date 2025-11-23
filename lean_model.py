import torch
import torch.nn as nn
from canvas_vit import CanvasViTEncoder
from residual_refiner import ResidualRuleRefiner
from patch_decoder import PatchTransformerDecoder

class LeanEBTSystem(nn.Module):
    """
    The New Synthesis: "Fat-Cutting" Architecture.
    
    Components:
    1. Canvas ViT Encoder (32x32)
    2. Residual Rule Refiner (Explicit Rule + Latent Refinement + Task Embeddings)
    3. Patch Transformer Decoder
    
    Flow:
    1. Encoder: Grids -> 32x32 Canvas Features
    2. Refiner: Support Pairs -> Rule Vector (r) + z_task correction
    3. Fusion: Input Features + Rule Vector
    4. Decoder: Predicts output grid pixels
    """
    def __init__(self, config=None):
        super().__init__()
        
        # Default config if none provided
        self.d_model = getattr(config, 'D_FEAT', 512)
        self.num_colors = getattr(config, 'NUM_COLORS', 10)
        # Number of tasks for offline training embedding table
        self.num_train_tasks = getattr(config, 'NUM_TRAIN_TASKS', 400)
        
        # 1. Unified Encoder (Canvas 32x32)
        self.encoder = CanvasViTEncoder(
            num_colors=self.num_colors,
            d_model=self.d_model,
            n_head=8,
            n_layers=6,
            canvas_size=32
        )
        
        # 2. Residual Rule Refiner
        self.refiner = ResidualRuleRefiner(
            rule_dim=self.d_model,
            latent_dim=64,
            num_train_tasks=self.num_train_tasks
        )
        
        # 3. Decoder
        self.decoder = PatchTransformerDecoder(
            num_colors=self.num_colors,
            d_model=self.d_model,
            n_head=8,
            n_layers=6,
            max_len=1024 # 32x32
        )
        
    def forward(self, x_test, x_train_list, y_train_list, z_task=None, task_ids=None, target_shape=None):
        """
        Args:
            x_test: (B, H_test, W_test) Test Input
            x_train_list: List of K (B, H_in, W_in) training inputs
            y_train_list: List of K (B, H_out, W_out) training outputs
            z_task: (B, latent_dim) Optional latent for TTT
            task_ids: (B,) Optional integer IDs for offline training
            target_shape: (H, W) desired output size.
        
        Returns:
            logits: (B, H_out, W_out, Num_Colors)
        """
        # 1. Encode Test Input
        # (B, 1024, D)
        x_test_enc = self.encoder(x_test)
        
        # 2. Encode Training Pairs
        examples = []
        for x_ex, y_ex in zip(x_train_list, y_train_list):
            x_ex_enc = self.encoder(x_ex)
            y_ex_enc = self.encoder(y_ex)
            examples.append((x_ex_enc, y_ex_enc))
            
        # 3. Induce Rule (Fast + Slow)
        # r_final: (B, D)
        # Pass both z_task and task_ids
        r_final = self.refiner(examples, x_test_enc, z_task=z_task, task_ids=task_ids)
        
        # 4. Fuse Rule with Input
        # Simple addition/broadcasting: x_test + rule
        # (B, 1024, D) + (B, 1, D)
        refined_memory = x_test_enc + r_final.unsqueeze(1)
        
        # 5. Decode
        if target_shape is None:
            target_shape = (32, 32)
            
        logits = self.decoder(refined_memory, output_shape=target_shape)
        
        return logits

if __name__ == "__main__":
    # Integration Test
    model = LeanEBTSystem()
    
    # Create dummy data (B=1)
    # Test input: 10x10 (will be padded to 32x32)
    x_test = torch.randint(0, 10, (1, 10, 10))
    
    # 2 Training examples
    x_train = [
        torch.randint(0, 10, (1, 10, 10)),
        torch.randint(0, 10, (1, 15, 15))
    ]
    y_train = [
        torch.randint(0, 10, (1, 10, 10)),
        torch.randint(0, 10, (1, 15, 15))
    ]
    
    # Forward pass with task_ids
    task_ids = torch.tensor([0])
    logits = model(x_test, x_train, y_train, task_ids=task_ids)
    
    print(f"Input Test: {x_test.shape}")
    print(f"Output Logits: {logits.shape}")
    
    assert logits.shape == (1, 32, 32, 10)
    
    # Test with specific target shape
    logits_small = model(x_test, x_train, y_train, task_ids=task_ids, target_shape=(10, 10))
    print(f"Output Logits (10x10): {logits_small.shape}")
    assert logits_small.shape == (1, 10, 10, 10)
    
    print("Lean EBT System Integration Test passed!")
