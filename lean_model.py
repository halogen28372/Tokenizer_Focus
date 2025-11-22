import torch
import torch.nn as nn
from object_vit import ObjectAugmentedViTEncoder
from rule_inductor import CrossAttentionRuleInductor
from patch_decoder import PatchTransformerDecoder

class LeanEBTSystem(nn.Module):
    """
    The Lean Energy-Based Transformer (EBT) Architecture.
    
    Replaces the complex multi-stream, renderer-based system with a pure Transformer stack.
    
    Flow:
    1. Encoder: Grids -> Object-Augmented Tokens
    2. Inductor: Test Input attends to Example Pairs (x, y-x) to retrieve rules
    3. Decoder: Predicts output grid pixels from Rule-Enriched Memory
    """
    def __init__(self, config=None):
        super().__init__()
        
        # Default config if none provided
        self.d_model = getattr(config, 'D_FEAT', 512)
        self.num_colors = getattr(config, 'NUM_COLORS', 10)
        
        # 1. Unified Encoder
        self.encoder = ObjectAugmentedViTEncoder(
            num_colors=self.num_colors,
            d_model=self.d_model,
            n_head=8,
            n_layers=6
        )
        
        # 2. Rule Inductor
        self.inductor = CrossAttentionRuleInductor(
            d_model=self.d_model,
            n_head=8
        )
        
        # 3. Decoder
        self.decoder = PatchTransformerDecoder(
            num_colors=self.num_colors,
            d_model=self.d_model,
            n_head=8,
            n_layers=6
        )
        
    def forward(self, x_test, x_train_list, y_train_list, target_shape=None):
        """
        Args:
            x_test: (B, H_test, W_test) Test Input
            x_train_list: List of K (B, H_in, W_in) training inputs
            y_train_list: List of K (B, H_out, W_out) training outputs
            target_shape: (H, W) desired output size. 
                          In inference, this is unknown! 
                          Ideally, we predict it or assume H_out = H_in (common heuristic).
        
        Returns:
            logits: (B, H_out, W_out, Num_Colors)
        """
        # 1. Encode Test Input
        # (B, L_test, D)
        x_test_enc = self.encoder(x_test)
        
        # 2. Encode Training Pairs
        # Note: We process them in a loop or batch.
        # Since sizes vary per example, we loop.
        x_ex_encs = []
        y_ex_encs = []
        
        for x_ex, y_ex in zip(x_train_list, y_train_list):
            x_ex_encs.append(self.encoder(x_ex))
            y_ex_encs.append(self.encoder(y_ex))
            
        # 3. Induce Rule Context
        # refined_memory: (B, L_test, D)
        # The test input tokens are now enriched with the transformation logic
        refined_memory = self.inductor(x_test_enc, x_ex_encs, y_ex_encs)
        
        # 4. Decode
        # If target_shape is unknown (inference), we typically default to input size
        # or use a separate small head to predict (H, W).
        # For this baseline, we assume H_out = H_in (Input Size) 
        # unless explicitly told otherwise (e.g. during training).
        if target_shape is None:
             # Default heuristic: Output size same as input size
             target_shape = (x_test.shape[1], x_test.shape[2])
             
        logits = self.decoder(refined_memory, output_shape=target_shape)
        
        return logits

if __name__ == "__main__":
    # Integration Test
    model = LeanEBTSystem()
    
    # Create dummy data (B=1)
    # Test input: 10x10
    x_test = torch.randint(0, 10, (1, 10, 10))
    
    # 2 Training examples
    x_train = [
        torch.randint(0, 10, (1, 10, 10)),
        torch.randint(0, 10, (1, 15, 15)) # Different size!
    ]
    y_train = [
        torch.randint(0, 10, (1, 10, 10)),
        torch.randint(0, 10, (1, 15, 15))
    ]
    
    # Forward pass
    # Assume we want output same size as input
    logits = model(x_test, x_train, y_train)
    
    print(f"Input Test: {x_test.shape}")
    print(f"Output Logits: {logits.shape}")
    
    assert logits.shape == (1, 10, 10, 10)
    print("Lean EBT System Integration Test passed!")

