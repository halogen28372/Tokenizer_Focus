import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionRuleEncoder(nn.Module):
    """
    Simplified version of the RuleHead.
    Uses Cross-Attention to query differences between X and Y in the support set.
    
    Structure:
    1. Takes pairs (x, y).
    2. Treats x as Queries/Keys.
    3. Treats (y - x) as Values.
    4. Computes a consensus rule vector.
    """
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.dim = dim
        
        # Projections
        self.query_proj = nn.Linear(dim, dim) # Queries from x_i
        self.key_proj = nn.Linear(dim, dim)   # Keys from x_j
        self.val_proj = nn.Linear(dim, dim)   # Values from (y_j - x_j)
        
        # Attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, examples):
        """
        Args:
            examples: List of tuples (x_enc, y_enc)
                      Each x_enc/y_enc is (B, L, D) or (B, D) depending on aggregation.
                      Assuming (B, D) global features for now, or pooled (B, L, D).
                      
        Returns:
            rule_vector: (B, dim)
        """
        # We need to construct Q, K, V from examples.
        # Let's assume examples are already encoded into feature vectors (B, D)
        # Or we flatten them.
        
        # For 'Lean' synthesis, let's assume we extract a single vector per example first.
        # If inputs are (B, L, D), we might pool them to (B, D) first for the rule query?
        # Or we can do pixel-level attention? Pixel-level is expensive (L*N * L*N).
        # Given "Fast guess", let's assume we work with pooled representations of examples.
        
        x_list = []
        delta_list = []
        
        for x, y in examples:
            # x, y are (B, L, D). Pool to (B, D) for rule extraction
            # Max pooling over spatial dims (better for object presence than mean)
            if x.dim() == 3:
                x_vec, _ = x.max(dim=1) # (B, D)
                y_vec, _ = y.max(dim=1) # (B, D)
            else:
                x_vec = x
                y_vec = y
                
            x_list.append(x_vec)
            delta_list.append(y_vec - x_vec)
            
        # Stack: (B, N_examples, D)
        X = torch.stack(x_list, dim=1) 
        Deltas = torch.stack(delta_list, dim=1)
        
        # Self-Attention among examples to find consistent rule
        # Q = X, K = X, V = Deltas
        # "For each example x_i, what is the delta implied by similar examples x_j?"
        
        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.val_proj(Deltas)
        
        # (B, N, D)
        attn_out, _ = self.attn(Q, K, V)
        
        # Average over all examples to get global rule
        # (B, D)
        rule_vector = attn_out.mean(dim=1)
        
        return self.out_proj(rule_vector)


class ResidualRuleRefiner(nn.Module):
    def __init__(self, rule_dim=512, latent_dim=64, num_train_tasks=400):
        super().__init__()
        
        # 1. The Explicit Rule Encoder (Fast System 1)
        self.rule_encoder = CrossAttentionRuleEncoder(dim=rule_dim) 
        
        # 2. The Residual Refiner (Slow System 2 - TTT)
        # Maps a small learnable latent z to a delta in rule space
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, rule_dim // 2),
            nn.GELU(),
            nn.Linear(rule_dim // 2, rule_dim)
        )
        
        # Initialize projection to near-zero so training starts with pure explicit rule
        self.latent_proj[-1].weight.data *= 0.01
        self.latent_proj[-1].bias.data.zero_()

        # 3. The Tiny Shared Memory (Offline Only) for Task Awareness
        # 400 tasks * 64 floats = ~100KB (Negligible)
        self.task_embed_table = nn.Embedding(num_train_tasks, latent_dim)

    def get_task_vector(self, task_ids=None, z_task=None):
        """
        Handles the logic: 
        - If Inference (z_task provided): Use the optimized z vector.
        - If Training (task_ids provided): Look up ID in table.
        - Else: Return None/Zero.
        """
        if z_task is not None:
            # Inference Mode: We are optimizing this vector directly
            return z_task
        elif task_ids is not None:
            # Offline Mode: Look up the static ID
            # task_ids should be (B,)
            return self.task_embed_table(task_ids)
        else:
            # Fallback (No task guidance)
            return None

    def forward(self, examples, test_input=None, z_task=None, task_ids=None):
        """
        Args:
            examples: List of (x, y) pairs from support set
            test_input: The query grid x* (Not used in rule inference, but kept for API compat)
            z_task: Optional learnable vector (B, latent_dim) for TTT
            task_ids: Optional integer IDs (B,) for offline training
            
        Returns:
            r_final: (B, rule_dim)
        """
        # A. FAST SYSTEM 1: Explicit Rule Inference
        # "Here is the structural relationship between example pairs"
        r_context = self.rule_encoder(examples) # (B, rule_dim)
        
        # Get the latent task vector (either from z_task or task_ids)
        task_vec = self.get_task_vector(task_ids, z_task)
        
        # B. SLOW SYSTEM 2: Residual Refinement
        if task_vec is not None:
            # Compute delta: "What did the encoder miss?"
            # task_vec is (B, latent_dim)
            if task_vec.dim() == 2 and task_vec.shape[0] != r_context.shape[0]:
                 task_vec = task_vec.expand(r_context.shape[0], -1)
                 
            delta_r = self.latent_proj(task_vec) # (B, rule_dim)
            
            # Additive Refinement: r_final = r_context + delta_r
            r_final = r_context + delta_r
        else:
            r_final = r_context
            
        return r_final

