import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from ttt_augmentation import TTTAugmenter

@dataclass
class DeepTTTConfig:
    """Hyperparameters for Deep Test-Time Training."""
    steps: int = 50
    lr_weights: float = 1e-4  # Learning rate for model weights (usually lower)
    lr_z: float = 0.05       # Learning rate for task embeddings (can be higher)
    batch_size: int = 8
    num_augmentations: int = 50
    update_refiner: bool = True
    update_encoder: bool = False
    update_decoder: bool = False
    use_per_aug_z: bool = True # Whether to give each augmentation its own z vector

class DeepTTTEngine:
    """
    Engine for performing Deep Test-Time Training (Deep TTT).
    
    Key features:
    1. Generates augmented support tasks.
    2. Updates MODEL WEIGHTS + Task Embeddings on these tasks.
    3. Restores model state after inference (Stateless).
    """
    def __init__(self, model, config: DeepTTTConfig, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.augmenter = TTTAugmenter(num_augmentations=config.num_augmentations, device=device)
        
    def run_inference(self, x_train_list, y_train_list, x_test, target_shape):
        """
        Run the full Deep TTT pipeline for a single test task.
        
        Args:
            x_train_list: List of original support inputs
            y_train_list: List of original support outputs
            x_test: Test input
            target_shape: Expected output shape
            
        Returns:
            pred_grid: Predicted output grid
            final_loss: Final loss on support set
        """
        # 1. Save Model State (Snapshot)
        # We only need to save the parts we modify.
        # For safety, let's save the whole state dict to CPU to save GPU memory.
        original_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        try:
            # 2. Prepare Data (Augmentation)
            # Generate augmented pairs: (N_aug, H, W)
            x_aug, y_aug = self.augmenter.augment(x_train_list, y_train_list)
            num_tasks = len(x_aug)
            print(f"    [DeepTTT] Generated {num_tasks} augmented tasks (Originals: {len(x_train_list)})")
            
            # 3. Initialize Task Embeddings
            # If use_per_aug_z is True, we need a batch of z vectors: (Num_Tasks, Latent_Dim)
            latent_dim = self.model.refiner.task_embed_table.embedding_dim
            
            if self.config.use_per_aug_z:
                # Unique z for each augmented task
                z_batch = torch.zeros(num_tasks, latent_dim, device=self.device, requires_grad=True)
            else:
                # Shared z for all (like our previous failed attempt)
                z_single = torch.zeros(1, latent_dim, device=self.device, requires_grad=True)
                z_batch = z_single # Will expand/broadcast later if needed
                
            # 4. Setup Optimizer
            # We optimize z_batch AND selected model parameters
            params_to_optimize = []
            
            # A. Task Embeddings
            if self.config.use_per_aug_z:
                params_to_optimize.append({'params': [z_batch], 'lr': self.config.lr_z})
            else:
                params_to_optimize.append({'params': [z_single], 'lr': self.config.lr_z})
                
            # B. Model Weights
            model_params = []
            if self.config.update_refiner:
                model_params.extend(self.model.refiner.parameters())
            if self.config.update_encoder:
                model_params.extend(self.model.encoder.parameters())
            if self.config.update_decoder:
                model_params.extend(self.model.decoder.parameters())
                
            if model_params:
                params_to_optimize.append({'params': model_params, 'lr': self.config.lr_weights})
                print(f"    [DeepTTT] optimizing {len(model_params)} parameter groups in model")
            
            optimizer = optim.Adam(params_to_optimize)
            
            # 5. Training Loop
            self.model.train() # Set to train mode (enable Dropout/BatchNorm updates? Maybe eval is safer?)
            # Actually, for TTT usually we want BatchNorm running stats to be frozen or adapting?
            # Standard is usually eval() mode for BN, but enable grad.
            self.model.eval() 
            
            final_loss = 0
            
            start_time = time.time()
            for step in range(self.config.steps):
                optimizer.zero_grad()
                total_step_loss = 0
                
                # Batch Optimization Loop
                # We shuffle indices every step? Or just deterministic iteration?
                indices = torch.randperm(num_tasks).tolist()
                
                for i in range(0, num_tasks, self.config.batch_size):
                    batch_indices = indices[i : i + self.config.batch_size]
                    current_batch_size = len(batch_indices)
                    
                    batch_loss = 0
                    
                    for idx in batch_indices:
                        x = x_aug[idx]
                        y = y_aug[idx]
                        
                        # Determine which z to use
                        if self.config.use_per_aug_z:
                            z_curr = z_batch[idx].unsqueeze(0) # (1, D)
                        else:
                            z_curr = z_single # (1, D)
                            
                        # Forward Pass
                        # Note: We pass the FULL AUGMENTED SET as context? 
                        # Or just the originals? Or just this one?
                        # Paper says "augment the few demos into 51 auxiliary tasks... each gets own embedding".
                        # The "Rule" should be derived from context.
                        # Ideally, the rule for "Rotated Task A" should be derived from "Rotated Task A Support".
                        # But here each augmented task is a single pair (1-shot).
                        # So context is just itself? Or itself + others?
                        # Let's assume context = [x] (1-shot learning for each aug task).
                        
                        # CAREFUL: model.forward signature is (x_test, x_train_list, y_train_list, ...)
                        # If we treat this augmented pair as a task, then:
                        # x_train_list = [x], y_train_list = [y] (Support)
                        # x_test = x (Query - reconstructing support)
                        
                        # Let's use self-reconstruction
                        logits = self.model(x, [x], [y], z_task=z_curr, target_shape=(y.shape[1], y.shape[2]))
                        
                        # Loss
                        logits_perm = logits.permute(0, 3, 1, 2)
                        loss = nn.CrossEntropyLoss()(logits_perm, y)
                        
                        # Accumulate (normalize by batch size)
                        loss = loss / current_batch_size
                        loss.backward()
                        
                        batch_loss += loss.item() * current_batch_size
                    
                    # No optimizer step here, we are accumulating gradients over batches? 
                    # No, standard mini-batch training updates weights every batch.
                    # Let's update weights every batch for stochasticity.
                    optimizer.step()
                    optimizer.zero_grad() # Reset for next batch
                    
                    total_step_loss += batch_loss
                
                # Average loss per sample
                avg_loss = total_step_loss / num_tasks
                final_loss = avg_loss
                
                if step % 10 == 0 or step == self.config.steps - 1:
                    print(f"      Step {step}: Loss = {avg_loss:.4f}")
            
            # 6. Final Prediction
            # We use the optimized model weights
            # AND the z vector corresponding to the ORIGINAL task (which is index 0 usually, or first few)
            # Actually, we have multiple original examples (Indices 0 to K-1).
            # Which z do we use for test?
            # The test task shares the orientation of the original support set.
            # So we should use the "consensus" z of the original examples.
            # Or just average the z's of the original examples?
            
            # Let's average the z's corresponding to the original support set.
            # x_aug starts with x_train_list.
            num_originals = len(x_train_list)
            
            with torch.no_grad():
                if self.config.use_per_aug_z:
                    z_final = z_batch[:num_originals].mean(dim=0, keepdim=True)
                else:
                    z_final = z_single
                    
                # Run inference on test input
                # Context is the ORIGINAL support set
                logits = self.model(x_test, x_train_list, y_train_list, z_task=z_final, target_shape=target_shape)
                pred_grid = logits.argmax(dim=-1)
                
            duration = time.time() - start_time
            print(f"    [DeepTTT] Finished in {duration:.1f}s. Final Loss: {final_loss:.4f}")
            
            return pred_grid, final_loss

        finally:
            # 7. Restore Model State
            print("    [DeepTTT] Restoring model state...")
            self.model.load_state_dict(original_state)
            

