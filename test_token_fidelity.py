"""
Test if tokenize → render → grid has perfect fidelity
"""
import torch
import numpy as np
from arc_lago_tokenizer import ArcLagoTokenizer, TokenizerConfig
from token_renderer import render_tokens_to_grid

# Initialize tokenizer
tokenizer = ArcLagoTokenizer(TokenizerConfig(
    max_layers=4, emit_symmetry_meta=True, emit_relations=True,
    assume_rect_occlusion_prior=True, merge_diagonal_lines=True
))

# Create a simple test grid
test_grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

print("Original grid:")
for row in test_grid:
    print(row)

# Tokenize
result = tokenizer.tokenize(test_grid)
tokens = result["shapes"]

print(f"\nExtracted {len(tokens)} tokens:")
for i, tok in enumerate(tokens):
    print(f"  {i}: {tok}")

# Render back
H, W = 5, 5
C = 10
rendered = render_tokens_to_grid(tokens, H, W, C, device='cpu')
print(f"\nRendered shape: {rendered.shape}")

# Convert to hard labels
rendered_hard = rendered.argmax(dim=-1)
print("\nRendered grid:")
print(rendered_hard.numpy())

# Check fidelity
original = torch.tensor(test_grid, dtype=torch.long)
matches = (rendered_hard == original).sum().item()
total = H * W
print(f"\nFidelity: {matches}/{total} pixels match ({100*matches/total:.1f}%)")

if matches != total:
    print("\n❌ FIDELITY LOSS DETECTED")
    print("Differences:")
    for i in range(H):
        for j in range(W):
            if original[i, j] != rendered_hard[i, j]:
                print(f"  ({i},{j}): original={original[i,j].item()}, rendered={rendered_hard[i,j].item()}")
else:
    print("\n✓ Perfect fidelity!")

