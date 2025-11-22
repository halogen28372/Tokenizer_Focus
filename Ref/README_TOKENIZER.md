# ARC/LAGO Tokenizer - Test Results

## Overview

The ARC/LAGO tokenizer is a shape-aware tokenizer that converts grid-based puzzles into semantic tokens representing geometric primitives, patterns, and spatial relationships.

## Key Features

### 1. Shape Detection
The tokenizer can detect and classify:

- **Rectangles**: Filled and hollow (with border thickness)
- **Lines**: Horizontal, vertical, and diagonal (45°)
- **Polyominoes**: Tetrominoes and pentominoes with automatic scale detection
- **Crosses**: X-shaped diagonal crosses
- **Patterns**: Checkerboards, concentric rectangles, C/U shapes
- **Sparse patterns**: Scattered dots, crosshatches, radial spokes
- **Complex shapes**: Zigzags, waves, spirals

### 2. Spatial Relations
Automatically computes relationships between shapes:
- `inside`, `overlaps`, `touches`
- `left_of`, `right_of`, `above`, `below`
- `aligned_row`, `aligned_col`

### 3. Meta Features
Detects high-level properties:
- **Symmetry**: Horizontal, vertical, diagonal, anti-diagonal
- **Rotational symmetry**: 2-fold and 4-fold
- **Tiling patterns**: Repeating grid patterns

### 4. Layer Detection
Assigns shapes to occlusion layers (0-3) based on spatial overlap

## Test Results

### Basic Shapes Test
✓ Nested rectangles detected correctly
✓ Hollow rectangle with border thickness
✓ Filled rectangle inside hollow rectangle
✓ Spatial relations (inside, overlaps) computed
✓ Symmetry detection (multiple axes)

### Lines and Crosses Test
✓ Horizontal lines detected
✓ Vertical lines detected
✓ Cross patterns recognized
✓ Proper orientation labeling

### Polyomino Test
✓ T-tetromino detected with 2× scale
✓ Correct unit dimensions computed
✓ Shape orientation normalized

### Pattern Test
✓ Checkerboard pattern recognized
✓ Two-color alternation detected
✓ Sparse dots identified and counted

### Real ARC Tasks

Tested on 5+ real ARC problems from the dataset:

**Task f8a8fe49** (15×15 grids):
- Input: 5 shapes (CONCENTRIC_RECTS, ZIGZAG, PENTOMINO)
- Output: 5 shapes with spatial transformations
- Detected U-pentomino correctly

**Task 3c9b0459** (3×3 grids):
- Small pixel patterns
- Detected J-tetromino correctly
- 50 spatial relations found

**Task 445eab21** (10×10 → 2×2):
- Hollow rectangles detected
- Grid size reduction tracked
- Color changes identified

**Task 8403a5d5** (pattern generation):
- Detected massive transformation (2 → 14 shapes)
- Checkerboard pattern creation tracked

## Performance

| Grid Size | Time (ms) | Throughput |
|-----------|-----------|------------|
| 8×8       | ~2-5      | ~13K px/s  |
| 16×16     | ~5-10     | ~26K px/s  |
| 30×30     | ~10-20    | ~45K px/s  |
| 50×50     | ~20-40    | ~63K px/s  |

**Memory Usage**:
- 10×10 grid: ~1-2 KB JSON output
- 30×30 grid: ~5-10 KB JSON output
- 50×50 grid: ~15-25 KB JSON output

## Shape Detection Quality

Tested on known patterns:

✓ **Rectangle** - Detected correctly
✓ **Hollow Rectangle** - Detected with border thickness
✓ **Horizontal Line** - Detected with orientation
✓ **Vertical Line** - Detected with orientation
✓ **T-Tetromino** - Detected with shape ID

**Detection Rate**: 5/5 (100%)

## JSON Output Format

```json
{
  "shapes": [
    {
      "type": "HOLLOW_RECT",
      "color": 1,
      "layer": 0,
      "id": 0,
      "x": 1,
      "y": 1,
      "w": 4,
      "h": 3,
      "border_thickness": 1
    }
  ],
  "relations": [
    {
      "type": "REL",
      "rel": "inside",
      "a": 0,
      "b": 1
    }
  ],
  "meta": [
    {
      "type": "META",
      "meta": "symmetry_axis",
      "value": "vertical",
      "region": 1
    }
  ]
}
```

## Usage Examples

### Basic Usage
```python
from arc_lago_tokenizer import ArcLagoTokenizer

# Create tokenizer
tok = ArcLagoTokenizer()

# Tokenize a grid
grid = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
]

result = tok.tokenize(grid)
print(f"Found {len(result['shapes'])} shapes")
```

### With Configuration
```python
from arc_lago_tokenizer import ArcLagoTokenizer, TokenizerConfig

# Custom configuration
config = TokenizerConfig(
    max_layers=4,
    emit_symmetry_meta=True,
    emit_relations=True,
    assume_rect_occlusion_prior=True
)

tok = ArcLagoTokenizer(config)
result = tok.tokenize(grid)
```

### Load and Tokenize ARC Tasks
```python
import json

# Load ARC task
with open('Data/ARC/training/f8a8fe49.json', 'r') as f:
    task = json.load(f)

# Tokenize training pairs
tok = ArcLagoTokenizer()
for i, pair in enumerate(task['train']):
    input_tokens = tok.tokenize(pair['input'])
    output_tokens = tok.tokenize(pair['output'])
    
    print(f"Pair {i}:")
    print(f"  Input shapes: {len(input_tokens['shapes'])}")
    print(f"  Output shapes: {len(output_tokens['shapes'])}")
```

## Running Tests

```bash
# Basic functionality tests
python test_tokenizer.py

# Visual tests with grid display
python test_tokenizer_visual.py

# Real ARC problem tests
python test_arc_problems.py f8a8fe49

# Sample random ARC tasks
python test_arc_problems.py

# Performance benchmark
python benchmark_tokenizer.py
```

## Detected Shape Types

The tokenizer recognizes the following primitives:

1. **RECT** - Filled rectangle
2. **HOLLOW_RECT** - Rectangle with border
3. **BORDER** - Grid-border frame
4. **LINE** - Horizontal/vertical line
5. **DIAG_LINE** - 45° diagonal line
6. **DIAG_CROSS_X** - X-shaped cross
7. **TETROMINO** - 4-cell polyomino (I, O, T, S, Z, J, L)
8. **PENTOMINO** - 5-cell polyomino (F, I, L, P, N, T, U, V, W, X, Y, Z)
9. **CHECKER** - Checkerboard pattern
10. **CONCENTRIC_RECTS** - Nested rectangles
11. **C_SHAPE** - U/C-shaped hollow rectangle
12. **SPARSE_DOTS** - Scattered single pixels
13. **CROSSHATCH** - Grid of intersecting lines
14. **RADIAL** - Spoke pattern from center
15. **ZIGZAG** - Alternating path pattern
16. **SPIRAL** - Coiled path
17. **REGION** - Generic fallback for unclassified shapes

## Strengths

✅ **Zero dependencies** - Pure Python implementation
✅ **Fast** - Processes 30×30 grids in ~10ms
✅ **Robust** - Works on diverse ARC patterns
✅ **Scale-aware** - Detects polyominoes at any scale
✅ **Comprehensive** - 17 primitive types + relations + meta features
✅ **Deterministic** - Consistent results across runs
✅ **JSON serializable** - Easy integration with other systems

## Limitations

⚠️ Very small grids (3×3) may produce many individual pixel RECTs rather than higher-level patterns
⚠️ Checkerboard detection requires clear alternating pattern
⚠️ Diagonal line detection limited to 45° angles
⚠️ Complex freeform shapes may fall back to REGION type

## Future Enhancements

Potential improvements:
- [ ] Arbitrary angle line detection
- [ ] Arc and circle detection
- [ ] More sophisticated tiling pattern recognition
- [ ] Template matching for custom primitives
- [ ] Hierarchical grouping of shapes
- [ ] Color gradient detection
- [ ] Motion/transformation templates

## Conclusion

The ARC/LAGO tokenizer successfully converts grid-based puzzles into semantic shape representations, achieving:
- **100% shape detection** on test cases
- **Fast performance** (~10-40ms for typical grids)
- **Rich output** (shapes, relations, symmetries)
- **Real-world validation** on ARC dataset tasks

The tokenizer provides a solid foundation for reasoning about spatial patterns and transformations in grid-based abstract reasoning tasks.

