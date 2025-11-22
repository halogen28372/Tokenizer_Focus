"""
Configuration for EBT training with ARC/LAGO tokenizer
"""

class Config:
    # Model dimensions (SCALED UP 2x for increased capacity)
    NUM_COLORS = 10  # ARC uses 10 colors (0-9)
    D_COLOR = 128  # Color embedding dimension (64 -> 128)
    D_FEAT = 1024  # Feature dimension (512 -> 1024)
    D_RULE = 512  # Rule dimension (256 -> 512)
    DEC_CHANNELS = 512  # Decoder channels (256 -> 512)
    
    # Neighbor consensus encoder
    USE_NEIGHBOR_ENCODER = False  # Disabled when using tokenizer
    NCE_DIM = 32
    NCE_SCALES = (3, 5)
    
    # Tokenizer-based encoder (SCALED UP)
    USE_TOKENIZER = True
    TOKENIZER_TYPE = 'object_aware'  # 'arc_lago' or 'object_aware'
    TOKENIZER_MAX_TOKENS = 120  # Max number of tokens to process (100 -> 120)
    TOKENIZER_TOKEN_DIM = 256  # Dimension for token embeddings (128 -> 256)
    TOKENIZER_USE_RELATIONS = True
    TOKENIZER_USE_META = True
    # Object-aware tokenizer specific (SCALED UP)
    OBJECT_AWARE_EMBED_DIM = 128  # (64 -> 128)
    OBJECT_AWARE_SHAPE_EMBED_DIM = 64  # (32 -> 64)
    
    # Multi-item canvas (disabled for now)
    USE_MULTI_ITEM_CANVAS = False
    CANVAS_TILE_STRIDE_X = 2
    CANVAS_TILE_STRIDE_Y = 2
    CANVAS_LANE_GAP = 1
    CANVAS_SEGMENT_DIM = 32
    CANVAS_USE_HARD_MASK = True
    
    # Spatial supervision (disabled for now)
    USE_SPATIAL_SUPERVISION = False
    SPATIAL_FOURIER_FREQS = 4
    
    # Neighbor solver (disabled when using tokenizer)
    USE_NEIGHBOR_SOLVER = False
    NEIGHBOR_DIRS = [(0,1), (1,0), (0,-1), (-1,0)]
    NEIGHBOR_COLOR_DROPOUT = 0.5
    
    # Energy function
    ENERGY_USE_CROSS_ATTN = False
    ENERGY_USE_FILM = False
    ENERGY_LOGIT_SCALE = 1.0
    
    # Rule aggregation
    ALPHA_RULE = 0.4  # Weight for demo rules
    BETA_RULE = 0.2   # Weight for test input rule
    
    # Training
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1  # ARC tasks are processed one at a time
    
    # Decoder type (SCALED UP)
    DECODER_TYPE = 'token'  # 'conv', 'transformer', or 'token'
    TOKEN_DECODER_MAX_TOKENS = 24  # (16 -> 24)
    TOKEN_DECODER_HIDDEN = 512  # (256 -> 512)
    TOKEN_DECODER_LAYERS = 6  # (4 -> 6)
    TOKEN_DECODER_TYPES = ('RECT', 'HOLLOW_RECT', 'LINE', 'BORDER')
    TOKEN_RENDER_EDGE_SHARPNESS = 12.0
    TOKEN_RENDER_MIN_SIZE = 0.08  # fraction of the grid span
    TOKEN_RENDER_LAYER_TEMP = 0.7
    TOKEN_PRESENCE_THRESHOLD = 0.35
    
    # Debug flags
    DEBUG_ENABLE_SCALING = False
    DEBUG_LOGIT_SCALE = 1.0
    DEBUG_ENERGY_SCALE = 1.0

config = Config()
