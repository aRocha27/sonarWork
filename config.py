# config.py

# Default folders (used by main.py)
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "out"

# Detection parameters (tunable)
CONFIDENCE_THRESHOLD = 0.5

# Model configuration
MODEL_WEIGHTS = "yolov8n.pt"
MODEL_CLASSES = None

# Material mapping (placeholder)
MATERIAL_MAP = {
    "bottle": "plastic",
    "cup": "plastic",
    "chair": "wood",
    "table": "wood",
    "car": "metal",
    "pipe": "PVC",
    "bucket": "metal",
}
