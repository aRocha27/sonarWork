# config.py
import os

# Directory paths
INPUT_DIR = "input"  # Directory where input .png and .npy files are stored
OUTPUT_DIR = "out"   # Base directory for output results (each data point gets a subfolder)

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for object detection (adjustable margin of error)

# Model configuration (for object detection)
MODEL_WEIGHTS = "yolov8n.pt"  # Path to the pre-trained YOLO model weights
MODEL_CLASSES = None  # Specify a list of class names to detect, or None to detect all classes

# Mapping from object classes to material types (can be customized for the application)
MATERIAL_MAP = {
    "bottle": "plastic",
    "cup": "plastic",
    "chair": "wood",
    "table": "wood",
    "car": "metal",
    "pipe": "PVC",
    "bucket": "metal",
    "person": "organic",
    "fish": "organic",
    "boat": "metal",
    "rock": "stone",
    # Add other object-to-material mappings as needed for sonar data
}

# Sonar-specific settings
SONAR_FILE_PATTERN = "_dp_pre_pros_sonar_scan.png"  # Pattern to identify sonar scan images
ECHO_FILE_PATTERN = "_dp_echo.npy"  # Pattern for echo data
SONAR_NPY_PATTERN = "_dp_sonar.npy"  # Pattern for sonar numpy data
JSON_PATTERN = "_dp.json"  # Pattern for JSON metadata
