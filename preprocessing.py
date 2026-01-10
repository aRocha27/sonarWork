# preprocessing.py
import numpy as np
import cv2
import os
import json

def load_data(image_path, data_path=None):
    """
    Load the image and corresponding npy data (e.g., depth or segmentation).
    Returns a tuple (image, data).
    """
    # Load image using OpenCV (returns image in BGR format)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load numpy data if available
    data = None
    if data_path is not None and os.path.exists(data_path) and data_path.endswith(".npy"):
        data = np.load(data_path)
    
    return image, data


def load_json_metadata(json_path):
    """
    Load JSON metadata file if it exists.
    Returns metadata dict or None.
    """
    if json_path is not None and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def load_all_sonar_data(base_path, data_point_id):
    """
    Load all sonar-related data for a given data point.
    
    Args:
        base_path: Directory containing the input files
        data_point_id: Integer ID of the data point (e.g., 0, 1, 2...)
    
    Returns:
        dict with 'image', 'echo_data', 'sonar_data', 'metadata'
    """
    # Build file paths based on your naming convention
    image_path = os.path.join(base_path, f"{data_point_id}_dp_pre_pros_sonar_scan.png")
    echo_path = os.path.join(base_path, f"{data_point_id}_dp_echo.npy")
    sonar_path = os.path.join(base_path, f"{data_point_id}_dp_sonar.npy")
    json_path = os.path.join(base_path, f"{data_point_id}_dp.json")
    
    result = {
        'image': None,
        'echo_data': None,
        'sonar_data': None,
        'metadata': None,
        'data_point_id': data_point_id
    }
    
    # Load image
    if os.path.exists(image_path):
        result['image'] = cv2.imread(image_path)
    
    # Load echo data
    if os.path.exists(echo_path):
        result['echo_data'] = np.load(echo_path)
    
    # Load sonar data
    if os.path.exists(sonar_path):
        result['sonar_data'] = np.load(sonar_path)
    
    # Load JSON metadata
    if os.path.exists(json_path):
        result['metadata'] = load_json_metadata(json_path)
    
    return result


def preprocess(image, data=None):
    """
    Apply any required preprocessing to the image or data before detection.
    (Placeholder for normalization, resizing, etc., if needed.)
    """
    if image is None:
        return None, data
    
    # Example preprocessing steps (uncomment as needed):
    # 1. Convert to grayscale if needed
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian blur to reduce noise
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 3. Enhance contrast for sonar images
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # l = clahe.apply(l)
    # lab = cv2.merge((l, a, b))
    # image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 4. Resize if needed
    # image = cv2.resize(image, (640, 640))
    
    return image, data


def get_data_point_ids(input_dir):
    """
    Scan input directory and return list of unique data point IDs.
    Based on files like: 0_dp_pre_pros_sonar_scan.png, 1_dp_pre_pros_sonar_scan.png, etc.
    """
    ids = set()
    
    for filename in os.listdir(input_dir):
        if filename.endswith("_dp_pre_pros_sonar_scan.png"):
            # Extract the ID from the filename
            try:
                data_id = int(filename.split("_dp_")[0])
                ids.add(data_id)
            except ValueError:
                continue
    
    return sorted(list(ids))
