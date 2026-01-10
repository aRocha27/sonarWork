# detection.py
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure ultralytics package is installed (for YOLOv8)
from config import CONFIDENCE_THRESHOLD, MODEL_WEIGHTS, MODEL_CLASSES, MATERIAL_MAP

# Load the YOLO model once at import time for efficiency
model = None

def load_model():
    """Load the YOLO model (lazy loading)."""
    global model
    if model is None:
        model = YOLO(MODEL_WEIGHTS)  # Load the pre-trained YOLO model
        if MODEL_CLASSES:
            model.names = MODEL_CLASSES  # Optionally restrict detection to specific classes
    return model


def detect_objects(image, data=None):
    """
    Perform object detection on the image.
    Optionally uses additional data (like depth or segmentation in `data`) if provided.
    Returns a list of detections, where each detection is a dict with keys:
    'label', 'confidence', 'bbox', 'material'.
    """
    if image is None:
        return []
    
    detections = []
    
    # Load model if not already loaded
    detection_model = load_model()
    
    # Run the model on the image (inference). Set verbose=False to suppress console output.
    results = detection_model.predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # YOLOv8 returns a Results list; take the first result (since we pass one image at a time)
    if len(results) > 0:
        result = results[0]
        
        # Iterate over detected boxes
        for box in result.boxes:
            cls_id = int(box.cls[0])  # class index of the detection
            conf = float(box.conf[0])  # confidence score of the detection
            
            if conf < CONFIDENCE_THRESHOLD:
                continue  # Skip detections below the confidence threshold
            
            # Determine the class label name
            if hasattr(detection_model, 'names'):
                # model.names is a dict mapping class indices to names
                label = detection_model.names.get(cls_id, str(cls_id))
            else:
                label = str(cls_id)
            
            # Get bounding box coordinates (x1, y1, x2, y2) as integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # YOLO gives coordinates which we convert to int
            bbox = [x1, y1, x2, y2]
            
            # Determine material type for this object
            material = MATERIAL_MAP.get(label, "unknown")
            
            # (Optional) Use `data` if available to refine material classification.
            # e.g., if `data` is a segmentation map of materials, one could sample it within the bbox.
            # Here we assume MATERIAL_MAP provides the material based on the label.
            
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": bbox,
                "material": material
            })
    
    return detections


def detect_with_sonar_data(image, echo_data=None, sonar_data=None, metadata=None):
    """
    Enhanced detection that can utilize sonar-specific data.
    
    Args:
        image: The preprocessed sonar scan image
        echo_data: Echo numpy array (from _dp_echo.npy)
        sonar_data: Sonar numpy array (from _dp_sonar.npy)
        metadata: JSON metadata dict
    
    Returns:
        List of detection dictionaries
    """
    # First, run standard object detection
    detections = detect_objects(image, echo_data)
    
    # Here you could enhance detections using sonar/echo data
    # For example:
    # - Use echo_data to estimate distance/depth of detected objects
    # - Use sonar_data for additional acoustic analysis
    # - Use metadata for contextual information
    
    # Example enhancement: Add depth information if echo_data is available
    if echo_data is not None and len(detections) > 0:
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            
            # Get center point of detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Sample echo data at detection center (if dimensions match)
            try:
                if echo_data.ndim >= 2:
                    # Ensure coordinates are within bounds
                    h, w = echo_data.shape[:2]
                    cy_clip = min(max(cy, 0), h - 1)
                    cx_clip = min(max(cx, 0), w - 1)
                    
                    # Add estimated depth/intensity from echo data
                    det["echo_value"] = float(echo_data[cy_clip, cx_clip]) if echo_data.ndim == 2 else float(echo_data[cy_clip, cx_clip, 0])
            except (IndexError, TypeError):
                pass
    
    return detections
