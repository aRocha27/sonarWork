# detection.py
import cv2
import numpy as np
from ultralytics import YOLO  # Ensure ultralytics package is installed (for YOLOv8)
from config import CONFIDENCE_THRESHOLD, MODEL_WEIGHTS, MODEL_CLASSES, MATERIAL_MAP

# Load the YOLO model once at import time for efficiency
model = YOLO(MODEL_WEIGHTS)  # Load the pre-trained YOLO model (e.g., COCO-pretrained or custom weights)
if MODEL_CLASSES:
    model.names = MODEL_CLASSES  # Optionally restrict detection to specific classes

def detect_objects(image, data=None):
    """
    Perform object detection on the image.
    Optionally uses additional data (like depth or segmentation in `data`) if provided.
    Returns a list of detections, where each detection is a dict with keys:
    'label', 'confidence', 'bbox', 'material'.
    """
    detections = []
    # Run the model on the image (inference). Set verbose=False to suppress console output.
    results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, verbose=False)
    # YOLOv8 returns a Results list; take the first result (since we pass one image at a time)
    if len(results) > 0:
        result = results[0]
        # Iterate over detected boxes
        for box in result.boxes:
            cls_id = int(box.cls[0])              # class index of the detection
            conf = float(box.conf[0])             # confidence score of the detection
            if conf < CONFIDENCE_THRESHOLD:
                continue  # Skip detections below the confidence threshold (already filtered by model)
            # Determine the class label name
            if hasattr(model, 'names'):
                # model.names is a dict mapping class indices to names
                label = model.names.get(cls_id, str(cls_id))
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
