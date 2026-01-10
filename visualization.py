# visualization.py
import cv2
import numpy as np

def draw_detections(image, detections):
    """
    Draw bounding boxes and labels (object + material + confidence) on the image.
    Returns a new image with annotations.
    """
    if image is None:
        return None
    
    # Make a copy of the image to draw on
    annotated_image = image.copy()
    
    # Define colors for different materials
    material_colors = {
        "plastic": (0, 255, 255),    # Yellow
        "wood": (42, 82, 190),       # Brown-ish
        "metal": (192, 192, 192),    # Silver
        "PVC": (255, 144, 30),       # Light blue
        "organic": (0, 255, 0),      # Green
        "stone": (128, 128, 128),    # Gray
        "unknown": (255, 0, 0),      # Blue
    }
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        material = det["material"]
        conf = det.get("confidence", 0)
        
        # Get color based on material
        color = material_colors.get(material, (0, 255, 0))  # Default green
        
        # Draw the rectangle for the object's bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness=2)
        
        # Prepare the text string to display: "<label> (<material>) [<confidence>%]"
        text = f"{label} ({material}) [{conf*100:.1f}%]"
        
        # Add echo value if available
        if "echo_value" in det:
            text += f" E:{det['echo_value']:.2f}"
        
        # Calculate text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position the text above the top-left corner of the bounding box
        text_x, text_y = x1, y1 - 10
        
        # Ensure text is visible (not above image boundary)
        if text_y < text_height + 5:
            text_y = y1 + text_height + 5
        
        # Draw background rectangle for text
        cv2.rectangle(annotated_image, 
                     (text_x, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + 5), 
                     color, -1)
        
        # Draw the text on the image
        cv2.putText(annotated_image, text, (text_x, text_y),
                    font, fontScale=font_scale, color=(0, 0, 0),
                    thickness=thickness)
    
    return annotated_image


def draw_detections_simple(image, detections):
    """
    Simple version without material colors - just green boxes.
    """
    if image is None:
        return None
    
    annotated_image = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        material = det["material"]
        conf = det.get("confidence", 0)
        
        # Define a color for the bounding box (BGR format). Here, use green for all boxes.
        color = (0, 255, 0)  # Green color in BGR
        
        # Draw the rectangle for the object's bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness=2)
        
        # Prepare the text string to display
        text = f"{label} ({material}) [{conf*100:.1f}%]"
        
        # Position the text above the top-left corner of the bounding box
        text_x, text_y = x1, y1 - 10
        
        # Draw the text on the image
        cv2.putText(annotated_image, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color,
                    thickness=2)
    
    return annotated_image


def create_summary_image(original_image, annotated_image, detections):
    """
    Create a side-by-side comparison image showing original and detected objects.
    """
    if original_image is None or annotated_image is None:
        return annotated_image
    
    # Ensure both images have the same height
    h1, w1 = original_image.shape[:2]
    h2, w2 = annotated_image.shape[:2]
    
    max_h = max(h1, h2)
    
    # Resize images to have the same height if needed
    if h1 != max_h:
        scale = max_h / h1
        original_image = cv2.resize(original_image, (int(w1 * scale), max_h))
    if h2 != max_h:
        scale = max_h / h2
        annotated_image = cv2.resize(annotated_image, (int(w2 * scale), max_h))
    
    # Concatenate images horizontally
    combined = np.hstack([original_image, annotated_image])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, f"Detected: {len(detections)} objects", 
                (original_image.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    return combined
