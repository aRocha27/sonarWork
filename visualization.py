# visualization.py
import cv2

def draw_detections(image, detections):
    """
    Draw bounding boxes and labels (object + material + confidence) on the image.
    Returns a new image with annotations.
    """
    # Make a copy of the image to draw on
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
        # Prepare the text string to display: "<label> (<material>) [<confidence>%]"
        text = f"{label} ({material}) [{conf*100:.1f}%]"
        # Position the text above the top-left corner of the bounding box
        text_x, text_y = x1, y1 - 10
        # Draw the text on the image (using a simple font)
        cv2.putText(annotated_image, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=2)
    return annotated_image
