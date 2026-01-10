# export.py
import os
import json
import csv
import cv2

def save_results(base_name, out_dir, detections, image):
    """
    Saves:
    - overlay image
    - JSON with detections
    - CSV summary
    into out_dir/base_name/
    """

    os.makedirs(out_dir, exist_ok=True)

    # Save image
    img_path = os.path.join(out_dir, f"{base_name}_detections.png")
    cv2.imwrite(img_path, image)

    # Save JSON
    json_path = os.path.join(out_dir, f"{base_name}_detections.json")
    with open(json_path, "w") as f:
        json.dump({"detections": detections}, f, indent=2)

    # Save CSV
    csv_path = os.path.join(out_dir, f"{base_name}_detections.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "material", "confidence", "x1", "y1", "x2", "y2"])
        for d in detections:
            writer.writerow([d["label"], d["material"], d["confidence"], *d["bbox"]])
