# export.py
import os
import json
import csv
import cv2
from datetime import datetime
from config import OUTPUT_DIR


def save_results(base_name, detections, annotated_image, original_image=None, metadata=None):
    """
    Save the detection results for a given data point.
    - base_name: identifier for the data point (used as subfolder name in OUTPUT_DIR).
    - detections: list of detection dictionaries.
    - annotated_image: image (numpy array) with drawn detections.
    - original_image: original image for comparison (optional).
    - metadata: additional metadata dict (optional).
    """
    # Create an output subdirectory for this data point
    output_path = os.path.join(OUTPUT_DIR, str(base_name))
    os.makedirs(output_path, exist_ok=True)
    
    # Save the annotated image (with detections overlaid)
    if annotated_image is not None:
        image_file = os.path.join(output_path, f"{base_name}_detections.png")
        cv2.imwrite(image_file, annotated_image)
    
    # Save original image for reference
    if original_image is not None:
        original_file = os.path.join(output_path, f"{base_name}_original.png")
        cv2.imwrite(original_file, original_image)
    
    # Prepare detection data with timestamp
    detection_data = {
        "data_point": base_name,
        "timestamp": datetime.now().isoformat(),
        "total_detections": len(detections),
        "detections": detections
    }
    
    # Include original metadata if available
    if metadata is not None:
        detection_data["original_metadata"] = metadata
    
    # Save detections to a JSON file
    json_file = os.path.join(output_path, f"{base_name}_detections.json")
    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump(detection_data, jf, indent=4, ensure_ascii=False)
    
    # Save detections to a CSV file
    csv_file = os.path.join(output_path, f"{base_name}_detections.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        # Write header
        header = ["label", "material", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        
        # Add echo_value column if any detection has it
        if any("echo_value" in det for det in detections):
            header.append("echo_value")
        
        writer.writerow(header)
        
        # Write one row per detected object
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            row = [det["label"], det["material"], det.get("confidence", 0), x1, y1, x2, y2]
            
            if "echo_value" in det:
                row.append(det.get("echo_value", ""))
            elif len(header) > 7:
                row.append("")
            
            writer.writerow(row)
    
    return output_path


def save_summary_report(processed_data_points, output_dir=OUTPUT_DIR):
    """
    Generate a summary report of all processed data points.
    """
    summary_file = os.path.join(output_dir, "processing_summary.json")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_data_points": len(processed_data_points),
        "data_points": processed_data_points
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    # Also create a CSV summary
    csv_summary_file = os.path.join(output_dir, "processing_summary.csv")
    with open(csv_summary_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["data_point_id", "total_detections", "objects_detected", "timestamp"])
        
        for dp in processed_data_points:
            objects = ", ".join(dp.get("objects", []))
            writer.writerow([
                dp.get("id", ""),
                dp.get("total_detections", 0),
                objects,
                dp.get("timestamp", "")
            ])
    
    print(f"\n📊 Summary report saved to: {summary_file}")
    return summary_file


def export_all_detections_combined(all_detections, output_dir=OUTPUT_DIR):
    """
    Export all detections from all data points to a single combined file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    combined_file = os.path.join(output_dir, "all_detections_combined.csv")
    
    with open(combined_file, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(["data_point", "label", "material", "confidence", 
                        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
        
        for data_point_id, detections in all_detections.items():
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                writer.writerow([
                    data_point_id,
                    det["label"],
                    det["material"],
                    det.get("confidence", 0),
                    x1, y1, x2, y2
                ])
    
    print(f"\n📁 Combined detections exported to: {combined_file}")
    return combined_file
