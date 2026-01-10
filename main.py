# main.py
import os
import time
from datetime import datetime
from config import INPUT_DIR, OUTPUT_DIR
from preprocessing import load_all_sonar_data, preprocess, get_data_point_ids
from detection import detect_with_sonar_data, load_model
from visualization import draw_detections, create_summary_image
from export import save_results, save_summary_report, export_all_detections_combined


def process_single_data_point(data_point_id, input_dir=INPUT_DIR):
    """
    Process a single data point and return results.
    """
    # Load all sonar-related data
    sonar_data = load_all_sonar_data(input_dir, data_point_id)
    
    if sonar_data['image'] is None:
        print(f"⚠️  No image found for data point {data_point_id}")
        return None
    
    # Preprocess the image
    processed_image, _ = preprocess(sonar_data['image'])
    
    # Run detection with sonar data
    detections = detect_with_sonar_data(
        processed_image,
        echo_data=sonar_data['echo_data'],
        sonar_data=sonar_data['sonar_data'],
        metadata=sonar_data['metadata']
    )
    
    # Draw detections on image
    annotated_image = draw_detections(processed_image, detections)
    
    # Save results
    output_path = save_results(
        base_name=data_point_id,
        detections=detections,
        annotated_image=annotated_image,
        original_image=sonar_data['image'],
        metadata=sonar_data['metadata']
    )
    
    return {
        'id': data_point_id,
        'detections': detections,
        'total_detections': len(detections),
        'objects': list(set(d['label'] for d in detections)),
        'output_path': output_path,
        'timestamp': datetime.now().isoformat()
    }


def process_all_data_points(input_dir=INPUT_DIR):
    """
    Process all available data points in the input directory.
    """
    print(f"\n{'='*60}")
    print("🔍 OBJECT AND MATERIAL DETECTION SYSTEM")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all data point IDs
    data_point_ids = get_data_point_ids(input_dir)
    
    if not data_point_ids:
        print(f"❌ No data points found in {input_dir}")
        print("   Expected files: *_dp_pre_pros_sonar_scan.png")
        return
    
    print(f"📂 Found {len(data_point_ids)} data points to process")
    print(f"   IDs: {data_point_ids}\n")
    
    # Load model once
    print("🤖 Loading YOLO model...")
    load_model()
    print("✅ Model loaded successfully!\n")
    
    # Process each data point
    processed_data = []
    all_detections = {}
    
    for i, dp_id in enumerate(data_point_ids):
        print(f"[{i+1}/{len(data_point_ids)}] Processing data point {dp_id}...")
        
        try:
            result = process_single_data_point(dp_id, input_dir)
            
            if result:
                processed_data.append(result)
                all_detections[dp_id] = result['detections']
                
                print(f"   ✅ Detected {result['total_detections']} object(s)")
                if result['objects']:
                    print(f"   📦 Objects: {', '.join(result['objects'])}")
            else:
                print(f"   ⚠️  Skipped (no valid data)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Generate summary report
    if processed_data:
        save_summary_report(processed_data)
        export_all_detections_combined(all_detections)
    
    print(f"\n{'='*60}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"   Total data points: {len(processed_data)}")
    print(f"   Output directory: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")
    
    return processed_data


def monitor_mode(input_dir=INPUT_DIR, poll_interval=2.0):
    """
    Continuous monitoring mode - watches for new files and processes them.
    """
    print(f"\n{'='*60}")
    print("🔄 CONTINUOUS MONITORING MODE")
    print(f"{'='*60}")
    print(f"📂 Watching directory: {input_dir}")
    print(f"⏱️  Poll interval: {poll_interval}s")
    print("Press Ctrl+C to stop\n")
    
    processed = set()
    
    # Load model once
    print("🤖 Loading YOLO model...")
    load_model()
    print("✅ Model loaded! Starting monitoring...\n")
    
    try:
        while True:
            # Get current data point IDs
            data_point_ids = get_data_point_ids(input_dir)
            
            # Process new data points
            for dp_id in data_point_ids:
                if dp_id in processed:
                    continue
                
                print(f"\n📥 New data point detected: {dp_id}")
                
                try:
                    result = process_single_data_point(dp_id, input_dir)
                    
                    if result:
                        processed.add(dp_id)
                        print(f"   ✅ Processed: {result['total_detections']} object(s) detected")
                    
                except Exception as e:
                    print(f"   ❌ Error processing: {e}")
            
            # Wait before next scan
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("🛑 Monitoring stopped by user")
        print(f"   Total processed: {len(processed)} data points")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--monitor" or sys.argv[1] == "-m":
            # Run in continuous monitoring mode
            monitor_mode()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("\n📖 Usage:")
            print("  python main.py           - Process all existing data points")
            print("  python main.py --monitor - Run in continuous monitoring mode")
            print("  python main.py -m        - Same as --monitor")
            print("  python main.py --help    - Show this help message")
            print()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Default: process all existing data points
        process_all_data_points()
