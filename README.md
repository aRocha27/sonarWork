Sonar Object Detection with YOLOv8 and Hybrid Intensity Classification
📌 Overview

This project implements a hybrid system for object detection and reflection intensity classification in sonar images.
The solution combines a deep learning detector (YOLOv8) with physics-informed post-processing, designed to overcome the limitations of direct multi-class learning on acoustical data.

The system detects objects in sonar images and classifies them into three reflection intensity levels:

lowReflection

mediumReflection

highReflection

🎯 Motivation

Sonar reflection intensity is a continuous physical quantity, not a discrete visual category.
Initial attempts to classify reflection levels directly using YOLO led to:

strong confusion between adjacent classes

inconsistent predictions across detections

poor explainability of results

To address this, the project adopts a hybrid architecture:

YOLOv8 is used only for object detection (object vs background), while reflection intensity is classified using a dedicated post-processing step based on acoustic signal strength inside each bounding box.

This approach is:

more robust

physically interpretable

academically defensible

🧠 Final Architecture
Sonar Image
     ↓
YOLOv8 (single-class: object)
     ↓
Bounding Boxes
     ↓
Hybrid Intensity Analysis (red_excess + density)
     ↓
low / medium / high reflection

📁 Project Structure
sonarWork/
│
├── configs/
│   └── data.yaml                # YOLO dataset configuration
│
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   └── obj/
│       └── best_obj_base200.pt  # trained YOLOv8 model
│
├── runs/                        # YOLO training outputs (auto-generated)
│
├── outputs/
│   ├── pred_object/             # YOLO-only detections
│   └── pred_hybrid/             # Final hybrid results
│
├── src/
│   └── infer_hybrid.py          # Hybrid inference script
│
├── requirements.txt
└── README.md


📌 The runs/ directory is intentionally kept outside the dataset folder, following standard ML best practices.

📦 Dataset

~100 sonar images

PNG format (RGB, false-color sonar colormap)

Images may contain zero or more objects

Images without objects contain empty YOLO label files

YOLO Label Format
0 x_center y_center width height


All labels are normalized and use a single class (0: object).

⚙️ YOLO Training

Model: yolov8n.pt

Classes: 1 (object)

Image size: 640 × 640

Epochs: 200

Batch size: 8

Device: CPU

Example command:

yolo detect train model=yolov8n.pt data=configs/data.yaml imgsz=640 epochs=200 batch=8 device=cpu name=obj_base200

🔍 Hybrid Intensity Classification
Key Idea

Instead of relying on peak intensity (which is unstable in sonar due to speckle and colormap saturation), the system measures dominant red reflection density, which correlates better with strong acoustic returns.

Red Excess Definition
red_excess=max(0,𝑅−max(𝐺,𝐵))
red_excess=max(0,R−max(G,B))
Final Score
score=0.7⋅mean(red_excess)+0.3⋅frac_strong
score=0.7⋅mean(red_excess)+0.3⋅frac_strong

Where:

mean(red_excess) measures overall acoustic energy

frac_strong measures the density of strong reflections

🧪 Hybrid Inference (Final Configuration)
python src/infer_hybrid.py \
  --weights models/obj/best_obj_base200.pt \
  --source data/images/val \
  --out outputs/pred_hybrid \
  --conf 0.25 \
  --iou 0.5 \
  --int_mode red_excess \
  --score mean_frac \
  --no_clip \
  --strong_thr 0.20 \
  --w_mean 0.7 \
  --w_frac 0.3 \
  --t_low 0.02 \
  --t_high 0.025

Output

Annotated images with final class labels

CSV file with:

bounding box coordinates

YOLO confidence

intensity score

final reflection class

✅ Results Summary

YOLO reliably detects sonar objects (mAP@0.5 ≈ 0.57)

Hybrid classifier successfully separates:

weak reflections (low)

intermediate returns (medium)

strong, dense reflections (high)

The final system avoids systematic bias toward high-intensity classes

📊 Key Advantages

Robust to sonar noise and speckle

Avoids class confusion inherent to multi-class YOLO training

Physically interpretable classification

Easily adjustable thresholds

Suitable for academic and engineering contexts

📅 Project Status

✔️ Architecture finalized

✔️ YOLO model trained and validated

✔️ Hybrid inference stabilized and calibrated

🔄 Quantitative analysis and baseline comparison ongoing

👤 Author

António
University Project — Engineering / Robotics / Computer Vision

📄 License

This project is developed for academic purposes.
