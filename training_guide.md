# VisionForge: AI Training Guide

This guide explains how to use the data collected via the "Add to Dataset" feature to train a custom Machine Learning model (like YOLOv8 or a simple ResNet) for CNC inspection.

## 1. Data Structure
When you click "Add to Dataset", the following files are created in your project directory:
- `dataset/images/[UUID].jpg`: The raw image captured from the camera or generator.
- `dataset/labels/[UUID].json`: A metadata file containing:
    - Detected dimensions (W, H, Hole D) in mm.
    - PASS/FAIL status.
    - Hole coordinates and count.

## 2. Preparing for AI Training (YOLO Format)
To train a model like YOLOv8, you need to convert the `.json` measurements into bounding box coordinates (`xywh` format).

### Setup Environment
```bash
pip install ultralytics opencv-python
```

### Conversion Script Example
You can use a simple Python script to read the `.json` files and create YOLO-compliant `.txt` labels:
```python
import json
import os

# Your image is 1280x720 (Live Scan) or 500x500 (Generated)
# Normalized YOLO format: [class] [x_center] [y_center] [width] [height]
```

## 3. Training the Model
Once labels are converted, you can start training:
```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Train on your collected dataset
model.train(data='cnc_data.yaml', epochs=50, imgsz=640)
```

## 4. Why Train?
- **Robustness**: AI can learn to ignore glare, oil spots, or shadows that confuse traditional Edge Detection.
- **Classification**: Train the model to recognize specific types of defects (cracks, burrs, missing holes) beyond simple measurements.
- **Speed**: Once trained, inference is extremely fast on modern hardware.

> [!TIP]
> Collect at least **100 samples** for both PASS and FAIL cases before starting your first training run for best results.
