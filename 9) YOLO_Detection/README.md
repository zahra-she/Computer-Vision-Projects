# YOLOv3 Object Detection with OpenCV

This project demonstrates how to perform object detection using the YOLOv3 deep learning model with OpenCV's DNN module. It processes a single image, detects objects, and visualizes the results.
---

## 📷 Overview

This repository uses the YOLOv3 (You Only Look Once version 3) object detection algorithm trained on the COCO dataset to detect multiple classes of objects (e.g., people, animals, vehicles) in images.

The implementation includes:

- Loading an image and preprocessing it
- Running inference using a pretrained YOLOv3 model
- Post-processing to extract bounding boxes and filter predictions
- Drawing detected objects on the original image
---

## 🧠 YOLOv3 Architecture (Simplified Explanation)

YOLOv3 is a real-time object detection model that divides an image into a grid and simultaneously predicts bounding boxes and class probabilities.

**Key features of YOLOv3:**

- **Backbone:** Uses **Darknet-53**, a convolutional neural network with residual connections and 53 convolutional layers.
- **Feature Pyramid Network (FPN):** YOLOv3 makes predictions at **three different scales**, which helps detect objects of varying sizes:
  - Large objects → predictions from deep layers
  - Medium objects → intermediate layers
  - Small objects → shallow layers
- **Anchors:** Predefined bounding boxes used to detect objects with different aspect ratios.
- **Output:** For each scale, YOLO outputs a 3D tensor containing bounding box coordinates, objectness scores, and class probabilities.

YOLOv3 is known for its balance between speed and accuracy.
---

## 📁 Directory Structure

project/
├── yolo_files/
│ ├── yolov3.cfg # Model architecture config file
│ ├── yolov3.weights # Pretrained weights file
│ └── coco.names # Class labels (80 classes)
├── images/
│ └── cat.jpg # Sample image
├── main.py # Main Python script
└── README.md # Project documentation
---

## ✅ Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

📥 Download YOLOv3 Weights  :
To use YOLOv3, download the pretrained weights file from the official source:  
[yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

### 🔧 Installation

pip install opencv-python numpy

🚀 Usage:
Download YOLOv3 files:

yolov3.cfg: Download

yolov3.weights: Download (248MB)

coco.names: Download

Place the files in yolo_files/ directory.

Add an image to the images/ folder (e.g., cat.jpg).

Run the script:
python main.py

📦 Output Example
When the image is processed, objects detected (e.g., "cat", "person", "car") will be highlighted with green bounding boxes and labeled with confidence scores.

🙌 Credits
Joseph Redmon – Creator of YOLO

Darknet – Original YOLO implementation

OpenCV DNN module – for deep learning inference