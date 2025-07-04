# YOLOv8 Vision Tasks with Ultralytics

This Jupyter Notebook demonstrates how to use the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model for various computer vision tasks including:

- 🟩 Object Detection
- 🟥 Segmentation
- 🔵 Classification
- 🟡 Pose Estimation
- 🟣 Object Tracking

---

## 📦 Requirements

Make sure to have Python ≥ 3.8 installed. Then install the required packages:

```bash
pip install -U ultralytics

🚀 Tasks Demonstrated

1. Object Detection
Loads pre-trained YOLOv8 models (yolov8n.pt, yolov8x.pt) to detect objects in images using:
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("path/to/image.jpg")
results.show()

2. Image Segmentation
Applies segmentation masks over detected objects.

3. Image Classification
Classifies input images using models trained on the ImageNet dataset (1000 classes).

4. Pose Estimation
Detects human body keypoints.

5. Object Tracking
Uses object detection in sequence to track movement across frames/videos.
-------------------------------------------------------------------------------------------------------
🗂 File Structure
use_yolo_v8.ipynb: Main notebook demonstrating each task.

Sample images/videos used for demo (if any, add to a /data folder).