# 🚀 FastRCNN Object Detection with EfficientDet

This project demonstrates how to perform object detection using the pre-trained **EfficientDet-D1** model from the TensorFlow Object Detection API. Built and tested in **Google Colab**, it leverages **Google Drive** for easy model storage and image input/output.

---

## 🧠 Features

- Uses **EfficientDet-D1**, trained on the COCO dataset
- Colab-friendly: easy setup and integration with Google Drive
- Clean and modular code for loading models, preparing images, and running inference
- Visual output with bounding boxes, class labels, and scores

---

## 📂 Project Structure
DLprojects/
├── models/ # TensorFlow Models repository
│ └── research/
│ └── object_detection/
├── efficientdet/
│ └── efficientdet_d1_coco17_tpu-32/
│ └── saved_model/
├── SSD/
│ └── Cristiano.jpg # Test image
├── detect.py # Main detection script
---

## ⚙️ Setup Instructions (in Google Colab)

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

2. Clone TensorFlow Models Repository
!git clone https://github.com/tensorflow/models.git

3. Download and Extract EfficientDet Weights
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz
!mkdir -p efficientdet
!tar -xf efficientdet_d1_coco17_tpu-32.tar.gz -C efficientdet/

4. Install Required Packages
!pip install protobuf==3.20.3

5. Compile Protobufs
%cd models/research
!protoc object_detection/protos/*.proto --python_out=.
------------------------------------------------------------------
🧪 Run Inference
# Load image
img, r_img = load_data("/path/to/your/image.jpg")

# Load the model
detection_model = load_model()

# Run inference
output_dict = inference(r_img, detection_model)

# Visualize results
visualize(output_dict, img)



