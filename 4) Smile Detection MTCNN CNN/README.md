Smile Detection using MTCNN and CNN
This project implements a smile detection system by combining face detection with MTCNN and classification using a Convolutional Neural Network (CNN).

Features:
Detects faces from images using the MTCNN face detector.

Classifies detected faces into "smile" or "not smile" using a CNN.

Trains the model on a labeled dataset of smiling and non-smiling faces.

Visualizes training accuracy and loss.

Predicts and visualizes smile detection results on new images.

Files:
FaceDetection_MTCNN.py: Detects faces from dataset images, preprocesses them, trains a CNN model for smile classification, and saves the model.

smile_detection.py: Loads the trained model to predict smile presence on new images, displays the result with bounding boxes and labels.

Requirements:
Python 3.x

TensorFlow / Keras

OpenCV

MTCNN

scikit-learn

matplotlib

numpy

Usage:
1. Train the model by running:
python FaceDetection_MTCNN.py
2. Use the trained model to predict smiles on new images:
python smile_detection.py
------------------------------------------------------------------
Dataset:
The model expects a dataset organized in subfolders named by label, e.g.:
smile_dataset/
    smile/
    not_smile/

