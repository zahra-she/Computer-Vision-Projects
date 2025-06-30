import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from mtcnn import MTCNN

"""
Smile Detection on Face Images

This script loads a pre-trained CNN model to classify whether a detected
face is smiling or not. It uses MTCNN for face detection and OpenCV for
image processing and display.

Steps:
- Detect face from input image using MTCNN.
- Preprocess face for model input.
- Predict smile probability using the loaded model.
- Display the result with bounding box and label.
"""

detector = MTCNN()

smile_net = load_model("face.h5")
labels = ["not smile", "smile"]
colors = [(0, 0, 255), (0, 255, 0)]

def detect_faces(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = detector.detect_faces(rgb_img)[0]
    x, y, w, h = out["box"]
    return rgb_img[y:y+h, x:x+w], (x, y, w, h)

def preprocess(face):
    face = cv2.resize(face, (32, 32))
    face = face / 255.0
    face = np.array([face])
    return face

img = cv2.imread("dicaprio.jpg")
face, (x, y, w, h) = detect_faces(img)
normalized_face = preprocess(face)
out = smile_net.predict(normalized_face)[0]
max_index = np.argmax(out)

predict = labels[max_index]
probability = out[max_index] * 100
text = f"{predict}: {probability:.2f}%"

cv2.rectangle(img, (x, y), (x+w, y+h), colors[max_index], 2)
cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[max_index], 2)

cv2.imshow("face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
