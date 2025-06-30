"""
Fire Image Prediction using Trained CNN Model

This script loads a pre-trained Convolutional Neural Network (CNN) model
and uses it to classify a new image as either 'Fire' or 'Non Fire'.

Steps:
1. Load the trained model from file.
2. Load a test image and resize it to 32x32.
3. Normalize the image and prepare it for prediction.
4. Use the model to predict the probability of each class.
5. Display the predicted label and confidence on the image using OpenCV.
"""

import numpy as np
import cv2 
from tensorflow.keras import models
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load the trained CNN model
net = models.load_model("cnn.h5")

# Load and preprocess the test image
img = cv2.imread("nature.jpg")
r_img = cv2.resize(img, (32, 32))
r_img = r_img / 255.0
r_img = np.array([r_img])

# Perform prediction
output = net.predict(r_img)[0]
max_output = np.argmax(output)

category_name = ["Fire", "None Fire"]
text = "{}: {:.2f} %".format(category_name[max_output], output[max_output] * 100)

# Display prediction on image
cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()