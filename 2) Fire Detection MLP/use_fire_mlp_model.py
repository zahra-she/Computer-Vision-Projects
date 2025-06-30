"""
This script loads a pre-trained MLP model to classify an input image as 
'Fire' or 'Non Fire'. It preprocesses the image, performs prediction, 
and displays the image with the predicted label and confidence percentage.
"""

import numpy as np
import cv2
from tensorflow.keras import models
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

net = models.load_model("mlp_advance.h5")

# Load and preprocess test image
img = cv2.imread("nature.jpg")
r_img = cv2.resize(img, (32, 32)).flatten()
r_img = r_img / 255.0
r_img = np.array([r_img])  # Shape: (1, 3072)

# Predict the class probabilities
output = net.predict(r_img)[0]
max_output = np.argmax(output)

category_name = ["Fire", "None Fire"]
text = "{}: {:.2f} %".format(category_name[max_output], output[max_output] * 100)

# Display the image with prediction text
cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()