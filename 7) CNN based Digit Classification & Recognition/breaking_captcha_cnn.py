"""
Digit Detection and Recognition from Image using a Trained CNN Model

This script reads a color image containing multiple handwritten or printed digits,
detects each digit using contour detection, preprocesses the digit regions, and
classifies them using a pre-trained CNN model.

Steps:
1. Load a trained digit classifier model (`digit_classifier.h5`).
2. Convert the input image to grayscale.
3. Apply binary thresholding to prepare for contour detection.
4. Detect external contours corresponding to digit regions.
5. For each contour:
   - Extract bounding box and crop the region of interest (ROI).
   - Resize and normalize the ROI.
   - Predict the digit using the CNN model.
   - Draw a green rectangle and predicted digit label on the original image.
6. Display the final image with annotations.
"""

import cv2
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model

# Load the trained CNN model
net = load_model("digit_classifier.h5")

# Load the test image and convert to grayscale
img = cv2.imread("cap1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary inverse thresholding
_, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

# Detect external contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Process each detected contour
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    roi = img[y-5:y+h+5, x-5:x+w+5]

    roi = cv2.resize(roi, (32, 32))
    roi = roi / 255.0
    roi = np.array([roi])

    output = net.predict(roi)[0]
    max_index = np.argmax(output) + 1
    print(max_index)

    cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2)
    cv2.putText(img, str(max_index), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()