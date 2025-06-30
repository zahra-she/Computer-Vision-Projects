# 🧠 Digit Classification with CNN

This project implements a simple Convolutional Neural Network (CNN) to classify digit images (0–9) using TensorFlow and Keras. The model is trained on a labeled dataset where each digit is stored in its respective folder.

---

## 📁 Dataset Structure

The dataset should be organized like this:

kapcha/
├── 0/
│ ├── img1.jpg
│ ├── ...
├── 1/
│ ├── img2.jpg
│ ├── ...
...
├── 9/
│ ├── imgN.jpg


Each subfolder name represents the digit label for the images inside it.

---

## ⚙️ Requirements

Install the required Python packages using:

pip install numpy opencv-python tensorflow scikit-learn matplotlib

1. Place your dataset inside a folder named kapcha/.

2. Run the training script:
python digit_classification_with_cnn.py

3. The model will:

Load and preprocess the images

One-hot encode the labels

Train a small CNN

Plot training & validation curves

Save the model as digit_classifier.h5
--------------------------------------------------------------------------

🧪 Model Architecture
Conv2D → MaxPool → Conv2D → MaxPool → Flatten → Dense → Dense (Softmax)

Optimizer: SGD

Loss Function: Categorical Crossentropy

Output: One of 10 digits (0–9)

📉 Sample Output Plot
The script generates a plot showing both accuracy and loss over training epochs for training and validation sets.

💾 Saved Model
The trained model is saved in HDF5 format: digit_classifier.h5

You can later load it using: 

from tensorflow.keras.models import load_model

model = load_model("digit_classifier.h5")
--------------------------------------------------------------------------
Digit Detection in Images Using Pre-trained CNN:

This project includes a script that detects multiple digits in a given image, extracts each digit, preprocesses them, and recognizes them using a trained Convolutional Neural Network (CNN) model.

How to Use the Digit Detection Script:
Prepare the pre-trained model
Ensure you have trained and saved the digit classification model as digit_classifier.h5. If not, run the training script first.

Place the input image
Put the image containing digits (e.g., cap1.jpg) in your working directory or specify the correct path in the script.

Run the digit detection script
The script will:

Load the image and convert it to grayscale.

Apply binary thresholding to separate digits from the background.

Find contours corresponding to each digit.

Crop each digit region, resize it to 32x32 pixels, and normalize pixel values.

Predict the digit using the CNN model.

Draw bounding boxes and predicted digit labels on the original image.

Display the annotated image in a window.

Example output
The script prints predicted digits to the console and shows the image with rectangles and digit labels.

Usage Example: 
python digit_detection.py
--------------------------------------------------------------------------
Requirements:

Python 3.x

OpenCV (cv2)

TensorFlow / Keras

NumPy

Install required packages via:
pip install opencv-python tensorflow numpy
--------------------------------------------------------------------------
Notes:
The detection threshold value (110) in the script can be adjusted based on your image lighting conditions.

Contours are detected using external contours only, so overlapping digits or touching digits might need more advanced methods.

Model input size is fixed at 32x32 pixels, matching training data dimensions.