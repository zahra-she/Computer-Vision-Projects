"""
Digit Classification with CNN

This script implements a Convolutional Neural Network (CNN) for classifying digit images
(0–9) from a labeled dataset. It performs the following steps:

1. Loads images from subfolders using OpenCV and glob.
2. Preprocesses the images: resize to 32x32, normalize pixel values.
3. Extracts class labels from folder names and encodes them using one-hot encoding.
4. Splits the dataset into training and testing sets.
5. Builds a compact CNN architecture using TensorFlow Keras.
6. Trains the model on the training data and validates on test data.
7. Plots training/validation accuracy and loss curves.
8. Saves the trained model as a `.h5` file.

The dataset is expected to be structured like:
kapcha/
├── 0/
│   ├── image1.jpg
├── 1/
│   ├── image2.jpg
...
├── 9/
│   ├── imageN.jpg
"""

import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def load_data_preprocess(dataset):
    all_images = []
    all_labels = []

    for i, item in enumerate(glob.glob(dataset)):
        img = cv2.imread(item)
        img = cv2.resize(img, (32, 32))
        img = img / 255.0
        all_images.append(img)

        label = item.split("\\")[-2]
        all_labels.append(label)

        if i % 100 == 0:
            print("[INFO] {}/2000 processed".format(i))

    all_images = np.array(all_images)

    lb = LabelBinarizer()
    all_labels = lb.fit_transform(all_labels)

    trainX, testX, trainy, testy = train_test_split(all_images, all_labels, test_size=0.2)

    return trainX, testX, trainy, testy


def miniCNN():
    net = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(9, activation="softmax")
    ])

    net.compile(loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])
    return net


def show_result_curve(h):
    plt.plot(h.history["accuracy"], label="train accuracy")
    plt.plot(h.history["val_accuracy"], label="test accuracy")
    plt.plot(h.history["loss"], label="train loss")
    plt.plot(h.history["val_loss"], label="test loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy/loss")
    plt.title("Digit Classification")
    plt.show()


trainX, testX, trainy, testy = load_data_preprocess("kapcha\\*\\*")
net = miniCNN()
h = net.fit(x=trainX, y=trainy, epochs=20, batch_size=32, validation_data=(testX, testy))
show_result_curve(h)

net.save("digit_classifier.h5")