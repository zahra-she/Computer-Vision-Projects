"""
Fire Detection using Convolutional Neural Network (CNN)

This script implements a basic image classification model using a Convolutional Neural Network (CNN)
built with TensorFlow Keras. The model is trained to classify images as 'fire' or 'non-fire'.

Steps:
1. Load images from the 'fire_dataset' directory and resize them to 32x32.
2. Extract labels from folder names and encode them.
3. Normalize image pixel values.
4. Split the dataset into training and testing sets.
5. Define and compile a CNN model.
6. Train the model and evaluate its accuracy.
7. Plot training and validation accuracy.
8. Save the trained model to a file.
"""

import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os
import matplotlib.pyplot as plt
plt.style.use("ggplot")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

data = []
labels = [] 
i = 0

for item in glob.glob("fire_dataset\\*\\*"):
    i += 1    
    img = cv2.imread(item)
    img = cv2.resize(img, (32, 32))
    label = item.split("\\")[-2]

    data.append(img)
    labels.append(label) 

    if i % 100 == 0:   
        print(f"[INFO] {i}/300 processed")

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data) / 255.0

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

net = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPool2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(100, activation="relu"),
    layers.Dense(2, activation="sigmoid")
])

net.compile(optimizer="SGD",
            loss="binary_crossentropy",
            metrics=["accuracy"])

h = net.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

loss, acc = net.evaluate(X_test, y_test)
print(f"loss: {loss}, accuracy: {acc}")

plt.plot(h.history["accuracy"], label="train")
plt.plot(h.history["val_accuracy"], label="test")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Fire Dataset Classification")
plt.show()

# Save the trained model
net.save("cnn.h5")

# To load the model in another script:
# net = models.load_model("cnn.h5")
