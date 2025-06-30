"""
This script performs image classification on a fire dataset using a simple 
multi-layer perceptron (MLP) neural network built with TensorFlow Keras.

Steps:
1. Load images from directories, resize to 32x32, and flatten them into vectors.
2. Extract labels from folder names.
3. Encode labels using integer encoding and one-hot encoding.
4. Normalize image data.
5. Split dataset into training and test sets.
6. Define, compile, and train an MLP model.
7. Evaluate model performance on the test set.
8. Plot training and validation accuracy.
9. Save the trained model to a file.
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
plt.style.use("ggplot")

data = []
labels = [] 
i = 0

for item in glob.glob("fire_dataset\\*\\*"):  # Loop through all image files in subdirectories
    i += 1    
    img = cv2.imread(item)
    img = cv2.resize(img, (32,32))
    img = img.flatten()  # Flatten image to 1D vector
    label = item.split("\\")[-2]  # Extract label from folder name

    data.append(img)
    labels.append(label) 

    if i % 100 == 0:   
        print(f"[INFO] {i}/300 processed")

le = LabelEncoder()
labels = le.fit_transform(labels)  # Integer encoding

labels = to_categorical(labels)   # One-hot encoding

data = np.array(data) / 255.0     # Normalize pixel values to [0,1]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

net = models.Sequential([
    layers.Dense(300, activation="relu", input_dim=3072),
    layers.Dense(40, activation="relu"),
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
plt.title("Fire dataset classification")
plt.show()

# Save the trained model
net.save("mlp_advance.h5")

# To load the model in another script, use:
# net = models.load_model("mlp_advance.h5")
