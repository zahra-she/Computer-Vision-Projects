import cv2
import glob
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from mtcnn import MTCNN

"""
Face Detection and Classification with MTCNN and CNN

This script detects faces from images in a dataset using MTCNN,
preprocesses and labels them, then trains a CNN model to classify
smiling vs non-smiling faces.

Steps:
- Load images and detect faces using MTCNN.
- Resize and normalize face images.
- Encode labels and split data into train/test sets.
- Define and train a CNN model for binary classification.
- Plot training accuracy and loss.
- Save the trained model.
"""

detector = MTCNN()

def detect_faces(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = detector.detect_faces(rgb_img)[0]
    x, y, w, h = out["box"]
    return rgb_img[y:y+h, x:x+w]

all_faces = []
all_labels = []

for i, item in enumerate(glob.glob("smile_dataset\\*\\*")):
    img = cv2.imread(item)
    try:
        face = detect_faces(img)
        face = cv2.resize(face, (32, 32))
        face = face/255.0
        all_faces.append(face)

        label = item.split("\\")[-2]
        all_labels.append(label)
    except:
        pass
    if i % 100 == 0:
        print("[INFO] {}/4000 processed".format(i))

all_faces = np.array(all_faces)

le = LabelEncoder()
all_labels_le = le.fit_transform(all_labels)
all_labels_le = to_categorical(all_labels_le)

trainX, testX, trainy, testy = train_test_split(all_faces, all_labels_le, test_size=0.2, random_state=42)

net = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.BatchNormalization(),
    layers.Dense(2, activation="softmax")
])

net.compile(optimizer="SGD",
            metrics=["accuracy"],
            loss="categorical_crossentropy")

h = net.fit(trainX, trainy, batch_size=32, epochs=25,
            validation_data=(testX, testy))

plt.plot(h.history["accuracy"], label="train accuracy")
plt.plot(h.history["val_accuracy"], label="test accuracy")
plt.plot(h.history["loss"], label="train loss")
plt.plot(h.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy/loss")
plt.title("face detection")
plt.show()

net.save("face.h5")