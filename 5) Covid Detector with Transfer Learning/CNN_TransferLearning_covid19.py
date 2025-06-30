import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Use a clean plot style
plt.style.use("ggplot")

# Suppress TensorFlow warnings unless they are critical
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_data_preprocess(dataset_path_pattern):
    """
    Loads image data from the given dataset path pattern, preprocesses them,
    and splits into train/test sets.

    Parameters:
    - dataset_path_pattern (str): Glob pattern to load image files.

    Returns:
    - trainX, testX: Arrays of training and testing image data.
    - trainy, testy: One-hot encoded labels for training and testing sets.
    """
    all_images = []
    all_labels = []

    for item in glob.glob(dataset_path_pattern):
        img = cv2.imread(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        all_images.append(img)

        label = item.split("\\")[-2]
        all_labels.append(label)

    # Encode labels to 0 and 1 and convert to categorical
    lb = LabelEncoder()
    all_labels = lb.fit_transform(all_labels)
    all_labels = to_categorical(all_labels, 2)

    # Normalize image data
    all_images = np.array(all_images) / 255.0

    # Split into train and test sets
    trainX, testX, trainy, testy = train_test_split(
        all_images, all_labels, test_size=0.15, random_state=42)

    return trainX, testX, trainy, testy


def build_model():
    """
    Builds and compiles a CNN model using VGG16 as a base.

    Returns:
    - Compiled Keras model.
    """
    baseModel = VGG16(weights="imagenet",
                      include_top=False,
                      input_tensor=layers.Input(shape=(224, 224, 3)))
#If you want to download the weights manually, here is the official link:
#https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5


    # Freeze VGG16 layers
    for layer in baseModel.layers:
        layer.trainable = False

    # Build the full model
    net = models.Sequential([
        baseModel,
        layers.MaxPooling2D((4, 4)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="softmax")
    ])

    # Compile the model with Adam optimizer
    opt = Adam(learning_rate=0.001, decay=0.001 / 25)
    net.compile(loss="binary_crossentropy",
                optimizer=opt,
                metrics=["accuracy"])
    return net


def show_result_curve(h):
    """
    Plots the training and validation accuracy/loss curves.

    Parameters:
    - h: History object returned from model.fit()
    """
    plt.plot(np.arange(len(h.history["accuracy"])), h.history["accuracy"], label="Train Accuracy")
    plt.plot(np.arange(len(h.history["val_accuracy"])), h.history["val_accuracy"], label="Validation Accuracy")
    plt.plot(np.arange(len(h.history["loss"])), h.history["loss"], label="Train Loss")
    plt.plot(np.arange(len(h.history["val_loss"])), h.history["val_loss"], label="Validation Loss")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy / Loss")
    plt.title("Training Curve - COVID-19 Detection")
    plt.show()


# Load and preprocess data
trainX, testX, trainy, testy = load_data_preprocess(r"covid19-dataset\covid19-dataset\dataset\*\*.*")

# Build model
net = build_model()

# Create data augmentation generator
aug = ImageDataGenerator(rotation_range=10, fill_mode="nearest")

# Train the model
h = net.fit(
    aug.flow(trainX, trainy, batch_size=8),
    steps_per_epoch=len(trainX) // 8,
    validation_data=(testX, testy)
)

# Show training results
show_result_curve(h)
