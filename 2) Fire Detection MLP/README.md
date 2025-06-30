# Fire Detection MLP

This project implements a simple Multi-Layer Perceptron (MLP) model to classify images into fire and non-fire categories using TensorFlow Keras.

## Dataset

- Images are loaded from the `fire_dataset` folder.
- Images are resized to 32x32 pixels and flattened.
- Labels are extracted from folder names.

## Features

- Image preprocessing and normalization
- Label encoding (integer + one-hot)
- Model definition, training, evaluation, and accuracy plotting
- Model saving and loading

## Usage

1. Clone the repository.
2. Install required packages:
   ```bash
   pip install numpy opencv-python tensorflow scikit-learn matplotlib

run ----> fire_detection_mlp_changed.py

run ----> use_fire_mlp_model.py