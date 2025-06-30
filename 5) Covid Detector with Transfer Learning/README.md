# COVID-19 Detection from Chest X-ray Images using VGG16

This project implements a binary image classification model to detect COVID-19 from chest X-ray images. It uses a pre-trained **VGG16** model as the feature extractor and adds custom classification layers on top.

## 📌 Features

- Loads and preprocesses chest X-ray images from a labeled dataset
- Uses transfer learning with **VGG16** (frozen base)
- Trains a classifier to distinguish between **COVID** and **non-COVID** cases
- Visualizes training and validation metrics
- Uses data augmentation to improve generalization

---

## 🧠 Model Architecture

- Base: VGG16 (pre-trained on ImageNet, no top layers)
- Custom layers:
  - MaxPooling2D
  - Flatten
  - Dense(64, ReLU)
  - Dense(2, Softmax)

---

## 📁 Project Structure

covid19-vgg16/
│
├── covid19-dataset/ # Dataset organized by class folders
│ ├── covid/
│ └── normal/
│
├── vgg16_weights.h5 # Pretrained VGG16 weights
├── train.py # Main script for loading data, training, and plotting
├── README.md # Project documentation


---

## 🔧 Requirements

- Python 3.x
- TensorFlow (>= 2.x)
- NumPy
- OpenCV
- scikit-learn
- matplotlib

Install them via pip:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy

🚀 How to Run
1. Make sure you have the dataset in the correct structure (e.g. covid19-dataset/covid and covid19-dataset/normal)

2. Download and place vgg16_weights.h5 in the correct path (or modify train.py to load VGG16 from Keras)

3. Run the training script:
python train.py

This will: 
Load and preprocess the data /
Train the model /
Plot accuracy and loss curves /

📈 Sample Output (Training Curve)
The script will generate a plot showing training/validation accuracy and loss to help assess model performance.

🙋‍♀️ Author
Zahra Sheikhvand
GitHub: zahra-she