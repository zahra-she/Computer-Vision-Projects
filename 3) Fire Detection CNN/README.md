🧪 How to Train the CNN Model:
The training script is located in cnn.py. This script loads and preprocesses images, builds a Convolutional Neural Network (CNN), trains it on the dataset, evaluates it, and saves the final model.

✅ Steps to Train:
Make sure your dataset is in the following structure:

Copy
Edit
fire_dataset/
├── fire_images/
│   ├── fire (1).jpg
│   ├── ...
├── non_fire_images/
│   ├── non_fire (1).jpg
│   ├── ...
Install required packages (if not already installed):
pip install numpy opencv-python tensorflow scikit-learn matplotlib

Run the training script:

python cnn.py


After training, the model will be saved as:
cnn.h5
🧠 What it does:
Resizes all images to 32x32 pixels.

Applies one-hot encoding to labels.

Trains a simple CNN model with two convolutional layers.

Saves the trained model for later use.
--------------------------------------------------------------------

🔍 How to Use the Trained Model
After training and saving the model (cnn.h5), you can test it on new images to detect fire.

📌 Steps:
Place the image you want to test (e.g., nature.jpg) in the project folder.

Run the prediction script:
python use_cnn_model.py
Make sure the filename in the script matches your test image.

🧠 What it does:
Loads the trained CNN model.

Preprocesses the input image (resize to 32x32, normalize).

Predicts whether the image contains fire or not.

Displays the image with a label and confidence score overlaid.