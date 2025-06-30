# MNIST GAN: Generative Adversarial Network for Handwritten Digits

This repository contains an implementation of a Generative Adversarial Network (GAN) to generate realistic handwritten digit images similar to the MNIST dataset.

---

## Project Overview

GANs consist of two neural networks — a **Generator** and a **Discriminator** — competing against each other. The Generator creates fake images from random noise aiming to fool the Discriminator, while the Discriminator tries to distinguish between real and fake images. Over time, both models improve, resulting in a Generator capable of producing highly realistic images.

This project uses TensorFlow/Keras to build and train the GAN on the MNIST handwritten digit dataset.

---

## Features

- **Generator Network**: Upsamples random noise vectors into 28x28 grayscale images.
- **Discriminator Network**: Classifies images as real or fake.
- **Training Loop**: Alternates training of Discriminator and Generator for stable GAN training.
- **Performance Monitoring**: Saves generated image grids and generator model checkpoints every 10 epochs.
- **Visualization**: Automatically saves generated digit image grids during training.

---

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the required packages with:

```bash
pip install tensorflow numpy matplotlib

Usage
1. Clone this repository:
git clone https://github.com/yourusername/mnist-gan.git
cd mnist-gan

2. Run the training script:
python gan_mnist.py
The script will train the GAN for 100 epochs (by default), saving generated images and model checkpoints periodically.
-------------------------------------------------------------------------------------------------------------
File Structure:
gan_mnist.py: Main script containing the GAN model definitions and training loop.

generated_plot_epoch_XXX.png: Saved image grids of generated digits during training.

generator_model_epoch_XXX.h5: Saved generator model checkpoints.
-------------------------------------------------------------------------------------------------------------
How It Works
Generator: Takes a 100-dimensional random noise vector and upsamples it through dense and transpose convolutional layers to produce a 28x28 image.

Discriminator: A convolutional classifier that outputs the probability that the input image is real.


Training:

Generate batches of real images and corresponding labels (1).

Generate batches of fake images from noise and corresponding labels (0).

Train Discriminator on real + fake batches.

Train Generator (via GAN model) with the goal of fooling the Discriminator.
-------------------------------------------------------------------------------------------------------------
Results
During training, image grids illustrating generated digits are saved as generated_plot_epoch_XXX.png. The generator progressively improves, producing clearer and more realistic digits.