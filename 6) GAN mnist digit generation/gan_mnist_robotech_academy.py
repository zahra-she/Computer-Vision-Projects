import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings except critical errors

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def define_generator(latent_dim=100):
    """
    Define the generator model.

    Args:
        latent_dim (int): Dimension of the latent space (noise vector).

    Returns:
        keras.Sequential: Generator model.
    """
    nodes = 128 * 7 * 7

    model = models.Sequential([
        layers.Dense(nodes, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same'),
    ])

    return model

def define_discriminator(input_shape=(28, 28, 1)):
    """
    Define the discriminator model.

    Args:
        input_shape (tuple): Shape of input images.

    Returns:
        keras.Sequential: Discriminator model.
    """
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def define_gan(generator, discriminator):
    """
    Define the combined GAN model.

    Args:
        generator (keras.Model): Generator model.
        discriminator (keras.Model): Discriminator model.

    Returns:
        keras.Sequential: GAN model where discriminator is not trainable.
    """
    discriminator.trainable = False

    model = models.Sequential([
        generator,
        discriminator
    ])

    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

def load_real_samples():
    """
    Load and preprocess the MNIST dataset.

    Returns:
        numpy.ndarray: Normalized training images with shape (num_samples, 28, 28, 1).
    """
    (train_images, _), (_, _) = mnist.load_data()
    images = np.reshape(train_images, (len(train_images), 28, 28, 1))
    images = images.astype('float32') / 255.0
    return images

def generate_real_samples(dataset, sample_size):
    """
    Randomly select real samples from the dataset.

    Args:
        dataset (numpy.ndarray): Dataset of real images.
        sample_size (int): Number of samples to select.

    Returns:
        tuple: Selected images and their labels (all ones).
    """
    indices = np.random.randint(0, dataset.shape[0], sample_size)
    selected_images = dataset[indices]
    labels = np.ones((sample_size, 1))
    return selected_images, labels

def generate_latent_points(latent_dim, sample_size):
    """
    Generate random points in the latent space as input for the generator.

    Args:
        latent_dim (int): Dimension of the latent space.
        sample_size (int): Number of samples to generate.

    Returns:
        numpy.ndarray: Random latent points with shape (sample_size, latent_dim).
    """
    latent_points = np.random.randn(latent_dim * sample_size)
    latent_points = latent_points.reshape(sample_size, latent_dim)
    return latent_points

def generate_fake_samples(generator, latent_dim, sample_size):
    """
    Generate fake samples using the generator model.

    Args:
        generator (keras.Model): Generator model.
        latent_dim (int): Dimension of the latent space.
        sample_size (int): Number of samples to generate.

    Returns:
        tuple: Generated images and their labels (all zeros).
    """
    latent_points = generate_latent_points(latent_dim, sample_size)
    generated_images = generator.predict(latent_points)
    labels = np.zeros((sample_size, 1))
    return generated_images, labels

def save_generated_plot(examples, epoch_num, grid_size=10):
    """
    Save a plot of generated images as a grid.

    Args:
        examples (numpy.ndarray): Generated images.
        epoch_num (int): Current epoch number.
        grid_size (int): Grid size (grid_size x grid_size images).
    """
    for i in range(grid_size * grid_size):
        plt.subplot(grid_size, grid_size, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')

    filename = f'generated_plot_epoch_{epoch_num+1:03d}.png'
    plt.savefig(filename)
    plt.close()

def summarize_performance(epoch_num, generator, discriminator, dataset, latent_dim, sample_size=100):
    """
    Evaluate and summarize model performance, save generated images and generator model.

    Args:
        epoch_num (int): Current epoch number.
        generator (keras.Model): Generator model.
        discriminator (keras.Model): Discriminator model.
        dataset (numpy.ndarray): Real samples dataset.
        latent_dim (int): Latent space dimension.
        sample_size (int): Number of samples for evaluation.
    """
    real_images, real_labels = generate_real_samples(dataset, sample_size)
    _, real_acc = discriminator.evaluate(real_images, real_labels, verbose=0)
    fake_images, fake_labels = generate_fake_samples(generator, latent_dim, sample_size)
    _, fake_acc = discriminator.evaluate(fake_images, fake_labels, verbose=0)
    print(f'>Epoch {epoch_num+1}, Real Accuracy: {real_acc*100:.2f}%, Fake Accuracy: {fake_acc*100:.2f}%')
    save_generated_plot(fake_images, epoch_num)
    filename = f'generator_model_epoch_{epoch_num + 1}.h5'
    generator.save(filename)

def train_gan(generator, discriminator, gan_model, dataset, latent_dim, epochs=100, batch_size=256):
    """
    Train the GAN model.

    Args:
        generator (keras.Model): Generator model.
        discriminator (keras.Model): Discriminator model.
        gan_model (keras.Model): Combined GAN model.
        dataset (numpy.ndarray): Real samples dataset.
        latent_dim (int): Latent space dimension.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
    """
    batches_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)

    for epoch in range(epochs):
        for batch in range(batches_per_epoch):
            # Select real and fake samples
            real_imgs, real_lbls = generate_real_samples(dataset, half_batch)
            fake_imgs, fake_lbls = generate_fake_samples(generator, latent_dim, half_batch)

            # Combine real and fake samples
            X_batch, y_batch = np.vstack((real_imgs, fake_imgs)), np.vstack((real_lbls, fake_lbls))

            # Train discriminator
            d_loss, _ = discriminator.train_on_batch(X_batch, y_batch)

            # Prepare latent points for generator training
            latent_points = generate_latent_points(latent_dim, batch_size)
            gen_labels = np.ones((batch_size, 1))  # Use 'real' labels for generator training

            # Train generator via GAN
            g_loss = gan_model.train_on_batch(latent_points, gen_labels)

            print(f">Epoch {epoch+1}, Batch {batch+1}/{batches_per_epoch}, d_loss={d_loss:.3f}, g_loss={g_loss:.3f}")

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            summarize_performance(epoch, generator, discriminator, dataset, latent_dim)

# Set latent space dimension
LATENT_DIMENSION = 100

# Initialize discriminator model
discriminator_model = define_discriminator()

# Initialize generator model
generator_model = define_generator(LATENT_DIMENSION)

# Initialize combined GAN model
gan_model = define_gan(generator_model, discriminator_model)

# Load and preprocess real MNIST images
mnist_dataset = load_real_samples()

# Start training the GAN model
train_gan(generator_model, discriminator_model, gan_model, mnist_dataset, LATENT_DIMENSION)
