import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from modules import ApplyNoise, AdaIN
from dataset import generate_random_inputs

# Load the StyleGAN generator model
model = tf.keras.models.load_model(
    '/home/Student/s4748611/PatternAnalysis-2024/recognition/generator_model_AD_time_20241018_021440.h5',
    custom_objects={'ApplyNoise': ApplyNoise, 'AdaIN': AdaIN},
    compile=False
)

# Generate random latent vectors and constant inputs
num_images = 5  # Number of images to generate
latent_dim = 512  # Dimensionality of the latent space
initial_size = 8  # Initial size used in the generator

# Generate latent vectors and constant inputs
latent_vectors, constant_inputs = generate_random_inputs(num_images, latent_dim, initial_size)

# Generate images using both inputs
generated_images = model([latent_vectors, constant_inputs])

# Process images
generated_images = (generated_images + 1) / 2.0

generated_images_gray = tf.image.rgb_to_grayscale(generated_images)

# Display and save generated images
for i in range(num_images):
    plt.imshow(generated_images_gray[i].numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(f"generated_image_{i+1}.png", bbox_inches='tight')
    plt.close()
