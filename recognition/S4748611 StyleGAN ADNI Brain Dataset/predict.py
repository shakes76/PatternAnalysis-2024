import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from modules import ApplyNoise, AdaIN
from dataset import generate_random_inputs

# Check if the user has provided 'AD' or 'NC' as an argument
if len(sys.argv) < 2 or sys.argv[1] not in ['AD', 'NC']:
    print("Usage: python script_name.py [AD/NC]")
    sys.exit(1)

# Choose the model based on user input
model_type = sys.argv[1]

if model_type == 'AD':
    model_path = '/home/Student/s4748611/PatternAnalysis-2024/recognition/generator_model_AD.h5'
elif model_type == 'NC':
    model_path = '/home/Student/s4748611/PatternAnalysis-2024/recognition/generator_model_NC.h5'

# Load the corresponding StyleGAN generator model
model = tf.keras.models.load_model(
    model_path,
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

# Check the shape of the generated images
print("Shape of generated images:", generated_images.shape)

# Process images (rescale from [-1, 1] to [0, 1])
generated_images = (generated_images + 1) / 2.0

# Display and save generated grayscale images
for i in range(num_images):
    # Remove channel dimension for displaying grayscale images with matplotlib
    image_to_display = tf.squeeze(generated_images[i], axis=-1).numpy()
    
    plt.imshow(image_to_display, cmap='gray')
    plt.axis('off')
    plt.savefig(f"generated_image_bw_{model_type}_{i+1}.png", bbox_inches='tight')
    plt.close()

print(f"Generated {num_images} images using the {model_type} model.")
