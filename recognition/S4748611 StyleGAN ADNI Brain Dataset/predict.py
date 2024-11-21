import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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

# Load the corresponding StyleGAN generator models
if model_type == 'AD':
    model_path = 'generator_model_AD.h5'
elif model_type == 'NC':
    model_path = 'generator_model_NC.h5'

# Load the selected generator model
model = tf.keras.models.load_model(
    model_path,
    custom_objects={'ApplyNoise': ApplyNoise, 'AdaIN': AdaIN},
    compile=False
)

# Generate random latent vectors and constant inputs
num_images = 5  
latent_dim = 512  
initial_size = 8 

# Generate latent vectors and constant inputs
latent_vectors, constant_inputs = generate_random_inputs(num_images, latent_dim, initial_size)

# Generate images using the selected model
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
    plt.savefig(f"generated_image_{model_type}_{i+1}.png", bbox_inches='tight')
    plt.close()

print(f"Generated {num_images} images using the {model_type} model.")

# -------------------------------------------------
# t-SNE Visualization for both AD and NC style codes
# -------------------------------------------------

# Load both models for t-SNE comparison
with tf.keras.utils.custom_object_scope({'ApplyNoise': ApplyNoise, 'AdaIN': AdaIN}):
    generator_model_NC = tf.keras.models.load_model('generator_model_NC.h5', compile=False)
    generator_model_AD = tf.keras.models.load_model('generator_model_AD.h5', compile=False)

# Function to generate style codes from the models
def generate_style_codes(generator_model, num_samples):
    latent_dim = 512  
    initial_size = 8 
    
    # Generate latent vectors and constant inputs
    latent_vectors, constant_inputs = generate_random_inputs(num_samples, latent_dim, initial_size)
    
    # Generate style codes (output from the generator)
    style_codes = generator_model([latent_vectors, constant_inputs]) 
    return style_codes.numpy() 


# Generate style codes for both models
num_samples_NC = 400
num_samples_AD = 400  

style_code_NC = generate_style_codes(generator_model_NC, num_samples_NC)
style_code_AD = generate_style_codes(generator_model_AD, num_samples_AD)

# Concatenate the style codes
combined_style_codes = np.concatenate((style_code_NC, style_code_AD), axis=0)

# Reshape the 4D array to 2D (flatten each style code)
flattened_style_codes = combined_style_codes.reshape(combined_style_codes.shape[0], -1)

# Now apply the StandardScaler to the flattened data
scaler = StandardScaler()
scaled_style_codes = scaler.fit_transform(flattened_style_codes)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(scaled_style_codes)

# Separate the t-SNE results for each model
tsne_NC = tsne_results[:num_samples_NC]
tsne_AD = tsne_results[num_samples_NC:num_samples_NC + num_samples_AD]

# Visualize the t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(tsne_NC[:, 0], tsne_NC[:, 1], color='blue', label='Normal Cognition (NC)', alpha=0.5)
plt.scatter(tsne_AD[:, 0], tsne_AD[:, 1], color='red', label='Alzheimer\'s Disease (AD)', alpha=0.5)
plt.title('t-SNE Visualization of Style Codes')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.grid()
plt.savefig("Tsne.png")

print("t-SNE plot saved as 'Tsne.png'.")
