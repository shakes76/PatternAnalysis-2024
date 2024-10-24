import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE

# Import the Discriminator from modules.py
from modules import Discriminator

# Function to register a hook on a layer to extract features
def get_features_from_layer(layer):
    def hook(model, input, output):
        layer_features.append(output)
    return hook

# Load the checkpointed discriminator model
def load_discriminator(checkpoint_path, log_resolution):
    # Initialize the discriminator
    discriminator = Discriminator(log_resolution=log_resolution)
    
    # Load the state_dict from the checkpoint
    checkpoint = torch.load(checkpoint_path)
    discriminator.load_state_dict(checkpoint)
    
    discriminator.eval()  # Set to evaluation mode
    return discriminator

# Image transformation (same as during training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjust size according to your model input
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load and preprocess a single image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Flatten the feature maps
def flatten_features(layer_features):
    flattened_features = []
    for feature in layer_features:
        batch_size = feature.shape[0]
        flattened = feature.view(batch_size, -1)  # Flatten spatial dimensions
        flattened_features.append(flattened)
    return torch.cat(flattened_features, dim=1).cpu().numpy()

# Plot t-SNE embeddings
def plot_tsne(tsne_embeddings):
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], s=10, cmap='Spectral')
    plt.title('t-SNE Projection of Discriminator Features')
    plt.show()

if __name__ == "__main__":
    # Path to the checkpoint (discriminator state_dict)
    checkpoint_path = 'critic_epoch45.pt'

    # Load the saved discriminator model from modules.py
    log_resolution = 7  # Adjust this according to your discriminator setup
    discriminator = load_discriminator(checkpoint_path, log_resolution)

    # Register hooks to capture features from specific layers
    layer_features = []
    discriminator.from_rgb[1].register_forward_hook(get_features_from_layer(discriminator.from_rgb[1]))  # Example: from_rgb layer
    discriminator.blocks[0].block[1].register_forward_hook(get_features_from_layer(discriminator.blocks[0].block[1]))  # Example: first block's conv layer

    # Load images (example: load multiple images)
    image_paths = ['generated_images/epoch40/img_0.png', 'generated_images/epoch40/img_1.png', 'generated_images/epoch40/img_2.png']  # Replace with actual image paths
    images = torch.cat([load_and_preprocess_image(image_path) for image_path in image_paths])

    # Forward pass through the discriminator to get the features
    with torch.no_grad():  # No need to compute gradients
        layer_features.clear()  # Clear any previous feature storage
        discriminator(images)  # Forward pass

    # Flatten the extracted feature maps
    all_features = flatten_features(layer_features)

    # Apply t-SNE to reduce dimensionality to 2D
    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_embeddings = tsne_model.fit_transform(all_features)

    # Plot the t-SNE embeddings
    plot_tsne(tsne_embeddings)
