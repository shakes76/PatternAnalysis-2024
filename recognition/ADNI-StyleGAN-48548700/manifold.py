import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Import the Discriminator from modules.py
from modules import Discriminator as ODiscriminator

# Function to register a hook on a layer to extract features
def get_features_from_layer(layer):
    def hook(model, input, output):
        layer_features.append(output)
    return hook

class ModifiedDiscriminator(ODiscriminator):
    def __init__(self, log_resolution, n_features=64, max_features=256):
        super().__init__(log_resolution, n_features, max_features)
    
    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.minibatch_std(x)
        if x.shape[-1] < 3:
            x = x.reshape(x.shape[0], -1)
        else:
            x = self.conv(x)
            x = x.reshape(x.shape[0], -1)
        return self.final(x)

# Load the checkpointed discriminator model
def load_discriminator(checkpoint_path, log_resolution):
    discriminator = ModifiedDiscriminator(log_resolution=log_resolution)
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

# Load and preprocess all images from the dataset directory
def load_images_from_epochs(root_dir):
    image_tensors = []
    for epoch_dir in os.listdir(root_dir):
        epoch_path = os.path.join(root_dir, epoch_dir)
        if os.path.isdir(epoch_path):
            # Traverse w1, w2, etc.
            for w_dir in os.listdir(epoch_path):
                w_path = os.path.join(epoch_path, w_dir)
                if os.path.isdir(w_path):
                    for image_name in os.listdir(w_path):
                        image_path = os.path.join(w_path, image_name)
                        if image_name.endswith(".png"):
                            # Load and preprocess the image
                            image = Image.open(image_path).convert("RGB")
                            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
                            image_tensors.append(image_tensor)
    return torch.cat(image_tensors)

# Flatten the feature maps
def flatten_features(layer_features):
    flattened_features = []
    for feature in layer_features:
        batch_size = feature.shape[0]
        flattened = feature.view(batch_size, -1)  # Flatten spatial dimensions
        flattened_features.append(flattened)
    return torch.cat(flattened_features, dim=1).cpu().numpy()

# Plot t-SNE embeddings with labeled clusters
def plot_tsne_with_named_clusters(tsne_embeddings, cluster_labels):
    plt.figure(figsize=(10, 8))
    
    # Map cluster labels to "AD" and "CN"
    cluster_names = np.where(cluster_labels == 0, 'AD', 'CN')
    
    # Plot each cluster with its label
    for cluster, label in [('AD', 'red'), ('CN', 'blue')]:
        mask = (cluster_names == cluster)
        plt.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], label=cluster, color=label, s=50)
    
    plt.legend()
    plt.title('t-SNE Projection with AD and CN Clusters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

if __name__ == "__main__":
    # Path to the checkpoint (discriminator state_dict)
    checkpoint_path = 'critic_epoch45.pt'

    # Root directory containing the dataset (epoch0, epoch5, etc.)
    root_dir = 'path_to_your_dataset'  # Example: 'generated_images/'

    # Load the saved discriminator model from modules.py
    log_resolution = 7  # Adjust this according to your discriminator setup
    discriminator = load_discriminator(checkpoint_path, log_resolution)

    # Register hooks to capture features from specific layers
    layer_features = []
    discriminator.from_rgb[1].register_forward_hook(get_features_from_layer(discriminator.from_rgb[1]))  # Example: from_rgb layer
    discriminator.blocks[0].block[1].register_forward_hook(get_features_from_layer(discriminator.blocks[0].block[1]))  # Example: first block's conv layer

    # Load and preprocess all images from the directory
    images = load_images_from_epochs(root_dir)

    # Forward pass through the discriminator to get the features
    with torch.no_grad():  # No need to compute gradients
        layer_features.clear()  # Clear any previous feature storage
        discriminator(images)  # Forward pass

    # Flatten the extracted feature maps
    all_features = flatten_features(layer_features)

    # Apply t-SNE to reduce dimensionality to 2D
    tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embeddings = tsne_model.fit_transform(all_features)

    # Perform K-Means clustering on the t-SNE embeddings
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(tsne_embeddings)

    # Plot the t-SNE embeddings with "AD" and "CN" labels
    plot_tsne_with_named_clusters(tsne_embeddings, cluster_labels)
