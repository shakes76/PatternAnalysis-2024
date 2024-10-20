import torch
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image

# Set the paths to your image directories
ad_image_dir = '/home/Student/s4742656/PatternAnalysis-2024/recognition/StyleGAN_VBV/saved_examples/step5/seperate_images/AD'  
nc_image_dir = '/home/Student/s4742656/PatternAnalysis-2024/recognition/StyleGAN_VBV/saved_examples/step5/seperate_images/NC' 

# Define the preprocessing transformations for the input images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize based on ImageNet stats
])

def load_and_preprocess_images(image_dir, label):
    """ Load and preprocess images from a specified directory. """
    features = []  # List to hold extracted features
    labels = []    # List to hold corresponding labels

    # Iterate over each image in the directory
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)  # Construct full image path
        img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        img_tensor = preprocess(img)  # Preprocess image
        img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension for model input

        # Simple feature extraction: use pixel values as features
        feature = img_tensor.numpy().flatten()  # Flatten the tensor and append to the list
        features.append(feature)  # Append feature to the list
        labels.append(label)  # Append the label for this image

    return np.array(features), labels  # Return the features and labels as arrays

# Load images for both NC and AD sets and extract features
nc_features, nc_labels = load_and_preprocess_images(nc_image_dir, label=0)  # Label 0 for NC
ad_features, ad_labels = load_and_preprocess_images(ad_image_dir, label=1)  # Label 1 for AD

# Combine features and labels from both sets
features = np.vstack((nc_features, ad_features))  # Stack features vertically
labels = np.array(nc_labels + ad_labels)  # Concatenate labels

# Optional: Reduce dimensions using PCA for visualization
pca = PCA(n_components=min(10, len(features)))  # Choose the number of components
reduced_features = pca.fit_transform(features)  # Fit and transform the features

# Apply t-SNE for further dimensionality reduction and visualization
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Define t-SNE parameters
tsne_results = tsne.fit_transform(reduced_features)  # Fit and transform the PCA-reduced features

# Visualize the t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1], label='Class')
plt.title('t-SNE Visualization of ADNI Brain Images')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.savefig('tsne_visualization.png')
