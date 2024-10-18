import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image

# Set the paths to your image directories
ad_image_dir = '/home/Student/s4742656/PatternAnalysis-2024/recognition/StyleGAN_VBV/saved_examples/step5/seperate_images/AD'  
nc_image_dir = '/home/Student/s4742656/PatternAnalysis-2024/recognition/StyleGAN_VBV/saved_examples/step5/seperate_images/NC' 

# Load the pre-trained ResNet model
model = models.resnet50(weights='IMAGENET1K_V1')  # Use appropriate weights version
model.eval()  # Set the model to evaluation mode

# Define the preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Function to load and preprocess images
def load_and_preprocess_images(image_dir, label):
    features = []
    labels = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert('RGB')  # Open image
        img_tensor = preprocess(img)  # Preprocess image
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Extract features
        with torch.no_grad():  # Disable gradient computation
            feature = model(img_tensor)
        features.append(feature.numpy().flatten())
        labels.append(label)  # Append the label for this set

    return np.array(features), labels

# Load images for both sets and extract features
nc_features, nc_labels = load_and_preprocess_images(nc_image_dir, label=0)  # Label 0 for NC
ad_features, ad_labels = load_and_preprocess_images(ad_image_dir, label=1)  # Label 1 for AD

# Combine features and labels
features = np.vstack((nc_features, ad_features))
labels = np.array(nc_labels + ad_labels)

# Optional: Reduce dimensions with PCA
pca = PCA(n_components=min(10, len(features)))
reduced_features = pca.fit_transform(features)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Adjust perplexity
tsne_results = tsne.fit_transform(reduced_features)


# Visualize the results
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
