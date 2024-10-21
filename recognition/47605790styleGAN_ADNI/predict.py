import torch
from train import test  # Import the test function from train.py
from modules import Generator, MappingNetwork
from dataset import load_data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import umap
import os

# Function to visualize and save UMAP embeddings with ground truth
def plot_umap(embeddings, labels, title="UMAP Embeddings", save_path="/home/Student/s4760579/test_images/umap_embedding.png"):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Spectral', s=10)
    plt.colorbar(scatter)
    plt.title(title)
    
    # Ensure the save directory exists
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the plot as an image file
    plt.savefig(save_path)

# Function to save generated images to disk
def save_images(images, folder_path="/home/Student/s4760579/test_images"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, img in enumerate(images):
        img = img.transpose(1, 2, 0)  # Rearrange dimensions to [H, W, C]
        img = (img * 127.5 + 127.5).astype("uint8")  # Denormalize to [0, 255]
        img_path = os.path.join(folder_path, f"generated_img_{idx}.png")
        Image.fromarray(img).save(img_path)

# Function to perform inference and UMAP embedding
def predict_and_visualize(generator, mapping_network, test_loader, device):
    # Call the test function to generate images from the generator
    generated_images = test(generator, mapping_network, test_loader, device, num_images=10)
    
    # Save generated images to disk
    save_images(generated_images)
    
    all_images = []
    labels = []

    # Gather real images from the test dataset along with their labels
    for real_imgs, real_labels in test_loader:
        all_images.append(real_imgs.cpu().numpy())
        labels.extend(real_labels.numpy())
        
        if len(labels) >= len(generated_images):
            break

    # Convert generated images to NumPy and concatenate with real images
    generated_images_np = (generated_images * 127.5 + 127.5).astype("uint8")  # Denormalize
    
    # Stack all real and generated images for embedding
    stacked_images = np.vstack([generated_images_np, np.vstack(all_images)])
    
    # Perform UMAP embedding on image data
    flat_images = stacked_images.reshape(stacked_images.shape[0], -1)  # Flatten images for UMAP
    umap_model = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean')
    embeddings = umap_model.fit_transform(flat_images)
    
    # Create labels: 0 for generated, real labels for actual images
    generated_labels = [-1] * generated_images_np.shape[0]  # Assign -1 for generated images
    all_labels = generated_labels + labels[:len(generated_images)]  # Combine generated and real labels

    # Plot UMAP embeddings
    plot_umap(embeddings, all_labels, title="UMAP Embedding of Generated and Real Images")

# Main function for predict.py
if __name__ == "__main__":
    # Load the models and dataset
    test_dir = r'/home/groups/comp3710/ADNI/AD_NC/test'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    generator = Generator().to(device)
    mapping_network = MappingNetwork().to(device)
    
    # Load pre-trained models
    generator.load_state_dict(torch.load('/home/Student/s4760579/models/generator_120.pth', map_location=device, weights_only=True))
    mapping_network.load_state_dict(torch.load('/home/Student/s4760579/models/mapping_network_120.pth', map_location=device,  weights_only=True))

    # Load test data
    _, test_loader = load_data(test_dir, test_dir, batch_size=32)
    
    # Perform prediction and UMAP embedding visualization
    predict_and_visualize(generator, mapping_network, test_loader, device)
