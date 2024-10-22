import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from modules import Generator, MappingNetwork

# Define constants
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
CHANNELS_IMG = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load both generators
generator_ad = Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
generator_ad.load_state_dict(torch.load('AD_generator_final.pth', map_location=DEVICE))

generator_nc = Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
generator_nc.load_state_dict(torch.load('NC_generator_final.pth', map_location=DEVICE))

# Set to evaluation mode
generator_ad.eval()
generator_nc.eval()

# Mapping network to transform Z to W
mapping_network = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)

def generate_w_vectors(generator, num_samples=50):
    """ Function to generate W vectors """  
    # Reduce number of samples
    latent_vectors = torch.randn(num_samples, Z_DIM).to(DEVICE)
    with torch.no_grad():
        w_vectors = mapping_network(latent_vectors)
    return w_vectors.cpu().numpy()  # Only return W vectors to save memory

def plot_tsne(latent_vectors, labels=None):
    """ Function to plot t-SNE """ 
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                          c=labels, alpha=0.7, cmap='viridis')
    
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label='AD' if label == 0 else 'NC', 
                           markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10) for label in unique_labels]
    
    plt.legend(handles=handles, title="Classes")
    plt.colorbar(scatter)
    
    plt.title('t-SNE Embedding of StyleGAN Latent Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig('tsne_visualization.png')

# Generate W vectors for AD and NC
num_samples = 50 
w_vectors_ad = generate_w_vectors(generator_ad, num_samples)
w_vectors_nc = generate_w_vectors(generator_nc, num_samples)

# Combine W vectors and create labels
w_vectors = np.concatenate((w_vectors_ad, w_vectors_nc), axis=0)
labels = np.array([0] * num_samples + [1] * num_samples)  # 0 for AD, 1 for NC

# Plot the t-SNE results
plot_tsne(w_vectors, labels)
