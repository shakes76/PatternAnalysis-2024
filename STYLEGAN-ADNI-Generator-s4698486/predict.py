import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.manifold import TSNE
from utils import get_style_vector, get_noise
from constants import save, image_height, image_width, w_dim, log_resolution, z_dim
from modules import Generator, MappingNetwork


"""
 Plot loss graph for both generator and discriminator
"""
def plot_loss(G_Loss, D_Loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator Loss During Training")
    plt.plot(G_Loss, label="G", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gen_loss.png')


    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Loss During Training")
    plt.plot(D_Loss, label="D", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('disc_loss.png')


"""
 Generate 10 example images from the generator.
"""
def generate_examples(gen, mapping_network, epoch, device):
    n = 10
    for i in range(n):
        with torch.no_grad():
            w = get_style_vector(1, mapping_network, device)
            noise = get_noise(1, device)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples'):
                os.makedirs(f'saved_examples')
            save_image(img * 0.5 + 0.5, f"saved_examples/{image_height}x{image_width}/epoch{epoch}_img_{i}.png")




# Initialize model architectures (ensure the parameters match those used during saving)
generator = Generator(log_resolution, w_dim)
mapping_network = MappingNetwork(z_dim=z_dim, w_dim=w_dim)  # Example, use your actual z_dim


# Load the state_dicts
generator.load_state_dict(torch.load(f'Models/Gen/{image_height}x{image_width}/50'))
mapping_network.load_state_dict(torch.load(f'Models/Mapping/{image_height}x{image_width}/50'))


# Move the models to the appropriate device (e.g., CUDA if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
mapping_network.to(device)


# Set models to evaluation mode if you plan to use them for inference
generator.eval()
mapping_network.eval()


# Generate examples (Note that the epoch value doesn't matter here; it's just for file naming)
generate_examples(gen=generator, mapping_network=mapping_network, epoch=0, device=device)


"""
 Just convert latent vectors to style vectors.
"""
def extract_style_vectors(mapping_network, z_samples):
    with torch.no_grad():
        w_vectors = mapping_network(z_samples)
    return w_vectors.cpu().numpy()


# Example: Generate 1000 random z vectors and map them to w space
n_samples = 1000
z_samples = torch.randn(n_samples, z_dim).to(device)  # Adjust z_dim as needed
w_vectors = extract_style_vectors(mapping_network, z_samples)


# Step 2: Apply t-SNE to the extracted style vectors
tsne = TSNE(n_components=2, perplexity=30, random_state=46984863) # I have decreased the perplexity to 2 as this doesn't preserve global structures
w_2d = tsne.fit_transform(w_vectors)


# Step 3: Plot the t-SNE Results
plt.figure(figsize=(8, 6))
plt.scatter(w_2d[:, 0], w_2d[:, 1], alpha=0.7)
plt.title("t-SNE of Style Space (W) Vectors")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig('tsne_style_space.png')
plt.show()