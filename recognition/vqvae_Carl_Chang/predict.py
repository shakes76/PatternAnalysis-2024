import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim 

# Import your custom modules
from modules import VQVAE
from dataset import get_data_loader 
from utils import calculate_ssim, show_img, show_combined

# Data path
test_path = "./data/keras_slices_test"

# Hyperparameters (same as used during training)
batch_size = 256
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25

# Prepare Data Loader
test_loader = get_data_loader(test_path, batch_size=batch_size, norm_image=True)

# Initialize Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Model
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost)
model.load_state_dict(torch.load("vqvae_model.pth", map_location=device))  # Load the saved model weights
model.to(device)
model.eval()  # Set to evaluation mode

# Get batch of test images and move to device
test_originals = next(iter(test_loader)).to(device)
with torch.no_grad(): # No need to calculate gradients
    # Pass test images through encoder and pre-VQ convolution layer
    vq_output_eval = model._pre_vq_conv(model._encoder(test_originals))
    # Quantize latent representation
    _, test_quantize, _, _ = model._vq_vae(vq_output_eval)
    # Decode to get reconstructed images
    test_reconstructions = model._decoder(test_quantize)

# Move images to back to cpu
test_reconstructions = test_reconstructions.cpu().data
test_originals = test_originals.cpu().data

# Calculate average SSIM
average_ssim = calculate_ssim(test_originals, test_reconstructions)
print(f"Average SSIM: {average_ssim:.3f}")

# Display images
print("Displaying original images:")
show_img(make_grid(test_originals[:16], nrow=8, normalize=True))
plt.show()

print("Displaying reconstructed images:")
show_img(make_grid(test_reconstructions[:16], nrow=8, normalize=True))
plt.show()

print("Displaying original vs reconstructed images:")
show_combined(test_originals, test_reconstructions, average_ssim)
plt.show()
