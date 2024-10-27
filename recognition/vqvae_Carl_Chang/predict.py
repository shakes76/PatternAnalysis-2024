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

# Initialize Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Model
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost)
model.load_state_dict(torch.load("vqvae_model.pth", map_location=device))  # Load the saved model weights
model.to(device)
model.eval()  # Set to evaluation mode

# Prepare Data Loader
test_loader = get_data_loader(test_path, batch_size=batch_size, norm_image=True)

# Perform Inference
test_originals = next(iter(test_loader)).to(device)
with torch.no_grad():
    vq_output_eval = model._pre_vq_conv(model._encoder(test_originals))
    _, test_quantize, _, _ = model._vq_vae(vq_output_eval)
    test_reconstructions = model._decoder(test_quantize)

test_reconstructions = test_reconstructions.cpu().data
test_originals = test_originals.cpu().data

# Calculate SSIM between original and reconstructed images
def calculate_ssim(original, reconstructed):
    ssim_scores = []
    for i in range(original.size(0)):
        orig = original[i].numpy().squeeze()
        recon = reconstructed[i].numpy().squeeze()
        ssim_score = ssim(orig, recon, data_range=recon.max() - recon.min())
        ssim_scores.append(ssim_score)
    return np.mean(ssim_scores)

average_ssim = calculate_ssim(test_originals, test_reconstructions)
print(f"Average SSIM: {average_ssim:.3f}")

# Display Results
def show(img):
    npimg = img.numpy()
    npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

print("Displaying reconstructed images:")
show(make_grid(test_reconstructions[:16], nrow=4, normalize=True))
plt.show()

print("Displaying original images:")
show(make_grid(test_originals[:16], nrow=4, normalize=True))
plt.show()

