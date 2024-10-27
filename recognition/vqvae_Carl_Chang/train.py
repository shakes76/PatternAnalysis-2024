import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

# Import your custom modules
from modules import VQVAE
from dataset import get_data_loader 

# Data paths
train_path = "./data/keras_slices_train"
validate_path = "./data/keras_slices_validate"

# Hyperparameters
batch_size = 256
num_training_updates = 1500

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

learning_rate = 1e-3

# Prepare Data Loaders
train_loader = get_data_loader(train_path, batch_size=batch_size, norm_image=True, early_stop=True)
validate_loader = get_data_loader(validate_path, batch_size=batch_size, norm_image=True, early_stop=True)

# Initialize Model and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# Training Loop
model.train()
train_recon_errors = []
train_perplexities = []

for update in tqdm(range(num_training_updates), desc="Training Progress"):
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Forward Pass
        vq_loss, data_recon, perplexity = model(data)

        # Compute Loss and Backpropagation
        recon_error = F.mse_loss(data_recon, data)
        loss = recon_error + vq_loss
        loss.backward()

        # Optimize
        optimizer.step()
    
        # Collect Statistics
        train_recon_errors.append(recon_error.item())
        train_perplexities.append(perplexity.item())


# Save model
torch.save(model.state_dict(), "vqvae_model.pth")
print("Model saved as vqvae_model.pth")


# Evaluate and Display Images
model.eval()

def show(img):
    npimg = img.numpy()
    npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

# Process Validation Data for Visualization
valid_originals = next(iter(validate_loader)).to(device)
with torch.no_grad():
    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

valid_reconstructions = valid_reconstructions.cpu().data
valid_originals = valid_originals.cpu().data

# Display Results
print("Displaying reconstructed images:")
show(make_grid(valid_reconstructions, nrow=8, normalize=True))
plt.show()

print("Displaying original images:")
show(make_grid(valid_originals, nrow=8, normalize=True))
plt.show()
