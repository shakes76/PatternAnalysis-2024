import torch
import torch.optim as optim
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

# Data paths
train_path = "./data/keras_slices_train"
validate_path = "./data/keras_slices_validate"

# Hyperparameters
batch_size = 256
num_epochs = 10

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

learning_rate = 1e-3

# Prepare Data Loaders
train_loader = get_data_loader(train_path, batch_size=batch_size, norm_image=True)
validate_loader = get_data_loader(validate_path, batch_size=batch_size, norm_image=True)

# Initialize Model and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# Initialize lists to track metrics
train_recon_errors = []
train_perplexities = []
train_ssim_scores = []

validation_recon_errors = []
validation_perplexities = []
validation_ssim_scores = []

# Function to calculate SSIM
def calculate_ssim(original, reconstructed):
    ssim_scores = []
    for i in range(original.size(0)):
        orig = original[i].cpu().detach().numpy().squeeze()
        recon = reconstructed[i].cpu().detach().numpy().squeeze()
        ssim_score = ssim(orig, recon, data_range=recon.max() - recon.min())
        ssim_scores.append(ssim_score)
    return np.mean(ssim_scores)

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Ensure model is in training mode
    epoch_train_recon_error = []
    epoch_perplexity = []
    epoch_train_ssim = []
    
    for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward Pass
        vq_loss, data_recon, perplexity = model(data)

        # Compute Loss and Backpropagation
        train_recon_error = F.mse_loss(data_recon, data)
        loss = train_recon_error + vq_loss
        loss.backward()

        # Optimize
        optimizer.step()
    
        # Collect Statistics
        epoch_train_recon_error.append(train_recon_error.item())
        epoch_perplexity.append(perplexity.item())
        
        # Calculate SSIM
        train_ssim = calculate_ssim(data, data_recon)
        epoch_train_ssim.append(train_ssim)
    
    # Store average stats for the epoch
    train_recon_errors.append(np.mean(epoch_train_recon_error))
    train_perplexities.append(np.mean(epoch_perplexity))
    train_ssim_scores.append(np.mean(epoch_train_ssim))
    
    print(f"Epoch {epoch + 1} - Train Recon Error: {train_recon_errors[-1]:.3f}, Perplexity: {train_perplexities[-1]:.3f}, SSIM: {train_ssim_scores[-1]:.3f}")

    # Validation step after each epoch
    model.eval()
    val_recon_errors = []
    val_perplexities = []
    val_ssim_scores = []

    with torch.no_grad():
        for val_data in validate_loader:
            val_data = val_data.to(device)
            vq_loss, val_data_recon, perplexity = model(val_data)
            val_recon_error = F.mse_loss(val_data_recon, val_data)
            val_recon_errors.append(val_recon_error.item())
            val_perplexities.append(perplexity.item())
            
            # Calculate SSIM for validation
            val_ssim = calculate_ssim(val_data, val_data_recon)
            val_ssim_scores.append(val_ssim)

    avg_val_recon_error = np.mean(val_recon_errors)
    avg_val_perplexity = np.mean(val_perplexities)
    avg_val_ssim = np.mean(val_ssim_scores)
    
    validation_recon_errors.append(avg_val_recon_error)
    validation_perplexities.append(avg_val_perplexity)
    validation_ssim_scores.append(avg_val_ssim)

    print(f"Epoch {epoch + 1} - Validation Recon Error: {avg_val_recon_error:.3f}, Perplexity: {avg_val_perplexity:.3f}, SSIM: {avg_val_ssim:.3f}")

# Save model
torch.save(model.state_dict(), "vqvae_model.pth")
print("Model saved as vqvae_model.pth")

# Plot Metrics
def plot_metrics(train, validation, metric_name):
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train, 'bo-', label=f'Train {metric_name}')
    plt.plot(epochs, validation, 'ro-', label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Epochs')
    plt.legend()
    plt.show()

# Plot reconstruction error, perplexity, and SSIM
plot_metrics(train_recon_errors, validation_recon_errors, "Reconstruction Error")
plot_metrics(train_perplexities, validation_perplexities, "Perplexity")
plot_metrics(train_ssim_scores, validation_ssim_scores, "SSIM")

# Final evaluation on the validation set for visualization
model.eval()

with torch.no_grad():
    # Get one batch from the validation set
    valid_originals = next(iter(validate_loader)).to(device)
    
    # Pass it through the model to get reconstructions
    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize).cpu().data
    valid_originals = valid_originals.cpu().data

# Display Results
def show(img):
    npimg = img.numpy()
    npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

print("Displaying original images:")
show(make_grid(valid_originals[:16], nrow=8, normalize=True))
plt.show()

print("Displaying reconstructed images:")
show(make_grid(valid_reconstructions[:16], nrow=8, normalize=True))
plt.show()
