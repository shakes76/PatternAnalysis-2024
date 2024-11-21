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
from utils import calculate_ssim, show_img, show_combined, plot_metrics

# Data paths
train_path = "./data/keras_slices_train"
validate_path = "./data/keras_slices_validate"

# Hyperparameters
batch_size = 256
num_epochs = 20
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

# Initialise Model and optimiser
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost).to(device)
optimiser = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# Initialise lists to track metrics
train_recon_errors = []
train_perplexities = []
train_ssim_scores = []

val_recon_errors = []
val_perplexities = []
validation_ssim_scores = []

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Ensure model is in training mode
    epoch_train_recon_error = []
    epoch_perplexity = []
    epoch_train_ssim = []
    
    # Training for each batch
    for data in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        data = data.to(device)
        # Reset gradients
        optimiser.zero_grad()
        # Forward Pass
        vq_loss, data_recon, perplexity = model(data)
        # Compute losses
        train_recon_error = F.mse_loss(data_recon, data)
        loss = train_recon_error + vq_loss
        # Backpropagation
        loss.backward()
        # Update model weights
        optimiser.step()
        # Collect metrics
        epoch_train_recon_error.append(train_recon_error.item())
        epoch_perplexity.append(perplexity.item())
        # Calculate SSIM
        train_ssim = calculate_ssim(data, data_recon)
        epoch_train_ssim.append(train_ssim)
    
    # Store average stats for epoch
    train_recon_errors.append(np.mean(epoch_train_recon_error))
    train_perplexities.append(np.mean(epoch_perplexity))
    train_ssim_scores.append(np.mean(epoch_train_ssim))
    print(f"Epoch {epoch + 1} - Training Reconstruction Error (MSE Loss): {train_recon_errors[-1]:.3f}, Perplexity: {train_perplexities[-1]:.3f}, SSIM: {train_ssim_scores[-1]:.3f}")

    # Validation step after each epoch
    model.eval()
    # Lists to track validation metrics for this epoch
    epoch_val_recon_error = []
    epoch_val_perplexity = []
    epoch_val_ssim_scores = []

    with torch.no_grad():
        # Evaluate on validation set
        for val_data in validate_loader:
            val_data = val_data.to(device)
            # Pass through model to get reconstructed images
            vq_loss, val_data_recon, perplexity = model(val_data)
            # Calculate and track MSE loss, perplexity, and SSIM
            val_recon_error = F.mse_loss(val_data_recon, val_data)
            epoch_val_recon_error.append(val_recon_error.item())
            epoch_val_perplexity.append(perplexity.item())
            val_ssim = calculate_ssim(val_data, val_data_recon)
            epoch_val_ssim_scores.append(val_ssim)
    
    # Compute average validation metrics
    avg_val_recon_error = np.mean(epoch_val_recon_error)
    avg_val_perplexity = np.mean(epoch_val_perplexity)
    avg_val_ssim = np.mean(epoch_val_ssim_scores)
    
    # Add to list of metrics
    val_recon_errors.append(avg_val_recon_error)
    val_perplexities.append(avg_val_perplexity)
    validation_ssim_scores.append(avg_val_ssim)

    print(f"Epoch {epoch + 1} - Validation Reconstruction Error (MSE Loss): {avg_val_recon_error:.3f}, Perplexity: {avg_val_perplexity:.3f}, SSIM: {avg_val_ssim:.3f}")

# Save model
torch.save(model.state_dict(), "vqvae_model.pth")
print("Model saved as vqvae_model.pth")

# Plot reconstruction error, perplexity, and SSIM
plot_metrics(train_recon_errors, val_recon_errors, "Reconstruction Error (MSE Loss)")
plot_metrics(train_perplexities, val_perplexities, "Perplexity")
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

print("Displaying original images:")
show_img(make_grid(valid_originals[:16], nrow=8, normalize=True))
plt.show()

print("Displaying reconstructed images:")
show_img(make_grid(valid_reconstructions[:16], nrow=8, normalize=True))
plt.show()

print("Displaying original vs reconstructed images:")
show_combined(valid_originals, valid_reconstructions, calculate_ssim(valid_originals, valid_reconstructions))
plt.show()
