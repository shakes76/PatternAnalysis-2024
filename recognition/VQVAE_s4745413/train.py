import wandb
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import os
from dataset import NiftiDataset
from modules import Encoder, Decoder, VectorQuantizer, VQVAE
import numpy as np
from tqdm import tqdm
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




num_epochs = 100
batch_size = 16
learning_rate = 1e-4
num_emb = 512
e_dim = 64
commit_cost = 0.25
n_res_layers = 2
res_h_dim = 32
h_dim = 64


input_transf = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.Normalize((0.5,), (0.5,))
    ])

hipmri_train_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_train"
hipmri_val_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_validate"
hipmri_test_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_test"

base_dir = os.getcwd()
hipmri_train_dir = os.path.join(base_dir, hipmri_train_dir)
hipmri_val_dir = os.path.join(base_dir, hipmri_val_dir)
hipmri_test_dir = os.path.join(base_dir, hipmri_test_dir)
#hipmri_train_dir = "keras_slices_data/keras_slices_train"
#hipmri_valid_dir= "keras_slices_data/keras_slices_validate"
#hipmri_test_dir= "keras_slices_data/keras_slices_test"
save_dir = os.path.join(base_dir, "PatternAnalysis-2024/recognition/VQVAE_s4745413/reconstructed_images")


training_set, validation_set, test_set = NiftiDataset.get_dataloaders(train_dir=hipmri_train_dir, val_dir=hipmri_val_dir, test_dir=hipmri_test_dir, batch_size=16, num_workers=4, transform=input_transf)

model = VQVAE(h_dim, res_h_dim, n_res_layers, num_emb, e_dim, commit_cost).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def loss_fn(recon_loss, img, embed_loss):
    bce = F.binary_cross_entropy(recon_loss, img)
    total_loss = bce + embed_loss
    return total_loss

# Using Variance for normalisation
mean = 0.0
mean_sq = 0.0
count = 0

for idx, data in enumerate(training_set):
    mean += data.sum()
    mean_sq += (data ** 2).sum()
    count += np.prod(data.shape)

total_mean = mean / count
total_var = (mean_sq / count) - (total_mean ** 2)
data_var = float(total_var.item())

wandb.init(project="hip_mri_vqvae", 
        name="training2",
        config = {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_emb": num_emb,
            "e_dim": e_dim,
            "commit_cost": commit_cost,
            "h_dim": h_dim,
            "res_h_dim": res_h_dim,
            "n_res_layers": n_res_layers,
            })

# Training
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in training_set:
        optimizer.zero_grad()
        batch = batch.to(device)
        scalar_loss, x_hat, scalar_metric = model(batch)
        # print(f"OUTPUT OF MODEL(BATCH): {x_hat}")

        # ssim_loss = 1 - ssim(batch, x_hat, data_range=1, size_average=True)
        recon_loss = torch.nn.functional.mse_loss(batch, x_hat) / data_var # normalise loss
        loss = recon_loss + scalar_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(training_set)

    # Validation
    model.eval()
    val_loss = 0
    val_ssim = []
    with torch.no_grad():
        for batch in validation_set:
            batch = batch.to(device)
            scalar_loss, x_hat, scalar_metric = model(batch)
            recon_loss = torch.nn.functional.mse_loss(batch, x_hat) / data_var
            loss = recon_loss + scalar_loss
            val_loss += loss.item()

            ssim_loss = ssim(batch, x_hat, data_range=1.0)
            val_ssim.append(ssim_loss.item())
    avg_validation_loss = val_loss / len(validation_set)
    avg_ssim = np.mean(val_ssim)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_validation_loss: .4f},  SSIM: {avg_ssim: .4f}")

    wandb.log({
        "train_loss": avg_train_loss,
        "validation_loss": avg_validation_loss,
        "ssim_loss": avg_ssim,
        "epoch": epoch + 1
    })
    if epoch + 1 == num_epochs:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, 'vqvae_model.pth')
wandb.finish()
# Function to save images
# def save_images(tensor_images, save_dir, prefix='recon', num_samples=10):
    # Make sure to clamp the images to the range [0, 1]
#    tensor_images = torch.clamp(tensor_images, 0, 1)
    
    # Unnormalize and save each image
#    for i in range(num_samples):
#        img_tensor = tensor_images[i].cpu()
        
        # Convert tensor to a PIL image
#        img_pil = torchvision.transforms.ToPILImage()(img_tensor)
        
        # Save the image to the directory
#        img_pil.save(os.path.join(save_dir, f"{prefix}_image_{i}.png"))

# Get a batch of images from your test dataloader
#with torch.no_grad():
#    for batch_idx, batch in enumerate(test_set):
#        batch = batch.to(device)
#        
#        # Pass the batch through the model to get reconstructions
#        _, x_hat, _ = model(batch)
#        
#        # Save the original and reconstructed images
#        save_images(batch, save_dir, prefix=f'original_{batch_idx}', num_samples=10)
#        save_images(x_hat, save_dir, prefix=f'reconstructed_{batch_idx}', num_samples=10)
        
        # Only process one batch for saving (you can remove this break to save more batches)
#        break

# print(f"Images saved to {save_dir}")
