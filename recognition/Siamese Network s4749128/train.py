
from dataset import load_data_2D, get_image_paths
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import csv
from modules import Encoder, Decoder, VQVAE, evaluate_ssim, display_images_and_ssim

# Get train image paths
train_paths = get_image_paths('HipMRI_study_keras_slices_data/keras_slices_train')

# Load the 2D medical images
train_images = load_data_2D(train_paths, normImage=True, categorical=False)

# Device configuration (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert the images to tensor and send to device
train_images_tensor = torch.Tensor(train_images).unsqueeze(1).to(device)  # Add channel dimension

# Hyperparameters
input_dim = 1  # Grayscale images
hidden_dim = 64
res_h_dim = 32
embedding_dim = hidden_dim  # Ensure embedding_dim matches hidden_dim for compatibility
num_embeddings = 512
n_res_layers = 2
batch_size = 32
num_epochs = 1
learning_rate = 1e-3
print(num_epochs)
# Instantiate encoder, decoder, and VQVAE model
encoder = Encoder(input_dim, hidden_dim, n_res_layers, res_h_dim).to(device)
decoder = Decoder(embedding_dim, hidden_dim, res_h_dim, n_res_layers).to(device)
vqvae = VQVAE(encoder, decoder, embedding_dim, num_embeddings).to(device)

# DataLoader setup
train_dataset = TensorDataset(train_images_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optimizer
optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)

# Training loop with progress bar
from tqdm import tqdm

vqvae.train()
for epoch in range(num_epochs):
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for batch in tepoch:
            images = batch[0].to(device)  # Move images to GPU
            recon_images, _ = vqvae(images)
            loss = F.mse_loss(recon_images, images)  # Mean Squared Error loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} completed, Loss: {loss.item():.4f}")

# Evaluation (SSIM Scores Calculation)
ssim_scores = []
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

# File to store SSIM scores
csv_file = os.path.join(output_dir, 'ssim_scores.csv')
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image_Index", "SSIM_Score"])  # Write header row

    vqvae.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_idx, batch in enumerate(train_loader):
            original_images = batch[0].to(device)
            recon_images, _ = vqvae(original_images)

            # Iterate over each image in the batch
            for i in range(original_images.size(0)):
                recon_image_np = recon_images[i].squeeze(0).cpu().detach().numpy()
                original_image_np = original_images[i].squeeze(0).cpu().numpy()

                # Compute SSIM score
                ssim_score = evaluate_ssim(original_image_np, recon_image_np)
                ssim_scores.append(ssim_score)

                # Save and display the first 10 images
                if len(ssim_scores) <= 10:
                    display_images_and_ssim(original_image_np, recon_image_np, ssim_score, len(ssim_scores), output_dir)

                # Save SSIM score to CSV
                writer.writerow([len(ssim_scores), ssim_score])

                if len(ssim_scores) >= 10:
                    break
            if len(ssim_scores) >= 10:
                break

# Compute and save average SSIM
average_ssim = np.mean(ssim_scores)
print(f'Average SSIM Score: {average_ssim:.4f}')
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Average_SSIM', average_ssim])

# Plot SSIM scores
plt.plot(ssim_scores)
plt.title('SSIM Scores')
plt.xlabel('Image Index')
plt.ylabel('SSIM Score')
plt.show()
