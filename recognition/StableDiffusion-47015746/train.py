import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from modules import *
from dataset import *


def visualize_images(original_images, reconstructed_from_latent, final_reconstructed, epoch):
    # Move images back to CPU and detach
    original_images = original_images.cpu().detach()
    reconstructed_from_latent = reconstructed_from_latent.cpu().detach()
    final_reconstructed = final_reconstructed.cpu().detach()

    # Unnormalize images to bring them from [-1, 1] to [0, 1] for visualization
    def unnormalize(imgs):
        return imgs * 0.5 + 0.5

    original_images = unnormalize(original_images)
    reconstructed_from_latent = unnormalize(reconstructed_from_latent)
    final_reconstructed = unnormalize(final_reconstructed)

    # Display first 8 images
    num_images = 8
    fig, axes = plt.subplots(3, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Original Images
        axes[0, i].imshow(np.transpose(original_images[i].numpy(), (1, 2, 0)))
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        # Reconstructed from Latent (Encoder → Decoder)
        axes[1, i].imshow(np.transpose(reconstructed_from_latent[i].numpy(), (1, 2, 0)))
        axes[1, i].axis('off')
        axes[1, i].set_title("From Latent")

        # Final Reconstructed (Full Model Output)
        axes[2, i].imshow(np.transpose(final_reconstructed[i].numpy(), (1, 2, 0)))
        axes[2, i].axis('off')
        axes[2, i].set_title("Final Output")

    plt.suptitle(f"Epoch {epoch} - Model Outputs")
    plt.show()

def train_model(model, dataloader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        for images, _ in dataloader:
            images = images.to(device)
            t = torch.randint(0, model.timesteps, (images.size(0),)).long().to(device)

            latent, noisy_latent, denoised_latent, output_images = model(images, t)

            # Calculate losses
            reconstructed_from_latent = model.decoder(latent)
            encoder_loss = F.mse_loss(reconstructed_from_latent, images)
            unet_loss = F.mse_loss(denoised_latent, latent)
            decoder_loss = F.mse_loss(output_images, images)
            total_loss = encoder_loss + unet_loss + decoder_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Encoder Loss: {encoder_loss.item():.4f}, "
              f"UNet Loss: {unet_loss.item():.4f}, Decoder Loss: {decoder_loss.item():.4f}, "
              f"Total Loss: {total_loss.item():.4f}")
        
        # Display images after each epoch
        visualize_images(images, reconstructed_from_latent, output_images, epoch + 1)
        
# Make sure this runs only when executed directly
if __name__ == '__main__':
    # Load data
    data_root = "C:/Users/msi/Desktop/AD_NC" 
    dataloader = load_data(data_root)

    # Instantiate the model
    latent_dim = 128
    timesteps = 1000
    stable_diffusion_model = StableDiffusionModel(latent_dim, timesteps)

    # Train the model
    train_model(stable_diffusion_model, dataloader)

