import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from modules import *
from dataset import *
import matplotlib.animation as animation

# List to store images for multiple points in each epoch
img_list = []

def visualize_images(final_reconstructed):
    # Move images back to CPU and detach
    final_reconstructed = final_reconstructed.cpu().detach()

    # Unnormalize images to bring them from [-1, 1] to [0, 1] for visualization
    def unnormalize(imgs):
        return imgs * 0.5 + 0.5

    final_reconstructed = unnormalize(final_reconstructed)

    # Take the first image from batch for the GIF and store it
    img_list.append(final_reconstructed[0].numpy())

def train_model(model, dataloader, epochs=5, lr=1e-4, beta=1.0, weight_smooth_l1=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode

        for images, _ in dataloader:
            images = images.to(device)
            t = torch.randint(0, model.timesteps, (images.size(0),)).long().to(device)

            latent, noisy_latent, denoised_latent, output_images = model(images, t)

            # Calculate losses with both MSE and Smooth L1 Loss
            reconstructed_from_latent = model.decoder(latent)
            
            encoder_loss = F.mse_loss(reconstructed_from_latent, images)
            
            unet_loss = F.mse_loss(denoised_latent, latent)

            decoder_loss = F.mse_loss(output_images, images)

            # Combine all losses
            total_loss = encoder_loss + unet_loss + decoder_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Encoder Loss: {encoder_loss.item():.4f}, "
              f"UNet Loss: {unet_loss.item():.4f}, Decoder Loss: {decoder_loss.item():.4f}, "
              f"Total Loss: {total_loss.item():.4f}")
        
        # Save output images for each epoch
        visualize_images(output_images)
    # Create and save GIF
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(img, (1, 2, 0)), animated=True)] for img in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save('training_progress.gif', writer='imagemagick', fps=10)
    plt.close(fig)
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

