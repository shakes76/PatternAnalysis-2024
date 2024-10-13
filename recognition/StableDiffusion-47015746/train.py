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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# List to store images for multiple points in each epoch
img_list = []

img_list_encoder = []

def visualize_images_final(final_reconstructed):
    # Move images back to CPU and detach
    final_reconstructed = final_reconstructed.cpu().detach()

    # Unnormalize images to bring them from [-1, 1] to [0, 1] for visualization
    def unnormalize(imgs):
        return imgs * 0.5 + 0.5

    final_reconstructed = unnormalize(final_reconstructed)

    # Take the first image from batch for the GIF and store it
    img_list.append(final_reconstructed[0].numpy())

def visualize_images_encoder(encoder_reconstructed):
    # Move images back to CPU and detach
    encoder_reconstructed = encoder_reconstructed.cpu().detach()

    # Unnormalize images to bring them from [-1, 1] to [0, 1] for visualization
    def unnormalize(imgs):
        return imgs * 0.5 + 0.5

    encoder_reconstructed = unnormalize(encoder_reconstructed)

    # Take the first image from batch for the GIF and store it
    img_list_encoder.append(encoder_reconstructed[0].numpy())



# Training loop for VAE with optimizer defined inside the function
def train_vae(vae, dataloader, lr=1e-4, epochs=10):
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)  # Define optimizer within the function
    criterion = nn.MSELoss()  # Use MSE for image reconstruction

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass
            reconstructed, _ = vae(images)
            loss = criterion(reconstructed, images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, VAE Loss: {total_loss:.4f}")
        visualize_images_encoder(reconstructed)


    # Save the model's state dictionary after training completes
    torch.save(vae.state_dict(), 'vae_state_dict.pth')
    print("Model saved as 'vae_state_dict.pth'")

    
    # Create and save GIF
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(img, (1, 2, 0)), animated=True)] for img in img_list_encoder]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save('training_progress_encoder.gif', writer='imagemagick', fps=10)
    plt.close(fig)

def train_diffusion_model(model, dataloader, epochs=10, lr=1e-3, beta=1.0, weight_smooth_l1=0.5):
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
        visualize_images_final(output_images)
        visualize_images_encoder(reconstructed_from_latent)
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


    vae = VAE(in_channels=3, latent_dim=128, out_channels=3).to(device)
    train_vae(vae, dataloader, lr=1e-4, epochs=10)



    #stable_diffusion_model = StableDiffusionModel(latent_dim, timesteps)

    # Train the model
    #train_model(stable_diffusion_model, dataloader)

