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
def train_vae(vae, dataloader, lr=1e-4, epochs=50):
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






# Instantiate and train only the diffusion model
# Adjust the training function
def train_diffusion_model(diffusion_model, dataloader, optimizer, epochs=10):
    diffusion_model.unet.train()  # Only UNet should be in training mode
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()

            # Random timestep for diffusion
            t = torch.randint(0, diffusion_model.noise_scheduler.timesteps, (images.size(0),)).long().to(device)

            # Forward pass through diffusion model
            denoised_latent, output_images = diffusion_model(images, t)

            # Calculate loss for denoising
            loss = criterion(output_images, images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Diffusion Model Loss: {total_loss / len(dataloader):.4f}")



# Make sure this runs only when executed directly


if __name__ == '__main__':
    # Load data
    data_root = "C:/Users/msi/Desktop/AD_NC" 
    dataloader = load_data(data_root)

    # Instantiate the model
    latent_dim = 128
    timesteps = 1000

    # Load the pre-trained VAE (encoder and decoder) and instantiate a new UNet and noise scheduler
    pre_trained_vae = VAE(in_channels=3, latent_dim=128, out_channels=3).to(device)
    pre_trained_vae.load_state_dict(torch.load("vae_state_dict.pth"))
    pre_trained_vae.eval()
    encoder = pre_trained_vae.encoder  # Use pre-trained encoder
    decoder = pre_trained_vae.decoder  # Use pre-trained decoder

    # Define the UNet and noise scheduler for the diffusion model
    unet = UNet(latent_dim=128).to(device)
    noise_scheduler = NoiseScheduler(timesteps=1000)
    unet.train()
    # Instantiate diffusion model with frozen encoder and decoder
    diffusion_model = DiffusionModel(encoder, unet, decoder, noise_scheduler).to(device)

    # Define optimizer for only the UNet part
    diffusion_optimizer = torch.optim.Adam(diffusion_model.unet.parameters(), lr=1e-4)

    # Train the diffusion model as before
    train_diffusion_model(diffusion_model, dataloader, diffusion_optimizer, epochs=10)


    #stable_diffusion_model = StableDiffusionModel(latent_dim, timesteps)

    # Train the model
    #train_model(stable_diffusion_model, dataloader)

