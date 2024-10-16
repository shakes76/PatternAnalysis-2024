import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from modules import *
from dataset import *
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.animation as animation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# List to store images for multiple points in each epoch
img_list = []

img_list_encoder = []

def visualize_images(im1, im2, im3, im4, epoch):
    # Move images back to CPU and detach
    im1= im1.cpu().detach()
    im2= im2.cpu().detach()
    im3= im3.cpu().detach()
    im4= im4.cpu().detach()
    

    # Unnormalize images to bring them from [-1, 1] to [0, 1] for visualization
    def unnormalize(imgs):
        return imgs * 0.5 + 0.5

    im1= unnormalize(im1)
    im2= unnormalize(im2)
    im3= unnormalize(im3)
    im4= unnormalize(im4)
    

    # Display first 8 images
    num_images = 8
    fig, axes = plt.subplots(4, num_images, figsize=(15, 5))

    for i in range(num_images):
        # Original Images
        axes[0, i].imshow(np.transpose(im1[i].numpy(), (1, 2, 0)))
        axes[0, i].axis('off')
        axes[0, i].set_title("Original Image")

        # Reconstructed from Latent (Encoder → Decoder)
        axes[1, i].imshow(np.transpose(im2[i].numpy(), (1, 2, 0)))
        axes[1, i].axis('off')
        axes[1, i].set_title("Image After Noise")

        # Final Reconstructed (Full Model Output)
        axes[2, i].imshow(np.transpose(im3[i].numpy(), (1, 2, 0)))
        axes[2, i].axis('off')
        axes[2, i].set_title("Predicted Noise")

        # Final Reconstructed (Full Model Output)
        axes[3, i].imshow(np.transpose(im4[i].numpy(), (1, 2, 0)))
        axes[3, i].axis('off')
        axes[3, i].set_title("Image After Noise - Predicted Noise")


    plt.suptitle(f"Epoch {epoch} - Model Outputs")
    plt.show()

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
    criterion = F.smooth_l1_loss()  # Use MSE for image reconstruction

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
def train_diffusion_model(diffusion_model, dataloader, optimizer, epochs=100):
    diffusion_model.unet.train()  # Only UNet should be in training mode
    epoch_losses = []  # List to store loss for each epoch

    # Calculate an abstract milestone as half the total epochs
    #milestone = epochs // 2

    # Define schedulers without the verbose parameter
    #sched_linear_1 = lr_scheduler.CyclicLR(
        #optimizer, base_lr=0.005, max_lr=0.05, step_size_up = milestone // 2, step_size_down = milestone // 2, mode="triangular"
    #)
    #sched_linear_3 = lr_scheduler.LinearLR(
        #optimizer, start_factor= 1, end_factor= 0.1
    #)
    #scheduler = lr_scheduler.SequentialLR(
        #optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[milestone]
    #)

    # Training loop with the two-phase learning rate schedule
    for epoch in range(epochs):
        total_loss = 0  # Reset total loss for the current epoch

        for images, _ in dataloader:
            images = images.to(device)
            
            optimizer.zero_grad()

            # Random timestep for diffusion
            t = torch.randint(0, diffusion_model.noise_scheduler.timesteps, (images.size(0),)).long().to(device)

            # Forward pass through diffusion model
            latent, noisy_latent, predicted_noise, noise = diffusion_model(images, t)

            # Calculate loss for denoising
            loss = F.smooth_l1_loss(noise, predicted_noise)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # Accumulate loss

        # Step through SequentialLR scheduler once per epoch
        #scheduler.step()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)  # Store the average loss

        # Print the current learning rate using get_last_lr
        #current_lr = scheduler.get_last_lr()[0]  # get_last_lr() returns a list
        #print(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {current_lr:.6f}, Diffusion Model Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch + 1}/{epochs}, Diffusion Model Loss: {avg_loss:.4f}")

        visualize_images(diffusion_model.decoder(latent), diffusion_model.decoder(noisy_latent), diffusion_model.decoder(predicted_noise), diffusion_model.decoder(noisy_latent - predicted_noise), epoch + 1)

    # Save the model's state dictionary after training completes
    torch.save(diffusion_model.state_dict(), 'diffusion_model_state_dict.pth')
    print("Model saved as 'diffusion_model_state_dict.pth'")

    # Plot loss vs. epoch
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("loss_vs_epoch.png")  # Save the figure
    plt.show()

    # Create and save GIF
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(img, (1, 2, 0)), animated=True)] for img in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save('training_progress_encoder.gif', writer='imagemagick', fps=10)
    plt.close(fig)


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
    diffusion_optimizer = torch.optim.Adam(diffusion_model.unet.parameters(), lr=1e-3)

    # Train the diffusion model as before
    train_diffusion_model(diffusion_model, dataloader, diffusion_optimizer, epochs=100)


    #stable_diffusion_model = StableDiffusionModel(latent_dim, timesteps)

    # Train the model
    #train_model(stable_diffusion_model, dataloader)

