import torch
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
from modules import *
from dataset import *
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# List to store images for multiple points in each epoch
img_list = []

img_list_encoder = []





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
        


    # Save the model's state dictionary after training completes
    torch.save(vae.state_dict(), 'vae_state_dict.pth')
    print("Model saved as 'vae_state_dict.pth'")


    






# Instantiate and train only the diffusion model
# Adjust the training function
def train_diffusion_model(diffusion_model, dataloader, optimizer, epochs=100):
    diffusion_model.unet.train()  # Only UNet should be in training mode
    epoch_losses = []  # List to store loss for each epoch
    #learning_rates = []

    # Calculate an abstract milestone as half the total epochs
    #milestone = epochs // 4

    # Define schedulers without the verbose parameter
    #sched_linear_1 = lr_scheduler.CyclicLR(
        #optimizer, base_lr=0.001, max_lr=0.01, step_size_up = milestone // 2, step_size_down = milestone // 2, mode="triangular"
    #)
    #sched_linear_3 = lr_scheduler.LinearLR(
        #optimizer, start_factor= 1, end_factor= 0.01
    #)
    #scheduler = lr_scheduler.SequentialLR(
        #optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[milestone]
    #)

    # Training loop with the two-phase learning rate schedule
    for epoch in range(epochs):
        total_loss = 0  # Reset total loss for the current epoch

        for images, labels in dataloader:
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
        #learning_rates.append(current_lr)
        #print(f"Epoch {epoch + 1}/{epochs}, Learning Rate: {current_lr:.6f}, Diffusion Model Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch + 1}/{epochs}, Diffusion Model Loss: {avg_loss:.4f}")
        #final = diffusion_model.noise_scheduler.remove_noise(noisy_latent, predicted_noise, t)    
        #visualize_images(diffusion_model.decoder(latent), diffusion_model.decoder(latent), diffusion_model.decoder(latent), diffusion_model.decoder(latent), epoch + 1)

    # Save the model's state dictionary after training completes
    torch.save(diffusion_model.state_dict(), 'diffusion_model_state_dict.pth')
    print("Model saved as 'diffusion_model_state_dict.pth'")

    # After your training completes and you've filled `epoch_losses` and `learning_rates`
    data = {
        'Epoch Losses': epoch_losses
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel('training_metrics.xlsx', index=False)



# Make sure this runs only when executed directly


if __name__ == '__main__':
    # Load data
    data_train = "C:/Users/msi/Desktop/AD_NC/train" 
    data_test = "C:/Users/msi/Desktop/AD_NC/test" 
    #data_train = "/home/groups/comp3710/ADNI/AD_NC/train"
    #data_test = "/home/groups/comp3710/ADNI/AD_NC/test"
    dataloader = load_data(data_train, data_test)

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
    unet = UNet().to(device)
    noise_scheduler = NoiseScheduler(timesteps=1000)
    unet.train()
    # Instantiate diffusion model with frozen encoder and decoder
    diffusion_model = DiffusionModel(encoder, unet, decoder, noise_scheduler).to(device)

    # Define optimizer for only the UNet part
    diffusion_optimizer = torch.optim.Adam(diffusion_model.unet.parameters(), lr= 0.00001)

    # Train the diffusion model as before
    train_diffusion_model(diffusion_model, dataloader, diffusion_optimizer, epochs=100)


    #stable_diffusion_model = StableDiffusionModel(latent_dim, timesteps)

    # Train the model
    #train_model(stable_diffusion_model, dataloader)

