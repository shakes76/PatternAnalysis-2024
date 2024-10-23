#s4701574

#CHATGPT was used to assist with writing some of this code


import torch
import torch.optim as optim
import torch
import torch.nn.functional as F
from modules import *
from dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


 
def train_vqvae(model, dataloader, epochs, device, lr= 0.00001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()  # Set the model in training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, z, quant_losses = model(images)

            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(outputs, images)

            # Codebook loss
            codebook_loss = quant_losses['codebook_loss']
            commitment_loss = quant_losses['commitment_loss']
            # Total loss
            total_loss = recon_loss + codebook_loss + commitment_loss
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        
        print(f'End of Epoch {epoch+1}, Average Loss: {running_loss/len(dataloader):.4f}')
    # Save the model's state dictionary after training completes
    torch.save(model.state_dict(), 'VQVAE_state_dict.pth')
    print("Model saved as 'VQVAE_state_dict.pth'")
    print("Training finished")





def train_diffusion_model(diffusion_model, dataloader, optimizer, epochs=50):
    diffusion_model.unet.train()  # Only UNet should be in training mode
    epoch_losses = []  # List to store loss for each epoch
    
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

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)  # Store the average loss
        print(f"Epoch {epoch + 1}/{epochs}, Diffusion Model Loss: {avg_loss:.4f}")


    # Save the model's state dictionary after training completes
    torch.save(diffusion_model.state_dict(), 'new_diffusion_model_state_dict.pth')
    print("Model saved as 'new_diffusion_model_state_dict.pth'")

