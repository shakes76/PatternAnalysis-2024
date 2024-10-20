import torch
from constants import w_dim, log_resolution
import torch.nn as nn

"""
 This function samples a random latent space vector z and passes it through our mapping network
 in order to generate the stylised vector space.
"""
def get_w(batch_size, mapping_network, device):

    z = torch.randn(batch_size, w_dim).to(device)
    w = mapping_network(z)
    # Expand w from the generator blocks
    return w[None, :, :].expand(log_resolution, -1, -1)

"""
 Generates random noise used in generated images - in order to introduce variability of features.
 Alongside being the noise that is controlled and stylised in StyleGAN.
"""
def get_noise(batch_size, device):
    
    noise = []
    #noise res starts from 4x4
    resolution = 4

    # For each gen block
    for i in range(log_resolution):
        # First block uses 3x3 conv
        if i == 0:
            n1 = None
        # For rest of conv layer
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device) # Generating tensors of batch_size-many res x res noise blocks.
        noise.append((n1, n2))

        # Upscaling our resolution in concurrence with generator architecture
        resolution *= 2

    return noise


# Initialize BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()

def discriminator_loss(real_output, fake_output):
    # Calculate loss for real and fake outputs
    real_loss = criterion(real_output, torch.ones_like(real_output)) # As discriminator is trying to identify real images as 1s
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output)) # As discriminator is trying to identify fake images as 0s
    
    # Total loss is sum of real and fake loss
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    # Compares discriminator's "evaluation" of the generated images to 1s. 
    # This is because the generator wants the discriminator to think its images are 1s (real). 
    return criterion(fake_output, torch.ones_like(fake_output))