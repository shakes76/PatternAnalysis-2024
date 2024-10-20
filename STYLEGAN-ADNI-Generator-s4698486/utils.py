import torch
from constants import w_dim, log_resolution

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