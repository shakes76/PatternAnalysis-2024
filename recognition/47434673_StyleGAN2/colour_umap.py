"""
Author: Thomas Barros
Date: October 2024

Contains the UMAP functionality. 
"""


from tqdm import tqdm  # For progress bars
import torch

import config
import utils


def sample_style_codes(mapping_network, num_samples, z_dim, batch_size, device):
    """
    Generates a specified number of style codes by sampling random latent 
    vectors z and passing them through the mapping network of a StyleGAN2 model.
    """
    style_codes = []
    num_batches = num_samples // batch_size
    for _ in tqdm(range(num_batches)):
        z = torch.randn(batch_size, z_dim).to(device)
        w = mapping_network(z)
        style_codes.append(w.cpu())
    # Handle any remaining samples
    remaining_samples = num_samples % batch_size
    if remaining_samples > 0:
        z = torch.randn(remaining_samples, z_dim).to(device)
        w = mapping_network(z)
        style_codes.append(w.cpu())
    style_codes = torch.cat(style_codes, dim=0)
    return style_codes


def generate_images_from_style_codes(generator, style_codes, batch_size, device):
    """
    Takes the sampled style codes and generates images using the generator component of the StyleGAN2 model.
    """
    images_list = []
    num_samples = style_codes.size(0)
    num_batches = num_samples // batch_size
    generator.eval()
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            w_batch = style_codes[i*batch_size:(i+1)*batch_size].to(device)
            # Expand w to match the expected input shape for the generator
            w = w_batch[None, :, :].expand(config.log_resolution, -1, -1)
            noise = utils.get_noise(batch_size, device)
            images = generator(w, noise)
            images_list.append(images.cpu())
        # Handle remaining samples
        remaining_samples = num_samples % batch_size
        if remaining_samples > 0:
            w_batch = style_codes[-remaining_samples:].to(device)
            w = w_batch[None, :, :].expand(config.log_resolution, -1, -1)
            noise = utils.get_noise(remaining_samples, device)
            images = generator(w, noise)
            images_list.append(images.cpu())
    images = torch.cat(images_list, dim=0)
    return images


def compute_mean_intensity(images):
    """
    Calculates the mean pixel intensity for each image in a batch of images.
    """
    # Assuming images are in the range [-1, 1], convert to [0, 1]
    images = (images + 1) / 2
    mean_intensities = images.mean(dim=[1, 2, 3])  # Mean over channels and spatial dimensions
    return mean_intensities.numpy()

