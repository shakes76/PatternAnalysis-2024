'''
[desc]
for prediting with the model
gives n_images from both classes

gives 64 of test images to compare

shwos umap embedding plot

@author Jamie Westerhout
@project Stable Diffusion
@date 2024
'''
import torch
import umap
import torchvision.utils as tvutils

from utils import *
from modules import *
from train import latent_dim, num_timesteps, noise_scheduler, data_loader_test, device

#umap reducer for plotting
reducer = umap.UMAP(min_dist=0, n_neighbors=35)

#load in models
vae_encoder = torch.load("models/encoder.model", weights_only=False)
vae_encoder.eval()
vae_decoder = torch.load("models/decoder.model", weights_only=False)
vae_decoder.eval()
unet = torch.load("models/unet.model", weights_only=False)
unet.eval()

if __name__ == '__main__':
    n_images = 10
    # Display the generated images
    sample_images = generate_sample(0, unet, vae_decoder, vae_encoder, latent_dim, num_timesteps, noise_scheduler, num_samples=n_images)
    display_images(sample_images, title="fully generated AD brain images")

    sample_images = generate_sample(0, unet, vae_decoder, vae_encoder, latent_dim, num_timesteps, noise_scheduler, num_samples=n_images)
    display_images(sample_images, title="fully generated AD brain images")

    #show test images
    test_images = next(iter(data_loader_test))[0]
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("64 Samples of Test Images")
    plt.imshow(np.transpose(tvutils.make_grid(test_images.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    #umap embedding plot
    umap_image_count = 10

    embedded0 = generate_sample_latent(0, unet, vae_decoder, vae_encoder, latent_dim, num_timesteps, num_samples=umap_image_count).view(-1,512*13*7)
    embedded1 = generate_sample_latent(1, unet, vae_decoder, vae_encoder, latent_dim, num_timesteps, num_samples=umap_image_count).view(-1,512*13*7)
    embedded = torch.cat((embedded0,embedded1),dim=0)

    embedding = reducer.fit_transform(embedded.cpu().numpy())
    plt.scatter(embedding[:, 0], embedding[:, 1], c=[0] * umap_image_count *10 + [1] * umap_image_count*10, cmap='Spectral', s=10)
    plt.title("Umap Embedding plot")
    plt.show()