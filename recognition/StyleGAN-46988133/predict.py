"""
predict.py created by Matthew Lockett 46988133

This file will load the weights of the Discriminator and Generator from the /saved_outputs folder once they have 
finished training from train.py. The weights will be used to initialise the Discriminator and Generator once again,
and will specifically be used to output a UMAP plot discerning between the two ADNI classes AD and CN. It will also 
create a comparison plot of the real images versus the generated images. All generated plots will be saved to the 
/saved_outputs folder.
"""
import os
import umap
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from modules import *
from dataset import load_ADNI_dataset
import torchvision.utils as vutils

IMAGE_SIZE = 64 # Todo: Desired image resolution output
SAMPLES_PER_LABEL = 1000 # Todo: How many samples desired per label class (AD and CN)

# Calculate the image resolution depth
depth = int(math.log2(IMAGE_SIZE / 4)) - 1

# PyTorch Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU.")

# Setup both the generator and discriminator inference models
gen = Generator().to(device)
disc = Discriminator().to(device)

# Load the generator and discriminator inference models
gen.load_state_dict(torch.load(os.path.join(hp.SAVED_OUTPUT_DIR, f"{IMAGE_SIZE}x{IMAGE_SIZE}_gen_final_model.pth"), weights_only=True))
disc.load_state_dict(torch.load(os.path.join(hp.SAVED_OUTPUT_DIR, f"{IMAGE_SIZE}x{IMAGE_SIZE}_disc_final_model.pth"), weights_only=True))
gen.eval()
disc.eval()

# Load the ADNI dataset validation images
image_loader = load_ADNI_dataset(image_size=IMAGE_SIZE, training_set=False)

# REF: The following code was inspired by ChatGPT-o1-preview via the following prompt.
# REF: Prompt: I want to create a UMAP plot by extracting the feature map out of my generator 
# REF: and using it to distinguish between two classes CN and AD. What is the code to do this for a UMAP plot?

# Generate latent vectors and labels: 0 for CN, 1 for AD
latent_vectors = torch.randn(SAMPLES_PER_LABEL * 2, hp.LATENT_SIZE, device=device)
labels = torch.cat([torch.zeros(SAMPLES_PER_LABEL), torch.ones(SAMPLES_PER_LABEL)]).long().to(device)

# Extract the image output and last layer feature map from the generator
with torch.no_grad():

    # Pass the latent vectors and labels through the generator to get the feature map
    gen_images , feature_maps = gen(latent_vectors, labels=labels, depth=depth, alpha=1.0) 
    feature_maps = feature_maps.cpu().numpy()

# Flatten the feature map
flat_feature_maps = feature_maps.reshape(feature_maps.shape[0], -1)

# Initialise the UMAP embedding
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_embedding = umap_model.fit_transform(flat_feature_maps)

# Move labels to the CPU
labels_cpu = labels.cpu().numpy()

################################################ Plot the UMAP Embedding ###########################################################
plt.figure(figsize=(10, 7))
plt.scatter(umap_embedding[labels_cpu == 0, 0], umap_embedding[labels_cpu == 0, 1], label='CN', alpha=0.6, s=50)
plt.scatter(umap_embedding[labels_cpu == 1, 0], umap_embedding[labels_cpu == 1, 1], label='AD', alpha=0.6, s=50)
plt.legend()
plt.title(f'{IMAGE_SIZE}x{IMAGE_SIZE} UMAP Projection of Generator Last Layer Feature Map (After Label Embedding)')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig(os.path.join(hp.SAVED_OUTPUT_DIR, f"{IMAGE_SIZE}x{IMAGE_SIZE}_AD_and_CN_UMAP_plot.png"), pad_inches=0)
plt.close()

#################################### Plot the Real Images vs Generated Images #######################################################
# Plot the real images
real_images, labels = next(iter(image_loader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title(f"{IMAGE_SIZE}x{IMAGE_SIZE} Real Images from ADNI Dataset")
plt.imshow(np.transpose(vutils.make_grid(real_images.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images created by the Generator
plt.subplot(1,2,2)
plt.axis("off")
plt.title(f"{IMAGE_SIZE}x{IMAGE_SIZE} Generated Images")
plt.imshow(np.transpose(vutils.make_grid(gen_images[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig(os.path.join(hp.SAVED_OUTPUT_DIR, f"{IMAGE_SIZE}x{IMAGE_SIZE}_real_versus_fake_images.png"), bbox_inches='tight', pad_inches=0)
plt.close()