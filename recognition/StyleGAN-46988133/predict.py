"""
predict.py created by Matthew Lockett 46988133
"""
import os
import umap
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from modules import *
from dataset import load_ADNI_dataset

# Todo: Desired image resolution that requires a Umap and other visualisations
IMAGE_SIZE = 8

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

# Stores the features and labels for each batch
features_list = []
labels_list = []
depth = int(math.log2(IMAGE_SIZE / 4)) - 1
print(depth)

# REF: The following code was inspired by ChatGPT-o1-preview via the following prompt.
# REF: Prompt: I want to save my model with pytorch after training then load it again 
# REF: in another file to do the UMAP plot, how do I do this?

with torch.no_grad():

    for images, labels in image_loader:

        # Apply to the GPU
        images = images.to(device)
        labels = labels.to(device)

        # Get the classifications from the discriminator on the dataset images
        real_fake_output, class_output, features = disc(images, labels, depth=depth, alpha=1.0)

        # Save the features and labels for UMAP plotting 
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

# Merge all features and labels together
all_features = np.concatenate(features_list, axis=0)
all_labels = np.concatenate(labels_list, axis=0)

# Apply the UMAP to reduce the dimensions of features
reducer = umap.UMAP(n_components=hp.LABEL_DIMENSIONS)
embedding = reducer.fit_transform(all_features)

# Plot the UMAP embedding
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=all_labels, cmap='coolwarm', alpha=0.7)
plt.colorbar()
plt.title('UMAP Projection of Discriminator Features')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')

# Adds a legend for the two ADNI dataset image classes
legend_labels = {0: 'CN', 1: 'AD'}
handles, _ = scatter.legend_elements()
plt.legend(handles, [legend_labels[int(label)] for label in np.unique(all_labels)])
plt.savefig(os.path.join(hp.SAVED_OUTPUT_DIR, f"{IMAGE_SIZE}x{IMAGE_SIZE}_AD_and_CN_UMAP_plot.png"), pad_inches=0)
plt.close()