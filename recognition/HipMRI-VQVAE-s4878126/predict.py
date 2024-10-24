import math
import numpy as np
import torch
from modules import Model, Decoder, Encoder, Quantise
from utils import parameters, dimensions, SaveOneImage, SaveMultipleImages
from dataset import GetTestSet

from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

import matplotlib.pyplot as plt

# Fetch the testing set from Rangpur.
test_set = GetTestSet()
test_loader = DataLoader(test_set, batch_size=parameters["batch"], shuffle=False)

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print('cuda is available.')
else:
    print('cuda not available')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"VQVAE being tested on {device}.\n")

encoder = Encoder(inputDim=dimensions["input"], hiddenDim=dimensions["hidden"], outputDim=dimensions["latent"])
quantise = Quantise(nEmbeddings=dimensions["embeddings"], embed_dim=dimensions["latent"])
decoder = Decoder(inputDim=dimensions["latent"], hiddenDim=dimensions["hidden"], outputDim=dimensions["output"])

# Initialise the SSIM measure
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

vqvae = Model(Encoder=encoder, Quantise=quantise, Decoder=decoder).to(device)

# Check whether the model is running inference on a GPU or a CPU. If not confirmed beforehand, PyTorch will throw an error when loading the model weights.
if device != 'cuda':
    vqvae.load_state_dict(torch.load('hipmri_vqvae.pth', weights_only=True, map_location=torch.device('cpu')))
else:
    vqvae.load_state_dict(torch.load('hipmri_vqvae.pth', weights_only=True))

# Set the model to testing mode.
vqvae.eval()

with torch.no_grad():
    test_iter = iter(test_loader)
    test_images = next(test_iter)

    test_images = test_images.to(device)
    print(test_images.shape)

    # Save both a single actual image for comparison and multiple images to prove the model's ability to handle a variety of MRI scans.
    SaveOneImage(test_images, "one_actual", "Actual Scan")
    SaveMultipleImages(test_images, "multiple_actual", "Actual Scans")

    pred, commitment_loss, codebook_loss, perplexity = vqvae(test_images)
    print(pred.shape)
    pred.to(device)

    # Save both a single image for clarity and multiple images to prove the model's ability to generate a variety of scans.
    SaveOneImage(pred, "one_reconstructed", "Reconstructed Scan")
    SaveMultipleImages(pred, "multiple_reconstructed", "Reconstructed Scans")

    print(f"commitment loss = {commitment_loss.item()}, codebook loss = {codebook_loss.item()}, perplexity = {perplexity.item()}")
    
    # Calculate the SSIM over the current batch of images
    print(f"Structural Similarity Index Measure between original and reconstructed images = {ssim(pred, test_images)}")
