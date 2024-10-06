import math
import numpy as np
import torch
from modules import Model, Decoder, Encoder, Quantise
from utils import parameters, dimensions, SaveOneImage, SaveMultipleImages
from dataset import GetTestSet

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

test_set = GetTestSet()
test_loader = DataLoader(test_set, batch_size=parameters["batch"], shuffle=False)

device = torch.device(parameters["gpu"] if torch.cuda.is_available() else parameters["cpu"])
print(f"VQVAE being tested on {device}.\n")

encoder = Encoder(input_dim=dimensions["input"], hidden_dim=dimensions["hidden"], output_dim=dimensions["latent"])
quantise = Quantise(n_embeddings=dimensions["embeddings"], embed_dim=dimensions["latent"])
decoder = Decoder(input_dim=dimensions["latent"], hidden_dim=dimensions["hidden"], output_dim=dimensions["output"])

vqvae = Model(Encoder=encoder, Quantise=quantise, Decoder=decoder).to(device)
vqvae.load_state_dict(torch.load('hipmri_vqvae.pth', weights_only=True))
vqvae.eval()

with torch.no_grad():
    test_iter = iter(test_loader)
    test_images = next(test_iter)

    test_images = test_images.to(device)

    SaveOneImage(test_images, "one_actual", "Actual Scan")
    SaveMultipleImages(test_images, "multiple_actual", "Actual Scans")

    pred, commitment_loss, codebook_loss, perplexity = vqvae(test_images)

    SaveOneImage(pred, "one_reconstructed", "Reconstructed Scan")
    SaveMultipleImages(pred, "multiple_reconstructed", "Reconstructed Scans")

    print(f"commitment loss = {commitment_loss.item()}, codebook loss = {codebook_loss.item()}, perplexity = {perplexity.item()}")


