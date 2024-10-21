from modules import Encoder, Decoder, Quantise, Model
from utils import parameters, dimensions
import dataset
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

print(f"CUDA available? {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"VQVAE is being trained on {device}.\n")
else:
    print(f"GPU not available, VQVAE is switching to {device}.\n")

train_set = dataset.GetTrainSet()
train_loader = DataLoader(train_set, batch_size=parameters["batch"], shuffle=True)

print(f"Train.py: Shape of first training dataset image {train_set[0].shape}")

encoder = Encoder(input_dim=dimensions["input"], hidden_dim=dimensions["hidden"], output_dim=dimensions["latent"])
quantise = Quantise(n_embeddings=dimensions["embeddings"], embed_dim=dimensions["latent"])
decoder = Decoder(input_dim=dimensions["latent"], hidden_dim=dimensions["hidden"], output_dim=dimensions["output"])

vqvae = Model(Encoder=encoder, Quantise=quantise, Decoder=decoder).to(device=device)

mse_loss = nn.MSELoss()
optimiser = Adam(vqvae.parameters(), lr=parameters["lr"])

print("Training the Vector Quantised Variational Autoencoder...")
vqvae.train()

reconloss_values = []
cbloss_values = []
cmloss_values = []
pxloss_values =  []

for epoch in range(parameters["epochs"]):
    running_loss = 0.0
    # Labels not being outputted?
    for i, image in enumerate(train_loader):
        image = image.to(device)

        optimiser.zero_grad()
        
        imageRec, commitment_loss, codebook_loss, perplexity = vqvae(image)
        reconstr_loss = mse_loss(imageRec, image)

        loss = reconstr_loss + commitment_loss * dimensions["commitment_beta"] + codebook_loss
        running_loss += loss.item()

        reconloss_values.append(reconstr_loss)
        cbloss_values.append(codebook_loss)
        cmloss_values.append(commitment_loss)
        pxloss_values.append(perplexity)

        loss.backward()
        optimiser.step()

        if i % 100 == 0:
            print(f"Loss for {epoch + 1},{i + 1} = {loss.item()}\nreconstruction loss = {reconstr_loss.item()}, perplexity: {perplexity.item()}, commitment loss: {commitment_loss.item()}, codebook loss: {codebook_loss.item()}")
    print(f"Epoch {epoch + 1} loss: {running_loss}.")

def lossStats(epochs, reconloss_values, cbloss_values, cmloss_values):
    plt.title(f"Traning Loss for the VQVAE over {epochs} Epochs.")
    plt.plot(epochs, reconloss_values, label="Reconstruction Loss", color='blue')
    plt.plot(epochs, reconloss_values, cbloss_values, label="Codebook Loss", color='orange')
    plt.plot(epochs, reconloss_values, cmloss_values, label="Commitment Loss", color='green')
    plt.legend(loc='upper right', title='Loss values', fontsize='Medium')
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.savefig('VQVAE_loss.png')

def recordPerplexity(epochs, pxloss_values):
    plt.title(f"Perplexity for the VQVAE over {epochs} Epochs.")
    plt.plot(epochs, pxloss_values)
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.savefig('VQVAE_perplexity.png')

lossStats(epochs=parameters["epochs"], reconloss_values=reconloss_values, cbloss_values=cbloss_values, cmloss_values=cmloss_values)
recordPerplexity(epochs=parameters["epochs"], pxloss_values=pxloss_values)

torch.save(vqvae.state_dict(), 'hipmri_vqvae.pth')


