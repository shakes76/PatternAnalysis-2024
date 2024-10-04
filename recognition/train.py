from modules import Encoder, Decoder, Quantise, Model
import dataset
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(111)

dimensions = {
    "size":(256, 144),
    "input": 1, 
    "hidden": 512, 
    "latent": 16,
    "embeddings": 512,
    "output": 1,
    "commitment_beta": 0.25
}

parameters = {
    "lr": 2e-4, 
    "epochs": 50, 
    "batch": 100,
    "gpu": "cuda",
    "cpu": "cpu"
}

trainSet = dataset.GetTrainSet()
trainLoader = DataLoader(trainSet, batch_size=parameters["batch"], shuffle=True)

device = torch.device(parameters["gpu"] if torch.cuda.is_available() else parameters["cpu"])
print(f"VQVAE being trained on {device}.\n")

print(f"Train.py: Shape of first training dataset image {trainSet[0].shape}")

encoder = Encoder(input_dim=dimensions["input"], hidden_dim=dimensions["hidden"], output_dim=dimensions["latent"])
quantise = Quantise(n_embeddings=dimensions["embeddings"], embed_dim=dimensions["latent"])
decoder = Decoder(input_dim=dimensions["latent"], hidden_dim=dimensions["hidden"], output_dim=dimensions["output"])

vqvae = Model(Encoder=encoder, Quantise=quantise, Decoder=decoder).to(device=device)

mse_loss = nn.MSELoss()
optimiser = Adam(vqvae.parameters(), lr=parameters["lr"])

print("Training the Vector Quantised Variational Autoencoder...")
vqvae.train()
for epoch in range(parameters["epochs"]):
    running_loss = 0.0
    # Labels not being outputted?
    for i, image in enumerate(trainLoader):
        image = image.to(device)

        optimiser.zero_grad()
        
        imageRec, commitment_loss, codebook_loss, perplexity = vqvae(image)
        reconstr_loss = mse_loss(imageRec, image)

        loss = reconstr_loss + commitment_loss * dimensions["commitment_beta"] + codebook_loss
        running_loss += loss.item()

        loss.backward()
        optimiser.step()

        if i % 100 == 0:
            print(f"Loss for {epoch + 1},{i + 1} = {loss.item()}\nreconstruction loss = {reconstr_loss.item()}, perplexity: {perplexity.item()}, commitment loss: {commitment_loss.item()}, codebook loss: {codebook_loss.item()}")
    print(f"Epoch {epoch + 1} loss: {running_loss}.")

#TBD

