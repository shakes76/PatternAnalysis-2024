from modules import Encoder, Decoder, VQEmbedLayer, Model
import dataset
import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(111)

trainSet, testSet = dataset.init()

dimensions = {
    "size":(256, 128),
    "input": 1, 
    "hidden": 256, 
    "latent": 128,
    "embeddings": 512,
    "output": 1,
    "commitment_beta": 0.25
}

parameters = {
    "lr": 2e-4, 
    "epochs": 50, 
    "batch": 100,
    "gpu": "cuda",
    "cpu": "cpu",
    "loss": nn.MSELoss()
}

device = parameters["device"] if torch.cuda.is_available() else parameters["cpu"]
print(f"Using {device}.\n")

print(f"Train.py: Shape of first training dataset image {trainSet[0].shape}")
print(f"Train.py: Shape of first testing dataset image {testSet[0].shape}")

encoder = Encoder(input_dim=dimensions["input"], hidden_dim=dimensions["hidden"], output_dim=dimensions["latent"])
embeddings = VQEmbedLayer(embeddings=dimensions["embeddings"], embed_dim=dimensions["latent"])
decoder = Decoder(latent_dim=dimensions["latent"], hidden_dim=dimensions["hidden"], output_dim=dimensions["output"])

vqvae = Model(Encoder=encoder, VQEmbeddings=embeddings, Decoder=decoder).to(device=device)