import modules
import dataset
import torch
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
    "cpu": "cpu"
}

device = parameters["device"] if torch.cuda.is_available() else parameters["cpu"]
print(f"Using {device}.\n")

print(f"Train.py: Shape of first training dataset image {trainSet[0].shape}")
print(f"Train.py: Shape of first testing dataset image {testSet[0].shape}")




