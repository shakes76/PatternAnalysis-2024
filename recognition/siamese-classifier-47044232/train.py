"""
The code used to train the siamese network on the ISIC kaggle challenge dataset.

Made by Joshua Deadman
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch.nn import TripletMarginLoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import config
import modules
from dataset import ISICKaggleChallengeSet
from utils import split_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("WARNING: Using the CPU to train...")

# Make and load datasets
train, test, val = split_data(config.DATAPATH+"/train-metadata.csv")
transforms = v2.Compose([
    v2.RandomRotation(degrees=(0, 10)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_set = ISICKaggleChallengeSet(config.DATAPATH+"/image", train, transforms=transforms)
test_set = ISICKaggleChallengeSet(config.DATAPATH+"/image", test, transforms=transforms)
val_set = ISICKaggleChallengeSet(config.DATAPATH+"/image", val, transforms=transforms)

train_set = DataLoader(train_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)
test_set = DataLoader(test_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)
val_set = DataLoader(val_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)

resnet = modules.resnet50(weights=None, progress=False).to(device)
print(resnet)
tripletloss = TripletMarginLoss(margin=config.LOSS_MARGIN)
optimiser = Adam(resnet.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

# TODO implement the training cycle with basic reporting for now.
# Also save the trained model and generate some loss plots.
