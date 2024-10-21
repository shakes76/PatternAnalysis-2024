"""
The code used to train the siamese network on the ISIC kaggle challenge dataset.

Made by Joshua Deadman
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import config
from modules import SiameseNetwork, BinaryClassifier
from dataset import ISICKaggleChallengeSet
from utils import split_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("WARNING: Using the CPU to train...")

# Make disjoint sets of data
train, test, val = split_data(config.DATAPATH+"/train-metadata.csv")

transforms = v2.Compose([
    v2.RandomRotation(degrees=(0, 10)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_set = ISICKaggleChallengeSet(config.DATAPATH+"/image/", train, transforms=transforms)
test_set = ISICKaggleChallengeSet(config.DATAPATH+"/image/", test, transforms=transforms)
val_set = ISICKaggleChallengeSet(config.DATAPATH+"/image/", val, transforms=transforms)

train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)
test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)
val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)

model = SiameseNetwork().to(device)
tripletloss = TripletMarginLoss(margin=config.LOSS_MARGIN)
optimiser = Adam(model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

train_loss = []
val_loss = []
test_loss = []

def processes_batch(anchor, positive, negative) -> torch.Tensor:
    """ Takes a triplet and returns the calculated loss.

    Arguments:
        anchor (torch.Tensor): The anchor image.
        positive (torch.Tensor): The positive image.
        negative (torch.Tensor): The negative image.

    Returns: A torch.Tensor storing the calculated loss using triplet loss.
    """
    # DataLoader works on cpu, so move received images to GPU
    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)

    # Evaluate images
    anchor_result, positive_result, negative_result = model(anchor, positive, negative)

    return tripletloss(anchor_result, positive_result, negative_result)

print("Starting training now...")
start = time.time()
# Training cycle
for epoch in range(config.EPOCHS):
    model.train()
    for i, (anchor, positive, negative, label) in enumerate(train_loader):
        model.zero_grad()
        loss = processes_batch(anchor, positive, negative)
        loss.backward()
        optimiser.step()

        train_loss.append(loss.item())
        if i % (len(train_loader)//4) == 0 and i != len(train_loader)-1:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
    
    # Validation
    model.eval()
    with torch.no_grad(): # Using this to reduce memory usage
        for i, (anchor, positive, negative, label) in enumerate(val_loader):
            loss = processes_batch(anchor, positive, negative)
            val_loss.append(loss)
            if i % (len(val_loader)//2) == 0 and i != len(val_loader)-1:
                print(f"Validation: Batch: {i}, Loss: {loss.item()}")

stop = time.time()
print(f"Training complete! It took {(stop-start)/60} minutes")

# TODO generate some loss plots

# The features can be stored while testing as training is done
testing_features = []
testing_labels = []

# Evaluating the Siamese Network with test data
model.eval()
with torch.no_grad(): # Reduces memory usage
    for i, (anchor, positive, negative, label) in enumerate(test_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Evaluate images
        anchor_result, positive_result, negative_result = model(anchor, positive, negative)
        testing_features.append(anchor_result)
        testing_labels.append(label)

        loss = tripletloss(anchor_result, positive_result, negative_result)
        test_loss.append(loss)
        if i % (len(test_loader)//2) == 0 and i != len(test_loader)-1:
            print(f"Testing: Batch: {i}, Loss: {loss.item()}")

# TODO add saving of model to config.MODELPATH here

""" Taking the feature vectors and using them to train a classifier. """
# Obtain the training features
training_features = []
training_labels = []
with torch.no_grad():
    for anchor, _, _, label in train_loader:
        features = model.forward_once(anchor.to(device))
        training_features.append(features)
        training_labels.append(label)

model = BinaryClassifier().to(device)
cross = CrossEntropyLoss()
optimiser = Adam(model.parameters(), 0.001, (0.9, 0.999))
for epoch in range(config.EPOCHS):
    for feature, label in zip(training_features, training_labels):
        optimiser.zero_grad()
        out = model(feature)
        cross(out, label.to(device)).backward()
        optimiser.step()

# Testing the classifier
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in zip(testing_features, testing_labels):
        out = model(features)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.to(device)).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))

# TODO add saving of model to config.MODELPATH here
