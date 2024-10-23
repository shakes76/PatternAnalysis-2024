"""
The code used to train the siamese network on the ISIC kaggle challenge dataset.

Made by Joshua Deadman
"""

import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import time
import torch
from torch.nn import TripletMarginLoss, CrossEntropyLoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import config
from modules import SiameseNetwork, BinaryClassifier
from dataset import ISICKaggleChallengeSet
from utils import split_data, generate_loss_plot, tsne_plot

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

# Form datsets and load them
train_set = ISICKaggleChallengeSet(config.DATAPATH+"/train-image/image/", train, transforms=transforms)
test_set = ISICKaggleChallengeSet(config.DATAPATH+"/train-image/image/", test, transforms=transforms)
val_set = ISICKaggleChallengeSet(config.DATAPATH+"/train-image/image/", val, transforms=transforms)
train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)
test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)
val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, num_workers=config.WORKERS)

# Initialise the Siamese Network
model = SiameseNetwork().to(device)
tripletloss = TripletMarginLoss(margin=config.LOSS_MARGIN)
optimiser = Adam(model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

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

"""
Siamese Network Training
"""
train_loss = []
val_loss = []

print("Starting training now...")
start = time.time()
# Training cycle
for epoch in range(config.EPOCHS_SIAMESE):
    model.train()
    t_loss_total = []
    v_loss_total = []
    
    # Training
    for i, (anchor, positive, negative, label) in enumerate(train_loader):
        model.zero_grad() # Stops gradients from acumalating across batches
        loss = processes_batch(anchor, positive, negative)
        loss.backward()
        optimiser.step()

        t_loss_total.append(loss.item())
        if i % (len(train_loader)//4) == 0 and i != len(train_loader)-1:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
    
    # Validation
    model.eval()
    with torch.no_grad(): # Using this to reduce memory usage
        for i, (anchor, positive, negative, label) in enumerate(val_loader):
            loss = processes_batch(anchor, positive, negative)
            v_loss_total.append(loss.item())
            if i % (len(val_loader)//2) == 0 and i != len(val_loader)-1:
                print(f"Validation: Batch: {i}, Loss: {loss.item()}")

    train_loss.append(sum(t_loss_total)/len(t_loss_total))
    val_loss.append(sum(v_loss_total)/len(v_loss_total))

stop = time.time()
print(f"Training complete! It took {(stop-start)/60} minutes\n")

generate_loss_plot(train_loss, val_loss, "Siamese Network", save=True)


# The feature vectors can be stored while testing as the training is done
testing_features = []
testing_labels = []

# Evaluating the Siamese Network with test data
print("Testing the model to see loss...")
model.eval()
with torch.no_grad(): # Reduces memory usage
    for i, (anchor, positive, negative, label) in enumerate(test_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Evaluate images
        anchor_result, positive_result, negative_result = model(anchor, positive, negative)
        testing_features.append(anchor_result)
        testing_labels.append(label.to(device))

        loss = tripletloss(anchor_result, positive_result, negative_result)
        if i % (len(test_loader)//2) == 0 and i != len(test_loader)-1:
            print(f"Testing: Batch: {i}, Loss: {loss.item()}")

# Obtain the features vectors from the training and validation dataset
training_features = []
training_labels = []
validation_features = []
validation_labels = []
with torch.no_grad():
    for anchor, _, _, label in train_loader:
        features = model.forward_once(anchor.to(device))
        training_features.append(features)
        training_labels.append(label.to(device))
    for anchor, _, _, label in val_loader:
        features = model.forward_once(anchor.to(device))
        validation_features.append(features)
        validation_labels.append(label.to(device))

# Generate a t-SNE plot on the training_features
cpu_features = []
cpu_labels = []
for feature in training_features:
    cpu_features.append(feature.cpu())
for label in training_labels:
    cpu_labels.append(label.cpu())
tsne_plot(cpu_features, cpu_labels, save=True)

"""
Using a binary classifier to learn the feature vectors of the Siame Network.
"""
model = BinaryClassifier().to(device)
cross_entropy = CrossEntropyLoss()
optimiser = Adam(model.parameters(), 0.001, (0.9, 0.999))

train_loss = []
val_loss = []

for epoch in range(config.EPOCHS_CLASSIFIER):
    model.train()
    t_loss_total = []
    v_loss_total = []

    # Training
    for features, labels in zip(training_features, training_labels):
        optimiser.zero_grad()
        out = model(features)
        loss = cross_entropy(out, labels)
        t_loss_total.append(loss.item())
        loss.backward()
        optimiser.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for features, labels in zip(validation_features, validation_labels):
            optimiser.zero_grad()
            out = model(features)
            loss = cross_entropy(out, labels)
            v_loss_total.append(loss.item())
    
    train_loss.append(sum(t_loss_total)/len(t_loss_total))
    val_loss.append(sum(v_loss_total)/len(v_loss_total))

# Testing the classifier
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in zip(testing_features, testing_labels):
        out = model(features)
        predicted = torch.argmax(out, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))

generate_loss_plot(train_loss, val_loss, "Binary Classifier", save=True)

torch.save(model.state_dict(), config.MODELPATH + "/siamese_"+ datetime.now().strftime('%d-%m-%Y_%H:%M' + ".pth"))
torch.save(model.state_dict(), config.MODELPATH + "/classifier_"+ datetime.now().strftime('%d-%m-%Y_%H:%M' + ".pth"))
