"""
This code is used to train the Siamese network on the ISIC kaggle challenge dataset.
The extracted feature vectors are then used to train a binary classifier.

Made by Joshua Deadman
"""

import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
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

# Argument handler
parser = argparse.ArgumentParser()
parser.add_argument("-nm", "--nomodels", action="store_false", default=True, help="Stops the trained models from being saved.")
parser.add_argument("-sp", "--saveplots", action="store_true", default=False, help="Save the plots to demonstrate training. If flag not present, plots will be shown.")
args = parser.parse_args()
# TODO verify argument handling is correct
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
siamese = SiameseNetwork().to(device)
tripletloss = TripletMarginLoss(margin=config.LOSS_MARGIN)
optimiser = Adam(siamese.parameters(), lr=config.LR_SIAMESE, betas=config.BETAS)

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
    anchor_result, positive_result, negative_result = siamese(anchor, positive, negative)

    return tripletloss(anchor_result, positive_result, negative_result)

"""
Siamese Network
"""
train_loss = []
val_loss = []

print("Starting training now...")
start = time.time()
# Training cycle
for epoch in range(config.EPOCHS_SIAMESE):
    siamese.train()
    t_loss_total = []
    v_loss_total = []
    
    # Training
    for i, (anchor, positive, negative, label) in enumerate(train_loader):
        siamese.zero_grad() # Stops gradients from acumalating across batches
        loss = processes_batch(anchor, positive, negative)
        loss.backward()
        optimiser.step()

        t_loss_total.append(loss.item())
        if i % (len(train_loader)//4) == 0 and i != len(train_loader)-1:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
    
    # Validation
    siamese.eval()
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

generate_loss_plot(train_loss, val_loss, "Siamese Network", save=args.saveplots)

# The feature vectors can be stored while testing as the training is done
testing_features = []
testing_labels = []

# Evaluating the Siamese Network with test data
print("Testing the model to see loss...")
siamese.eval()
with torch.no_grad(): # Reduces memory usage
    for i, (anchor, positive, negative, label) in enumerate(test_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Evaluate images
        anchor_result, positive_result, negative_result = siamese(anchor, positive, negative)
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
        features = siamese.forward_once(anchor.to(device))
        training_features.append(features)
        training_labels.append(label.to(device))
    for anchor, _, _, label in val_loader:
        features = siamese.forward_once(anchor.to(device))
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
classifier = BinaryClassifier().to(device)
cross_entropy = CrossEntropyLoss()
optimiser = Adam(classifier.parameters(), config.LR_CLASSIFIER, config.BETAS)

train_loss = []
val_loss = []

for epoch in range(config.EPOCHS_CLASSIFIER):
    classifier.train()
    t_loss_total = []
    v_loss_total = []

    # Training
    for features, labels in zip(training_features, training_labels):
        optimiser.zero_grad()
        out = classifier(features)
        loss = cross_entropy(out, labels)
        t_loss_total.append(loss.item())
        loss.backward()
        optimiser.step()
    
    # Validation
    classifier.eval()
    with torch.no_grad():
        for features, labels in zip(validation_features, validation_labels):
            optimiser.zero_grad()
            out = classifier(features)
            loss = cross_entropy(out, labels)
            v_loss_total.append(loss.item())
    
    train_loss.append(sum(t_loss_total)/len(t_loss_total))
    val_loss.append(sum(v_loss_total)/len(v_loss_total))

# Testing the classifier
classifier.eval()
with torch.no_grad():
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    for features, labels in zip(testing_features, testing_labels):
        out = classifier(features)
        predicted = torch.argmax(out, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.data.cpu())
        all_predictions.extend(predicted.cpu())

    print('\nTest Accuracy for Classification: {} %'.format(100 * correct / total))

    # Generate a confusion matrix ti visualise accuracy
    plt.close() # As ConfusionMatrixDisplay uses it's own figure
    cm_display = ConfusionMatrixDisplay.from_predictions(all_labels, all_predictions, display_labels=["Benign", "Malignant"])
    cm_display.plot(colorbar=False)
    if args.saveplots:
        plt.savefig(config.IMAGEPATH + "/confusion_matrix.png", dpi=80)
    else:
        plt.show()

generate_loss_plot(train_loss, val_loss, "Binary Classifier", save=args.saveplots)

if not args.nomodels:
    torch.save(siamese.state_dict(), config.MODELPATH + "/siamese_"+ datetime.now().strftime('%d-%m-%Y_%H:%M' + ".pth"))
    torch.save(classifier.state_dict(), config.MODELPATH + "/classifier_"+ datetime.now().strftime('%d-%m-%Y_%H:%M' + ".pth"))
