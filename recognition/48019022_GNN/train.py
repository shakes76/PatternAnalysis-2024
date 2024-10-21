"""
Code for training, validating, tesing and saving the model.
@author Anthony Ngo
@date 21/10/2024
"""
import torch
from dataset import GNNDataLoader
from modules import GCNModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# hyperparams
lr = 0.01 # default
decay = 0.01 # default

# loading data
data = GNNDataLoader(filepath='facebook.npz')

# data splits
# we do a 80/10/10 split between training, validation and testing sets
train_label = torch.randperm(data.num_nodes)[:int(0.8*data.num_nodes)]
valid_label = torch.randperm(data.num_nodes)[int(0.8 * data.num_nodes):int(0.9 * data.num_nodes)]
test_label = torch.randperm(data.num_nodes)[int(0.9 * data.num_nodes):]

# now we can define the model
# note here that the output dimensions is defined to be 4 classes:
# politicians, governmental organizations, television shows and companies
model = GCNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1) # +1 as labels start from 0

optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay) # using Adam optimiser
criterion = torch.nn.CrossEntropyLoss() 

# Improvements:
# LR scheduler
# early stopping

# To plot
losses = []
valid_losses = []
accuracies = []

def train_model():
    """
    The function for training the model using the pre-defined parameters and loss functions
    """
    model.train() # set model to training mode
    optimiser.zero_grad() # clear gradients from last backprop

    out = model(data) # forward pass
    loss = criterion(out[train_label], data.y[train_label])

    loss.backward() # backprop the gradients
    optimiser.step() # update params

    return loss.item() # return loss to monitor

def test_model():
    """
    Function for testing the model accuracy against the validation and test data set
    """
    model.eval() # set model to eval mode

    with torch.no_grad():
        out = model(data)
        valid_loss = criterion(out[valid_label], data.y[valid_label].item()) # we calculate validation loss

        # get highest probability predictions for testing data
        predictions = out[test_label].argmax(dim=1)

        # calc accuracy by comparing predicted and true labels
        accuracy = accuracy_score(data.y[test_label].cpu(), predictions.cpu())

    return valid_loss, accuracy


