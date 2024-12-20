"""
A script that loads a selected pre-trained GNN architecture and evaluates it
on the testing data set.
@author Anthony Ngo
@date 23/10/2024
"""

import torch
from dataset import GNNDataLoader
from modules import *
from sklearn.metrics import accuracy_score

# Set seed if required
seed = 42
# Load data
data, train_idx, valid_idx, test_idx  = GNNDataLoader('recognition/48019022_GNN/facebook.npz', seed=seed)

# Initialise model
architecture = "GAT"

if architecture == "GCN":
    # Select GCN
    model = GCNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1) # +1 as labels start from 0
elif architecture == "GAT":
    # Select GAT
    model = GATModelBasic(input_dim=128, hidden_dim=64, output_dim=data.y.max().item()+1)
elif architecture == "SAGE":
    # Select GRAPH SAGE
    model = GraphSAGE(input_dim=128, hidden_dim=64, output_dim=data.y.max().item()+1)
elif architecture == "SGC":
    # Select SGC
    model = SGCModel(input_dim=128, output_dim=data.y.max().item()+1, k=2)

savedpath = "best_" + architecture + "_model.pth"

model.load_state_dict(torch.load(savedpath))

def predict(model, data, test):
    """
    Function for testing the model accuracy against the test data set and 
    getting accuracy scores
    """
    model.eval() # set model to eval mode

    with torch.no_grad():
        out = model(data)

        # get highest probability predictions for testing data
        predictions = out[test].argmax(dim=1)

        # calc accuracy by comparing predicted and true labels
        accuracy = accuracy_score(data.y[test].cpu(), predictions.cpu())

    return predictions, accuracy

predictions, accuracy = predict(model=model, data=data, test=test_idx)

print("===== Printing predictions =====")
print(predictions)

print("=========== Accuracy ===========")
print("Model achieved accuracy of ", accuracy)