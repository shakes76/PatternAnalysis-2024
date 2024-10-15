from torch.utils.data import *
from dataset import *
import matplotlib.pyplot as plt
import torch
from utiles import *
from modules import *

# Hyper-parameters
num_epochs = 5
hidden_layer = 64
classes = ["Politicians", "Governmental Organisations", "Television Shows", "Companies"]
learning_rate = 5e-4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading up the dataset and applying custom augmentations
data, masks = load_data()
train_mask = masks[0]
test_mask = masks[2]
validation_mask = masks[2]

number_of_features = data.x.shape[1]

# Creating an instance of the UNet to be trained
model = GCN(in_channels = number_of_features, hidden_channels= hidden_layer, out_channels =  len(classes))
model = model.to(device)

# Setup the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()



