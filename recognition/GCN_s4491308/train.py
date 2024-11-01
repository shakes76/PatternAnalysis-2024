"""
Author: Ananya Dubey
Student No: 44913083
This script trains a GCN model on the Facebook dataset using PyTorch Geometric.
Reference: Based on https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py#L81
"""

import torch
from torch.optim import Adam
from dataset import load_data, perform_split
from modules import GCN
import numpy as np 
import random 
import matplotlib.pyplot as plt 


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f'Using device: {device}')

# Load and split data
file_path = '/content/drive/MyDrive/comp3710_project/facebook.npz'
data = load_data(file_path)
train_mask, val_mask, test_mask = perform_split(data, 0.70, 0.15, 0.15)

# Model configuration
in_channels = data.num_features
hidden_channels = 16
out_channels = len(data.y.unique())
dropout = 0.5

# Initialize model, optimizer, and loss function
model = GCN(in_channels, hidden_channels, out_channels, dropout)
model = model.to(device)
data = data.to(device)
optimizer = Adam(model.parameters(), lr=0.01)
loss_criterion = torch.nn.CrossEntropyLoss()

# list for train, test and validation losses 
train_losses = []
validation_losses = []


# Training loop
epochs = 200
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_out = model(data.x, data.edge_index)
        val_pred = val_out[val_mask].argmax(dim=1)
        val_loss = loss_criterion(val_out[val_mask], data.y[val_mask])
        val_acc = (val_pred == data.y[val_mask]).sum().item() / val_mask.sum().item()
        validation_losses.append(val_loss)

    print(f'Epoch {epoch:03d}, Training Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    

# Save the trained model
torch.save(model.state_dict(), 'gcn_model.pth')
print("Model saved as gcn_model.pth")

#plot the losses 
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, epochs + 1), validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.savefig('/content/drive/MyDrive/comp3710_project/loss_plot.png')
plt.show()

