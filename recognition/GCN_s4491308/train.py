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
train_mask, val_mask, test_mask = perform_split(data, 0.80, 0.10, 0.10)

# Model configuration
in_channels = data.num_features
hidden_channels = 16
out_channels = len(data.y.unique())
dropout = 0.3

# Initialize model, optimizer, and loss function
model = GCN(in_channels, hidden_channels, out_channels, dropout)
model = model.to(device)
data = data.to(device)
optimizer = Adam(model.parameters(), lr=0.05)
loss_criterion = torch.nn.CrossEntropyLoss()

# list for train, test and validation losses 
train_losses = []
validation_losses = []

# early stopping paramaters 
# wait for 10 epochs to see improvement 
patience = 10
best_val_loss = float('inf')
epochs_without_improvement = 0

#number of epochs to train for 
epochs = 200

# Training loop

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    train_losses.append(loss.cpu().item())

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_out = model(data.x, data.edge_index)
        val_pred = val_out[val_mask].argmax(dim=1)
        val_loss = loss_criterion(val_out[val_mask], data.y[val_mask])
        val_acc = (val_pred == data.y[val_mask]).sum().item() / val_mask.sum().item()
        validation_losses.append(val_loss.cpu().item())

    print(f'Epoch {epoch:03d}, Training Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Early stopping check and save best model 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_gcn_model.pth')
    else:
        epochs_without_improvement += 1
        
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break
    
print("Training complete. Best model saved as best_gcn_model.pth")

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

