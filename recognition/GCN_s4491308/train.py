"""
Author: Ananya Dubey
Student No: 44913083
This script trains a GCN model on the Facebook dataset using PyTorch Geometric.
Reference: Based on https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
"""

import torch
from torch.optim import Adam
from dataset import load_data, perform_split
from modules import GCN
# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f'Using device: {device}')

# Load and split data
file_path = '/content/facebook.npz'
data = load_data(file_path)
train_mask, val_mask, test_mask = perform_split(data, 0.80, 0.10, 0.10)

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
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_out = model(data.x, data.edge_index)
        val_pred = val_out[val_mask].argmax(dim=1)
        val_acc = (val_pred == data.y[val_mask]).sum().item() / val_mask.sum().item()

    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'gcn_model.pth')
print("Model saved as gcn_model.pth")
