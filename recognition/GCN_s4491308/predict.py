"""
Author: Ananya Dubey
Student No: 44913083
This script loads a trained GCN model, runs inference on the test set, and displays results.
"""

import torch
from dataset import load_data, perform_split
from modules import GCN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load and split data
file_path = '/content/facebook.npz'
data = load_data(file_path)
train_mask, val_mask, test_mask = perform_split(data, 0.80, 0.10, 0.10)

# Model configuration
in_channels = data.num_features
hidden_channels = 16
out_channels = len(data.y.unique())
dropout = 0.5

# Initialize model and load saved weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels, hidden_channels, out_channels, dropout).to(device)
model.load_state_dict(torch.load('gcn_model.pth'))
model.eval()
data = data.to(device)

# Run inference on the test set
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=1).cpu()
    true_labels = data.y[test_mask].cpu()

# Calculate test accuracy
test_acc = (pred == true_labels).sum().item() / test_mask.sum().item()
print(f'Test Accuracy: {test_acc:.4f}')


