"""
Author: Ananya Dubey
Student No: 44913083
This script loads a trained GCN model, runs inference on the test set, and displays results.
"""

import torch
import random
import numpy as np 
from dataset import load_data, perform_split
from modules import GCN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#set the seed for reproducability
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load and split data
file_path = '/content/drive/MyDrive/comp3710_project/facebook.npz'
data = load_data(file_path)
train_mask, val_mask, test_mask = perform_split(data, 0.80, 0.10, 0.10)

# Model configuration
in_channels = data.num_features
hidden_channels = 16
out_channels = len(data.y.unique())
dropout = 0.3

# Initialize model and load saved weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels, hidden_channels, out_channels, dropout).to(device)
model.load_state_dict(torch.load('best_gcn_model.pth'))
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

#t-SNE visualisation function 
#Reference: https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
def plotTSNE (model_out, labels):
  z = TSNE(n_components=2).fit_transform(model_out.detach().cpu().numpy())
  plt.figure(figsize=(10, 10))
  plt.xticks([])
  plt.yticks([])
  plt.scatter(z[:, 0], z[:, 1], s=70, c=labels, cmap="Set2")
  plt.colorbar()
  plt.title("t-SNE of Node Embeddings")
  plt.xlabel("tSNE dimension 1")
  plt.ylabel("tSNE dimension 2")
  plt.savefig('/content/drive/MyDrive/comp3710_project/tSNE.png')
  plt.legend()
  plt.show()

plotTSNE(out,data.y.cpu().numpy())



