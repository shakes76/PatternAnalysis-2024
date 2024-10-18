# train.py

import torch
from torch.optim import Adam
from torch_geometric.data import Data
from modules import GNN
from dataset import load_data, edges_path, features_path, labels_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
features, edge_index, labels, page_type_mapping = load_data(edges_path, features_path, labels_path)
data = Data(x=features, edge_index=edge_index, y=labels).to(device)