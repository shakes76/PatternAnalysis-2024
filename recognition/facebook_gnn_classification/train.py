# train.py

import torch
from torch.optim import Adam
from torch_geometric.data import Data
from modules import GNN
from dataset import load_data, edges_path, features_path, labels_path