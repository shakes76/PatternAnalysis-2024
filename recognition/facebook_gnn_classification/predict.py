# predict.py
import torch
from torch_geometric.data import Data
from modules import GNN
from dataset import load_data, edges_path, features_path, labels_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')