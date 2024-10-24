import torch
import pandas as pd
import json
import numpy as np
from torch_geometric.data import Data

class DataLoader:
    def __init__(self, edge_path, features_path, target_path):
        self.edge_path = edge_path
        self.features_path = features_path
        self.target_path = target_path

    def load_edges(self):
        return torch.tensor(pd.read_csv(self.edge_path).values.T, dtype=torch.long)

    def load_features(self):
        with open(self.features_path) as f:
            features_dict = json.load(f)
        return torch.tensor(np.array(list(features_dict.values())), dtype=torch.float)

    def load_target(self):
        return torch.tensor(pd.read_csv(self.target_path)['target'].values, dtype=torch.long)

    def create_data(self):
        edges = self.load_edges()
        x = self.load_features()
        y = self.load_target()

        num_nodes = y.size(0)
        train_mask, test_mask = self.create_masks(num_nodes)

        data = Data(x=x, edge_index=edges, y=y, train_mask=train_mask, test_mask=test_mask)
        return data

    @staticmethod
    def create_masks(num_nodes, train_split=0.8):
        mask = np.zeros(num_nodes, dtype=bool)
        train_size = int(train_split * num_nodes)
        indices = np.random.permutation(num_nodes)
        mask[indices[:train_size]] = True
        return torch.tensor(mask), torch.tensor(~mask)
