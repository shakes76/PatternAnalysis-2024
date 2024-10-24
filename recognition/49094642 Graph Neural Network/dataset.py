import torch
import pandas as pd
import json
import torch_geometric.transforms as T
from torch_geometric.data import Data

class DataLoader:
    def __init__(self, edge_path, features_path, target_path):
        self.edge_path = edge_path
        self.features_path = features_path
        self.target_path = target_path

    def load_edges(self):
        edges_df = pd.read_csv(self.edge_path)
        return torch.tensor(edges_df.values.T, dtype=torch.long)

    def load_features(self):
        with open(self.features_path) as f:
            features_dict = json.load(f)
        features_list = []
        for node_id, feature in features_dict.items():
            features_list.append([int(node_id)] + feature)
        features_df = pd.DataFrame(features_list)
        return torch.tensor(features_df.iloc[:, 1:].values, dtype=torch.float)

    def load_target(self):
        target_df = pd.read_csv(self.target_path)
        return torch.tensor(target_df['target'].values, dtype=torch.long)

    def create_data(self):
        edges = self.load_edges()
        x = self.load_features()
        y = self.load_target()

        # Split the dataset into train/test and add train/test masks
        num_nodes = y.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Randomly select 80% of nodes for training and 20% for testing
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)
        split_idx = int(0.8 * num_nodes)
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]
        train_mask[train_indices] = True
        test_mask[test_indices] = True

        data = Data(x=x, edge_index=edges, y=y, train_mask=train_mask, test_mask=test_mask)
        return T.NormalizeFeatures()(data)
        x = self.load_features()
        y = self.load_target()
        dataset = Data(x=x, dege_index=edges, y=y)
        return T.NormalizeFeatures()(data)
