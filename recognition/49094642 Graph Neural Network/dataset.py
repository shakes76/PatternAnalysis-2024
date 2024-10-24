import torch
import pandas as pd
import json
from torch_geometric.utils import to_undirected

# Data Loader Class
class FacebookDatasetLoader:
    def __init__(self, edge_path, target_path, feature_path):
        self.edge_path = edge_path
        self.target_path = target_path
        self.feature_path = feature_path

    def load_data(self):
        edges = pd.read_csv(self.edge_path)
        labels = pd.read_csv(self.target_path)
        with open(self.feature_path) as f:
            features = json.load(f)

        # Process node features, truncate or pad to 128 dimensions
        node_features = torch.zeros((len(features), 128))
        for node, feats in features.items():
            feats = feats[:128]
            if len(feats) < 128:
                feats += [0] * (128 - len(feats))
            node_features[int(node)] = torch.tensor(feats, dtype=torch.float)

        # Convert edge list to tensor and labels
        edge_index = to_undirected(torch.tensor(edges.values.T, dtype=torch.long))
        labels['page_type'], _ = pd.factorize(labels['page_type'])
        y = torch.tensor(labels['page_type'].values, dtype=torch.long)

        return node_features, edge_index, y, labels
