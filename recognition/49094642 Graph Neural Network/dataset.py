import pandas as pd
import torch
import json
from torch_geometric.utils import to_undirected

class DatasetPreparation:
    def __init__(self, edges_path, labels_path, features_path):
        self.edges_path = edges_path
        self.labels_path = labels_path
        self.features_path = features_path

    def load_data(self):
        edges = pd.read_csv(self.edges_path)
        labels = pd.read_csv(self.labels_path)
        with open(self.features_path) as f:
            features = json.load(f)
        return edges, labels, features

    def prepare_node_features(self, features):
        node_features = torch.zeros((len(features), 128))
        for node, feats in features.items():
            feats = feats[:128]
            if len(feats) < 128:
                feats += [0] * (128 - len(feats))
            node_features[int(node)] = torch.tensor(feats, dtype=torch.float)
        return node_features

    def convert_to_undirected(self, edges):
        return to_undirected(torch.tensor(edges.values.T, dtype=torch.long))

    def prepare_labels(self, labels):
        labels['page_type'], _ = pd.factorize(labels['page_type'])
        return torch.tensor(labels['page_type'].values, dtype=torch.long)


