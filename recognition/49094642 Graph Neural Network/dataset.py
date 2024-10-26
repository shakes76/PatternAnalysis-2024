import torch
import pandas as pd
import json
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data


def load_data(edges_path, labels_path, features_path):
    # Load edges, labels, and features
    edges = pd.read_csv(edges_path)
    labels = pd.read_csv(labels_path)
    with open(features_path) as f:
        features = json.load(f)

    # Prepare node features from JSON data
    node_features = torch.zeros((len(features), 128))
    for node, feats in features.items():
        feats = feats[:128]  # Ensure the feature vector has a maximum size of 128
        if len(feats) < 128:
            feats += [0] * (128 - len(feats))  # Padding if necessary
        node_features[int(node)] = torch.tensor(feats, dtype=torch.float)

    # Convert edge list to undirected format for GCN
    edge_index = to_undirected(torch.tensor(edges.values.T, dtype=torch.long))

    # Encode labels as integers
    labels['page_type'], _ = pd.factorize(labels['page_type'])
    y = torch.tensor(labels['page_type'].values, dtype=torch.long)

    # Create a Data object
    data = Data(x=node_features, edge_index=edge_index, y=y)

    return data, len(labels['page_type'].unique())

