import torch
import pandas as pd
import json
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

# Function to load and prepare the dataset
def load_data(edge_file, target_file, feature_file):
    # Load edges, labels, and features
    edges = pd.read_csv(edge_file)
    labels = pd.read_csv(target_file)
    with open(feature_file) as f:
        features = json.load(f)

    # Prepare node features
    node_features = torch.zeros((len(features), 128))
    for node, feats in features.items():
        feats = feats[:128]
        if len(feats) < 128:
            feats += [0] * (128 - len(feats))
        node_features[int(node)] = torch.tensor(feats, dtype=torch.float)

    # Convert edge list to undirected format
    edge_index = to_undirected(torch.tensor(edges.values.T, dtype=torch.long))

    # Factorize and convert labels
    labels['page_type'], _ = pd.factorize(labels['page_type'])
    y = torch.tensor(labels['page_type'].values, dtype=torch.long)

    # Create Data object and train/test masks
    data = Data(x=node_features, edge_index=edge_index, y=y)
    data.train_mask = torch.rand(data.num_nodes) < 0.8
    data.test_mask = ~data.train_mask

    return data, y
