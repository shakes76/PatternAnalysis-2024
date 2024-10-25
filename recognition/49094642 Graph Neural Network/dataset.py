import torch
import pandas as pd
import json
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from sklearn.utils.class_weight import compute_class_weight


# Function to load and prepare the data
def prepare_data(edge_path, target_path, features_path):
    # Load edges, labels, and features from file
    edges = pd.read_csv(edge_path)
    labels = pd.read_csv(target_path)
    with open(features_path) as f:
        features = json.load(f)

    # Prepare node features, padding to 128 dimensions if necessary
    node_features = torch.zeros((len(features), 128))
    for node, feats in features.items():
        feats = feats[:128]
        if len(feats) < 128:
            feats += [0] * (128 - len(feats))
        node_features[int(node)] = torch.tensor(feats, dtype=torch.float)

    # Convert edge list to undirected format
    edge_index = to_undirected(torch.tensor(edges.values.T, dtype=torch.long))

    # Convert labels to numeric values
    labels['page_type'], _ = pd.factorize(labels['page_type'])
    y = torch.tensor(labels['page_type'].values, dtype=torch.long)

    return node_features, edge_index, y, labels


# Function to compute class weights for imbalanced classes
def compute_class_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels['page_type']), y=labels['page_type'])
    # Move weights to the appropriate device (CPU or GPU)
    return torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')


# Function to create a Data object for PyTorch Geometric
def create_data(node_features, edge_index, y):
    data = Data(x=node_features, edge_index=edge_index, y=y)
    data.train_mask = torch.rand(data.num_nodes) < 0.8  # 80% training set
    data.test_mask = ~data.train_mask  # 20% test set
    return data


