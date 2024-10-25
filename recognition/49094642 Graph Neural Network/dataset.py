import torch
import pandas as pd
import json
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

def load_data():
    edges = pd.read_csv(r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv")
    labels = pd.read_csv(r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv")
    
    with open(r"C:\Users\wuzhe\Desktop\musae_facebook_features.json") as f:
        features = json.load(f)
    
    node_features = torch.zeros((len(features), 128))
    for node, feats in features.items():
        feats = feats[:128]
        if len(feats) < 128:
            feats += [0] * (128 - len(feats))
        node_features[int(node)] = torch.tensor(feats, dtype=torch.float)

    edge_index = to_undirected(torch.tensor(edges.values.T, dtype=torch.long))
    labels['page_type'], _ = pd.factorize(labels['page_type'])
    y = torch.tensor(labels['page_type'].values, dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, y=y)
    data.train_mask = torch.rand(data.num_nodes) < 0.8
    data.test_mask = ~data.train_mask

    return data

