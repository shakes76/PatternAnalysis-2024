import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from sklearn.preprocessing import StandardScaler

def load_data(npz_path):
    """
    Loads and preprocesses the dataset from a .npz file.

    Args:
        npz_path (str): Path to the .npz file containing the dataset.

    Returns:
        data (torch_geometric.data.Data): The graph data object with node features, edge indices, and labels.
    """
    # Load data from the .npz file
    data = np.load(npz_path)

    # Extract edges, features, and labels
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['target'], dtype=torch.long)

    # Normalize the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features.numpy())
    features = torch.tensor(normalized_features, dtype=torch.float32)

    # Create the graph data object
    data = Data(x=features, edge_index=edge_index, y=labels)

    return data

def split_data(data):
    """
    Splits the data into training, validation, and test sets using RandomNodeSplit.

    Args:
        data (torch_geometric.data.Data): The graph data object.

    Returns:
        data (torch_geometric.data.Data): The graph data object with train, validation, and test masks.
    """
    # Split the data using RandomNodeSplit
    split = RandomNodeSplit(num_val=0.1, num_test=0.2)  # Adjust the proportions as needed
    data = split(data)

    return data

#Example
if __name__ == "__main__":
    npz_path = '/Users/eaglelin/Downloads/facebook.npz'  # Update the path to your .npz file
    data = load_data(npz_path)
    data = split_data(data)
    print("Data loaded and split successfully.")
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
