import json
import pandas as pd
import torch    
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# column mapping for page_type of musae_facebook_target.csv
CATEGORY_MAPPING = {
    'tvshow': 0,
    'government': 1,
    'company': 2,
    'politician': 3,
}

def load_data(target_path):
    """
    Load the facebook dataset from the given target path 
    and return the processed data.
    The function reads the data from the target_path, 
    maps the 'page_type' to 'category' using the predefined CATEGORY_MAPPING, 
    and extracts the relevant data to create a label tensor.

    Parameters:
    -----------
    target_path : str
        The path to the target file

    Returns:
    --------
    labels : torch.tensor
        The label tensor for the dataset, in long format
    """
    target_df = pd.read_csv(target_path)
    # Map the page_type to category
    target_df['category'] = target_df['page_type'].map(CATEGORY_MAPPING)
    # Extract the relavent columns
    labels = torch.tensor(target_df['category'].values, dtype=torch.long)
    return labels

def load_facebook_data(features_path, edges_path, target_path, feature_dim=128):
    """
    Load and preprocess Facebook graph data, 
    including node features, edges and labels.
    The function reads the node features from a JSON file,
    edges from a CSV file, and labels from the _target.CSV.
    Pad or truncate the node features to the specified feature_dim.
    Create a Pytorch Geometric Data object with the node features, edge index.

    Parameters:
    -----------
    features_path : str
        The path to the node features file
    edges_path : str
        The path to the edges file  
    target_path : str       
        The path to the target file 
    feature_dim : int
        The dimension of the node features (default=128)

    Returns:
    --------        
    data : torch_geometric.data.Data
        The Pytorch Geometric Data object containing the node features('x'), 
        edge index('edge_index'), and labels('y')
    """
    # Load features from JSON file
    with open(features_path,'r') as f:
        node_features = json.load(f)
    node_features_list = []
    for feature in node_features.values():
        if len(feature) != feature_dim:
            # Adjust the feature vector length - pad / truncate
            feature = feature[:feature_dim] + [0] * (feature_dim - len(feature))
        node_features_list.append(feature)

    # Convert to tensor
    node_features_tensor = torch.tensor(node_features_list, dtype=torch.float)

    # Load edges from CSV file
    edges_df = pd.read_csv(edges_path)
    # Transpose to get the correct shape (2, num_edges)
    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long) 

    # Load labels from CSV file with the load_data function
    labels = load_data(target_path)

    # Create a Pytorch Geometric Data object
    data = Data(x=node_features_tensor, edge_index=edge_index, y=labels)
    return data


if __name__ == '__main__':
    """
    Loads the node features, edges, and labels from the given paths 
    and prints the data.
    This script is used to test the load_facebook_data function.
    It then prints the resulting PyTorch Geometric `Data` object, 
    which contains the node features, edge index, and labels.

    Files:
    - musae_facebook_features.json: Node features
    - musae_facebook_edges.csv: Edges between nodes
    - musae_facebook_target.csv: Node labels 
    
    """

    # File paths
    path = "recognition/Q2_Facebook_Data/facebook_large"
    features_path = f"{path}/musae_facebook_features.json"
    edges_path = f"{path}/musae_facebook_edges.csv"
    target_path = f"{path}/musae_facebook_target.csv"

    # Load the data
    data = load_facebook_data(features_path, edges_path, target_path)
    
    # Display the PyTorch Geometric Data object
    print(data)


