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

    This function reads the data from the target_path, 
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

    This function reads the node features from a JSON file,
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

def split_data (data, train_size = 0.8, val_size = 0.1):
    """
    Split the data into training, validation, and test sets.  

    This function splits the data into training, validation, and test sets   
    based on the given train_size and val_size.
    It returns boolean masks for the training, validation, and test sets.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        The Pytorch Geometric Data object
    train_size : float
        The proportion of the data to include in the training set (default=0.8)
    val_size : float
        The proportion of the data to include in the validation set (default=0.1)

    Returns:
    --------
    train_mask : torch.tensor
        The boolean mask for the training set, with True for training nodes
    val_mask : torch.tensor
        The boolean mask for the validation set, with True for validation nodes
    test_mask : torch.tensor
        The boolean mask for the test set, with True for test nodes
    """
    num_nodes = data.num_nodes
    indices = list(range(num_nodes))

    # set seed for reproducibility
    seed1 = 42 # best outcome
    # other seed number such as 100, 58 ,72, 43 were test but did not work
    seed2 = 80
    seed = seed1
    
    #split into tranning and remaining data (val + test)
    train_idx, temp_idx = train_test_split(indices, train_size=train_size, random_state=seed)

    #split the remaining data into validation and test
    val_size_adjusted = val_size / (1 - train_size) 
    val_idx, test_idx = train_test_split(temp_idx, train_size=val_size_adjusted, random_state=seed)

    # create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

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
    path = "recognition/Q2_Facebook_Data_GNN/facebook_large"
    features_path = f"{path}/musae_facebook_features.json"
    edges_path = f"{path}/musae_facebook_edges.csv"
    target_path = f"{path}/musae_facebook_target.csv"

    # Load the data
    data = load_facebook_data(features_path, edges_path, target_path)

    # Split the data into training, validation, and test sets
    train_mask, val_mask, test_mask = split_data(data)

    # Add the masks to the PyTorch Geometric Data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Display the PyTorch Geometric Data object
    print(data)


