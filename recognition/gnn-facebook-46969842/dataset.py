"""
dataset.py

Data Loader File. Consists of a single class FacebookDataset which 
loads the dataset from the given .npz file

Author: Tristan Hayes - 46969842
"""
import numpy as np
import torch
from torch_geometric.data import Data

class FacebookDataset:
    """
    Class for loading and storing data from the pre-processed Facebook dataset.

    Attributes:
        data (Data): A Data structure storing the data for the dataset.
        num_features (int): The number of features for each node.
        num_classes (int): The total number of unique classes (labels).

    Parameters:
        path (str): The file path to the pre-processed .npz dataset.
        train (float): Proportion of data to use for training (default: 0.8).
        validate (float): Proportion of data to use for validation (default: 0.1).
    """
    
    def __init__(self, path, train=0.8, validate=0.1):
        """
        Initialize the dataset by loading data and creating masks.

        Parameters:
            path (str): The file path to the pre-processed .npz dataset.
            train (float): Proportion of data to use for training.
            validate (float): Proportion of data to use for validation.
        """
        self.data = self.load_data(path)  # Load the dataset
        
        # Calculate the number of nodes
        self.num_nodes = self.data.y.size(0)

        # Create masks for training, validation, and testing
        self.data.train_mask = torch.randperm(self.data.num_nodes)[:int(train * self.num_nodes)]
        self.data.val_mask = torch.randperm(self.data.num_nodes)[int(train * self.num_nodes):int((train + validate) * self.num_nodes)]
        self.data.test_mask = torch.randperm(self.data.num_nodes)[int((train + validate) * self.num_nodes):]

        # Store the number of features and classes
        self.num_features = self.data.x.size(1)  # Number of features per node
        self.num_classes = int(self.data.y.max().item()) + 1  # Total number of unique classes

    def load_data(self, path):
        """
        Load data from a .npz file.

        Parameters:
            path (str): The file path to the .npz dataset.

        Returns:
            Data: A Data structure storing the data for the dataset.
        """
        # Load data from the .npz file
        data_dict = np.load(path)

        # Convert loaded data into tensors
        features = torch.tensor(data_dict['features'], dtype=torch.float)  # Node features
        labels = torch.tensor(data_dict['target'], dtype=torch.long)        # Node labels
        edges = torch.tensor(data_dict['edges'], dtype=torch.long).t().contiguous()  # Edge connections

        # Create a Data object to store the data
        data = Data(x=features, edge=edges, y=labels)

        return data

    def get_data(self):
        """
        Get the loaded data.

        Returns:
            Data: The PyTorch Geometric Data object containing the dataset.
        """
        return self.data

if __name__ == "__main__":
    # For testing purposes
    dataset = FacebookDataset(path='facebook.npz')  # Load the dataset
    data = dataset.get_data()  # Retrieve the data

    # Print shapes of the loaded data
    print("Features shape:", data.x.shape)
    print("Labels shape:", data.y.shape)
    print("Train mask shape:", data.train_mask.shape)
    print("Validation mask shape:", data.val_mask.shape)
    print("Test mask shape:", data.test_mask.shape)
