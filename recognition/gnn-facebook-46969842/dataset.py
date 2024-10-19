"""Data Loader File"""
import numpy as np
import torch
from torch_geometric.data import Data

class FacebookDataset:
    """
    Class for loading and storing data from the pre-processed Facebook dataset.

    Attributes:
        data (Data): A Data structure storing the data for the dataset
        num_features (int): The number of features for each node.
        num_classes (int): The total number of unique classes (labels).

    Parameters:
        path (str): The file path to the pre-processed .npz dataset.
    """
    
    def __init__(self, path):
        self.data = self.load_data(path)  # Load the dataset
        self.num_features = self.data.x.size(1)  # Number of features per node
        self.num_classes = int(self.data.y.max().item()) + 1  # Total number of unique classes

    def load_data(self, path):
        """
        Load data from a .npz file.

        Parameters:
            path (str): The file path to the .npz dataset.

        Returns:
            Data: A Data structure storing the data for the dataset
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