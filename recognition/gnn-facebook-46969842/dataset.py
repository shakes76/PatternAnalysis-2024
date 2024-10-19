"""Data Loader File"""
import numpy as np
import torch
from torch_geometric.data import Data

"""
Class for storing all the data from the pre-processed dataset.
"""
class FacebookDataset:
    def __init__(self, path):
        self.data = self.load_data(path)

    def load_data(self, path):
        # Load data from the .npz file
        data_dict = np.load(path)

        features = torch.tensor(data_dict['features'], dtype=torch.float)
        labels = torch.tensor(data_dict['target'], dtype=torch.long)
        edges = torch.tensor(data_dict['edges'], dtype=torch.long).t().contiguous()

        # Create a Data object to store the data
        data = Data(x=features, edges=edges, y=labels)

        return data

    def get_data(self):
        return self.data
    
if __name__ == "__main__":
    # For testing purposes
    dataset = FacebookDataset(path='facebook.npz')
    data = dataset.get_data()

    print("Features shape:", data.x.shape)
    print("Labels shape:", data.y.shape)
    print("Edges shape:", data.edges.shape)
