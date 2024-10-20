# dataset.py

import os
import sys
import torch
import numpy as np

# Google Colab setup and data loading
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define paths for this project
base_path = "/content/drive/My Drive/COMP3710/Project"
npz_path = os.path.join(base_path, "facebook.npz")

# Add the project directory to the system path
sys.path.append(base_path)

# Data loading
def load_data(npz_path):
    # Load data from the .npz file
    data = np.load(npz_path)

    # Extract edges, features, and labels
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    features = torch.tensor(data['features'], dtype=torch.float)
    labels = torch.tensor(data['target'], dtype=torch.long)

    # Create page type mapping based on unique labels
    unique_labels = torch.unique(labels)
    page_type_mapping = {int(label): idx for idx, label in enumerate(unique_labels)}

    # Convert labels to match the numeric mapping
    labels = torch.tensor([page_type_mapping[label.item()] for label in labels], dtype=torch.long)

    return features, edge_index, labels, page_type_mapping

# Test loading function
if __name__ == "__main__":
    features, edge_index, labels, page_type_mapping = load_data(npz_path)
    print("Dataset loaded successfully.")
