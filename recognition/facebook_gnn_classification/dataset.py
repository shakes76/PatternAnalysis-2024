# dataset.py

from google.colab import drive
import os
import sys
import pandas as pd
import torch
import json
import numpy as np

# Google Colab setup and data loading

# Mount Google Drive
drive.mount('/content/drive')

# Define paths for this project
base_path = "/content/drive/My Drive/COMP3710/Project"
data_path = os.path.join(base_path, "facebook_large")

# Add the project directory to the system path
sys.path.append(base_path)

# Paths to dataset files
edges_path = os.path.join(data_path, "musae_facebook_edges.csv")
features_path = os.path.join(data_path, "musae_facebook_features.json")
labels_path = os.path.join(data_path, "musae_facebook_target.csv")

# Data loading

def load_data(edges_path, features_path, labels_path):
    # Load edges
    edges_df = pd.read_csv(edges_path)
    edge_index = torch.tensor(edges_df.values, dtype=torch.long).t().contiguous()

    # Load features from JSON
    with open(features_path) as f:
        features_dict = json.load(f)
    
    # Ensure consistent order by sorting nodes
    nodes = sorted(features_dict.keys(), key=lambda x: int(x))
    features = []

    # Pad features to make sure all vectors have the same length
    expected_length = 128
    for node in nodes:
        feature_vector = features_dict[node]
        if len(feature_vector) < expected_length:
            feature_vector += [0] * (expected_length - len(feature_vector))
        elif len(feature_vector) > expected_length:
            feature_vector = feature_vector[:expected_length]
        features.append(feature_vector)

    features = torch.tensor(features, dtype=torch.float)

    # Load labels and convert categorical labels to numeric
    labels_df = pd.read_csv(labels_path)
    page_types = labels_df['page_type'].unique()
    page_type_mapping = {page_type: idx for idx, page_type in enumerate(page_types)}
    labels = labels_df['page_type'].map(page_type_mapping).values
    labels = torch.tensor(labels, dtype=torch.long)

    return features, edge_index, labels, page_type_mapping

# Load dataset

features, edge_index, labels, page_type_mapping = load_data(edges_path, features_path, labels_path)