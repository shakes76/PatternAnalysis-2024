# dataset.py

# Google Colab setup and data loading

from google.colab import drive
import os
import sys

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