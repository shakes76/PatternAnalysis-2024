# -----------------------------------------------------------
# Project: Graph Attention Network for Node Classification
# Filename: predict.py
# Author: Tsung-Tse Tu
# Student ID: s4780187
# Date: October 2024 (Last edited 10/17/2024)
# Description: This script loads the pre-trained GAT model 
#              and evaluates it on the test set. It outputs 
#              the test accuracy and visualizes the learned 
#              node embeddings using t-SNE.
# -----------------------------------------------------------


import torch
from modules import GAT
from dataset import load_facebook_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict():
    try:
        # Set the seed
        set_seed(42)

        print("Loading data for prediction...")
        data = load_facebook_data()

        if data is None:
            print("Data loading error. Exiting...")
            return

        edges = data['edges']
        features = data['features']
        target = data['target']

        _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        print("Loading the trained model...")
        input_dim = X_test.shape[1]
        output_dim = len(torch.unique(y_test))
        model = GAT(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_layers=4, heads=4, dropout=0.2)

        model.load_state_dict(torch.load('gnn_model.pth', weights_only=True))
        model.eval()

        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.unique(edges.flatten())) if old_idx < X_test.size(0)}

        new_edges = []
        for edge in edges:
            if edge[0] in node_map and edge[1] in node_map:
                new_edges.append([node_map[edge[0]], node_map[edge[1]]])

        edge_reindex = torch.tensor(new_edges, dtype=torch.long).t()

        print("Making predictions on test data...")
        with torch.no_grad():
            out = model(X_test.clone().detach(), edge_reindex.clone().detach())
            preds = torch.argmax(out, dim=1)

        accuracy = accuracy_score(y_test.cpu(), preds.cpu())
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        print("Generating t-SNE visualization...")
        tsne = TSNE(n_components=2)
        embeddings = tsne.fit_transform(out.cpu().numpy())

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y_test.cpu().numpy(), cmap='viridis', s=10)
        plt.colorbar(scatter)
        plt.title('t-SNE visualization of GAT embeddings')
        plt.show()

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    predict()
