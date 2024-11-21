# -----------------------------------------------------------
# Project: Mixed Graph Neural Networks for Node Classification
# Filename: train.py
# Author: Tsung-Tse Tu
# Student ID: s4780187
# Date: October 2024 (Last update 10/21/2024)
# Description: This script trains the mixed GNN model (GCN, GAT,
#              GraphSAGE) using node features and graph structure
#              for multi-class classification. It implements 
#              early stopping and a learning rate scheduler to 
#              optimize the model performance.
# -----------------------------------------------------------
import torch
import numpy as np
import random
from modules import MixedGNN
from sklearn.model_selection import train_test_split
from dataset import load_facebook_data
from torch.nn import functional as F


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

print("script start")

def train():
    try:
        # Set the seed
        set_seed(42)

        print("start training section")

        # Load data
        print("Loading data...")
        data = load_facebook_data()

        if data is None:
            print("Data loading error. Exiting...")
            return

        print(f"Data successfully loaded. Available arrays: {data.files}")
    
        edges = data['edges']
        features = data['features']
        target = data['target']

        print(f"Edge shape: {edges.shape}, features shape: {features.shape}, target shapes: {target.shape}")

        # Split the data into training (80%) and testing (20%)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

        # Convert split data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create a mapping of original node indices to reindexed node indices for X_train
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.unique(edges.flatten())) if old_idx < X_train.size(0)}

        # Re-index the edges for the training set based on the node_map
        new_edges = []
        for edge in edges:
            if edge[0] in node_map and edge[1] in node_map:
                new_edges.append([node_map[edge[0]], node_map[edge[1]]])

        edge_reindex = torch.tensor(new_edges, dtype=torch.long).t()  # Transpose to match expected shape

        # Initialize the model
        print("Model starting...")
        input_dim = X_train.shape[1]
        output_dim = len(torch.unique(y_train))  # Get number of unique classes

        # Mixed GNN model with GCN layers, GAT layers, and GraphSAGE layers
        model = MixedGNN(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, 
                         num_gcn_layers=2, num_gat_layers=2, num_sage_layers=2, 
                         heads=4, dropout=0.3)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = torch.nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10)

        early_stop_patience = 50
        early_stop_counter = 0
        best_loss = float("inf")

        print("Start training...")
        model.train()
        for epoch in range(10000):
            optimizer.zero_grad()

            # Forward pass using the entire X_train and edge_reindex for every epoch
            out = model(X_train.clone().detach(), edge_reindex.clone().detach())
            loss = loss_fn(out, y_train)

            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, Learning Rate: {current_lr}')

            scheduler.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_patience:
                print(f"Early stop at epoch {epoch+1}")
                break

        torch.save(model.state_dict(), 'gnn_model.pth', _use_new_zipfile_serialization=True)
        print("Model saved to gnn_model.pth")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':                                                                     
    train()
