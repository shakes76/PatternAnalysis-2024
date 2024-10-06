import torch
from modules import GAT
from sklearn.model_selection import train_test_split
from dataset import load_facebook_data
from torch.nn import functional as F
import numpy as np

print("script start")

def train():
    try:
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

        # Debugging: print shapes
        print(f"X_train shape: {X_train.shape}")  # Expecting (17976, 128)
        print(f"edge_reindex shape: {edge_reindex.shape}")  # Should be (2, num_edges)

        # Initialize the model
        print("Model starting...")
        input_dim = X_train.shape[1]
        output_dim = len(torch.unique(y_train))  # Get number of unique classes
        model = GAT(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_layers=4, heads=4, dropout=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

        # Early stopping setup
        early_stop_patience = 20
        early_stop_counter = 0
        best_loss = float("inf")

        # Training loop without batching (forward pass with full X_train)
        print("Start training without batches...")
        model.train()
        for epoch in range(10000):
            optimizer.zero_grad()

            # Forward pass using the entire X_train and edge_reindex for every epoch
            out = model(X_train.clone().detach(), edge_reindex.clone().detach())
            loss = loss_fn(out, y_train)

            # Backpropagation
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

            # Scheduler step based on epoch loss
            scheduler.step(loss)

            # Early stopping logic
            if loss.item() < best_loss:
                best_loss = loss.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # If early stop counter hits patience threshold, stop training
            if early_stop_counter >= early_stop_patience:
                print(f"Early stop at epoch {epoch+1}")
                break

        # Save the trained model
        torch.save(model.state_dict(), 'gnn_model.pth', _use_new_zipfile_serialization=True)
        print("Model saved to gnn_model.pth")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    train()
