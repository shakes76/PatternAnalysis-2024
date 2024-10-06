import torch
from modules import GAT
from dataset import load_facebook_data
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import numpy as np

def predict():
    try:
        print("Loading data for prediction...")
        data = load_facebook_data()

        if data is None:
            print("Data loading error. Exiting...")
            return

        # Extract edges, features, and target from the data
        edges = data['edges']
        features = data['features']
        target = data['target']

        # Split the data into training (80%) and testing (20%)
        _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

        # Convert the test data to tensors
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Load the trained model
        print("Loading the trained model...")
        input_dim = X_test.shape[1]  # Input dimension based on test data features
        output_dim = len(torch.unique(y_test))  # Output dimension is number of unique classes
        model = GAT(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_layers=4, heads=4, dropout=0.2)

        model.load_state_dict(torch.load('gnn_model.pth', weights_only=True))
        model.eval()

        # Create a mapping of original node indices to reindexed node indices for X_test
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.unique(edges.flatten())) if old_idx < X_test.size(0)}

        # Re-index the edges for the testing set based on the node_map
        new_edges = []
        for edge in edges:
            if edge[0] in node_map and edge[1] in node_map:
                new_edges.append([node_map[edge[0]], node_map[edge[1]]])

        edge_reindex = torch.tensor(new_edges, dtype=torch.long).t()  # Transpose to match expected shape

        # Make predictions using the test data
        print("Making predictions on test data...")
        with torch.no_grad():
            out = model(X_test.clone().detach(), edge_reindex.clone().detach())
            preds = torch.argmax(out, dim=1)

        # Calculate accuracy
        accuracy = accuracy_score(y_test.cpu(), preds.cpu())
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    predict()
