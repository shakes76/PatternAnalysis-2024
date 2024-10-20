# predict.py
import torch
from torch_geometric.data import Data
from modules import GNN
from dataset import load_data, edges_path, features_path, labels_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
features, edge_index, labels, page_type_mapping = load_data(edges_path, features_path, labels_path)
data = Data(x=features, edge_index=edge_index, y=labels).to(device)

# Define the GNN model with matching dimensions
input_dim = features.shape[1]
hidden_dim = 512
output_dim = len(page_type_mapping)
model = GNN(input_dim, hidden_dim, output_dim).to(device)

# Load the trained model weights
model_path = "/content/drive/My Drive/COMP3710/Project/gnn_model_weights.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded successfully from {model_path}")

# Make predictions
with torch.no_grad():
    out = model(data)
    predictions = out.argmax(dim=1)

# Evaluate accuracy
correct = (predictions == data.y).sum().item()
accuracy = correct / data.num_nodes
print(f"Prediction Accuracy: {accuracy:.4f}")

# Example predictions
print("Sample Predictions:")
for i in range(5):
    node_label = list(page_type_mapping.keys())[list(page_type_mapping.values()).index(predictions[i].item())]
    true_label = list(page_type_mapping.keys())[list(page_type_mapping.values()).index(labels[i].item())]
    print(f"Node {i}: Predicted = {node_label}, True = {true_label}")