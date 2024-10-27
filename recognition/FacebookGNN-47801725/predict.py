import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from modules import GCN  
from data import load_data, split_data  
from matplotlib.colors import ListedColormap
def load_best_model(model, model_path, device):
    """
    Load the best saved model weights.

    Args:
        model (torch.nn.Module): The GCN model.
        model_path (str): Path to the saved model file.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: Model loaded with the best weights.
    """
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)

    model.eval()
    return model

def extract_embeddings(model, data):
    """
    Extract embeddings from the final hidden layer of the GCN model.

    Args:
        model (torch.nn.Module): The GCN model.
        data (torch_geometric.data.Data): The graph data object.

    Returns:
        torch.Tensor: Node embeddings from the GCN model.
    """
    model.eval()
    with torch.no_grad():
        # Pass through the layers to get embeddings before the final output layer
        x = model.conv1(data.x, data.edge_index).relu()
        x = model.conv2(x, data.edge_index).relu()
        embeddings = model.conv3(x, data.edge_index).cpu()  # Extract embeddings from the third hidden layer
    return embeddings

def plot_tsne(embeddings, labels):
    """
    Generate a 2D t-SNE plot of the embeddings with colors representing the ground truth labels.

    Args:
        embeddings (torch.Tensor): Node embeddings (features).
        labels (torch.Tensor): Ground truth labels for the nodes.
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Define a custom colormap with exactly four colors
    custom_cmap = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])  # Four distinct colors

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap=custom_cmap, s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(4), label="Class Label")
    plt.title("t-SNE Visualization of Node Embeddings with Four-Class Colors")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

if __name__ == '__main__':
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and split the dataset
    file_path = '/Users/eaglelin/Downloads/facebook.npz'  # Update with your dataset path
    data = load_data(file_path)
    data = split_data(data)
    data = data.to(device)

    # Define model parameters (ensure they match training setup)
    input_dim = data.num_features
    hidden_dims = [100, 64, 32]
    output_dim = len(torch.unique(data.y))

    # Initialize the model and load the best trained weights
    model = GCN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_rate=0.5).to(device)
    model = load_best_model(model, 'best_gcn_model.pth', device)

    # Extract embeddings from the GCN model
    embeddings = extract_embeddings(model, data)
    labels = data.y.cpu().numpy()

    # Plot t-SNE with ground truth labels
    plot_tsne(embeddings, labels)