"""
TSNE embeddings plotting for the saved model
@author Anthony Ngo
@date 21/10/2024
"""
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import GNNDataLoader
from modules import *


def plot_tsne(architecture, model, data):
    model.eval()
    with torch.no_grad():
        # Get the embeddings from the model
        out = model(data).detach().cpu().numpy()

    # TSNE dimensionality reduction
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(out)

    # Plotting embeddings plot
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=data.y.cpu(), cmap="coolwarm", s=5)
    plt.title('TSNE of ' + architecture + ' Embeddings')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # Change seed if required
    seed = 42
    # Load data
    data, train_idx, valid_idx, test_idx = GNNDataLoader('recognition/48019022_GNN/facebook.npz', seed=seed)

    # Select model
    architecture = "SGC"

    # Initialize model, ensuring the input, hidden, and output dimensions match the training setup
    if architecture == "GCN":
        model = GCNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1)
    elif architecture == "GAT":
        model = GATModelBasic(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1)
    elif architecture == "SAGE":
        model = GraphSAGE(input_dim=128, hidden_dim=64, output_dim=data.y.max().item()+1)
    elif architecture == "SGC":
        model = SGCModel(input_dim=128, output_dim=data.y.max().item()+1, k=2)
    
    # Load the best performing model of the chosen architecture
    savedpath = "best_" + architecture + "_model.pth"
    # Load the previously saved model weights
    model.load_state_dict(torch.load(savedpath))

    # Call the TSNE visualization function
    plot_tsne(architecture, model, data)