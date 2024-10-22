"""
TSNE embeddings plot for the saved model
"""
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from dataset import GNNDataLoader
from modules import GCNModel


def plot_tsne(model, data):
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
    plt.title('TSNE of GCN Embeddings')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # Load data
    data, train_idx, valid_idx, test_idx = GNNDataLoader('/Users/anthonyngo/Documents/UQ/24sem2/COMP3710/project/PatternAnalysis-2024/facebook.npz')

    # Initialize model, ensuring the input, hidden, and output dimensions match the training setup
    model = GCNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1)

    # Load the previously saved model weights
    model.load_state_dict(torch.load('GCN_model.pth'))

    # Call the TSNE visualization function
    plot_tsne(model, data)