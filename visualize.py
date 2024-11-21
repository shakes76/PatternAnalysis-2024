import torch
from umap import UMAP
import matplotlib.pyplot as plt
from dataset import load_facebook_data
from modules import GNNModel

def plot_umap(model, data):
    model.eval()
    with torch.no_grad():
        # Get the embeddings from the model
        out = model(data).detach().cpu().numpy()

    #  UMAP for dimensionality reduction
    umap = UMAP(n_components=2)
    embeddings = umap.fit_transform(out)

    # UMAP graph
    plt.figure(figsize=(10, 10))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=data.y.cpu(), cmap="coolwarm", s=5)
    plt.title('UMAP of GNN Embeddings')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":

    data = load_facebook_data('facebook.npz')

    # Initialize model
    model = GNNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1)

    # Load the saved model weights
    model.load_state_dict(torch.load('gnn_model.pth'))


    plot_umap(model, data)


