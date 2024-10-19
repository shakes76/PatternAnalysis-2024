"""Example usage of trained model File"""
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from dataset import FacebookDataset
from modules import GCN
import os

def tsne_embeddings(model, data):
    """
    Plot embeddings using t-SNE from the provided model and data.

    Parameters:
        model (torch.nn.Module): The trained GCN model.
        data (Data): The data object.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Retrieve model's output and the embeddings
    with torch.no_grad():
        _, embeddings = model(data)
    
    # Prepare labels
    labels = data.y.cpu().numpy()

    # TSNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title('t-SNE Visualization of Facebook Page Node Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    # Ensure the output directory exists
    os.makedirs('outputs/', exist_ok=True)
    plt.savefig("outputs/tsne_visualization.png")
    plt.close()

def pca_embeddings(model, data):
    """
    Plot embeddings using PCA from the provided model and data.

    Parameters:
        model (torch.nn.Module): The trained GCN model.
        data (Data): The data object.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Retrieve model's output and the embeddings
    with torch.no_grad():
        _, embeddings = model(data)
    
    # Prepare labels
    labels = data.y.cpu().numpy()

    # PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title('PCA Visualization of Facebook Page Node Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    # Ensure the output directory exists
    os.makedirs('outputs/', exist_ok=True)
    plt.savefig("outputs/pca_visualization.png")
    plt.close()

if __name__ == "__main__":
    dataset = FacebookDataset(path='facebook.npz')
    data = dataset.get_data()
    model = GCN(dataset)
    model.load_state_dict(torch.load("model.pth")) # Loads the model from the saved file
    pca_embeddings(model, data)