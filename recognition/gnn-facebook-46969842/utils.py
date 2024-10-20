"""Utilities File for Plotting"""
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import umap

class_titles = {
    0: "Politicians",
    1: "Governmental Organizations",
    2: "Television Shows", 
    3: "Companies",
}

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
    # Scatter plot
    for label in class_titles.keys():
        plt.scatter(reduced_embeddings[labels == label, 0], 
                    reduced_embeddings[labels == label, 1],
                    label=class_titles[label], alpha=0.7)
    plt.title('t-SNE Visualization of Facebook Page Node Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title="Classes", loc="upper right")
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
    # Scatter plot
    for label in class_titles.keys():
        plt.scatter(reduced_embeddings[labels == label, 0], 
                    reduced_embeddings[labels == label, 1],
                    label=class_titles[label], alpha=0.7)

    plt.title('PCA Visualization of Facebook Page Node Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title="Classes", loc="upper right")
    # Ensure the output directory exists
    os.makedirs('outputs/', exist_ok=True)
    plt.savefig("outputs/pca_visualization.png")
    plt.close()

def umap_embeddings(model, data):
    """
    Plot embeddings using UMAP from the provided model and data.

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

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    # Scatter plot
    for label in class_titles.keys():
        plt.scatter(reduced_embeddings[labels == label, 0], 
                    reduced_embeddings[labels == label, 1],
                    label=class_titles[label], alpha=0.7)
    
    plt.title('UMAP Visualization of Facebook Page Node Embeddings')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(title="Classes", loc="upper right")
    # Ensure the output directory exists
    os.makedirs('outputs/', exist_ok=True)
    plt.savefig("outputs/umap_visualization.png")
    plt.close()
