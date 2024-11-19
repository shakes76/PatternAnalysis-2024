"""
utils.py

Utilities file for plotting. Contains functions
    - plot_loss() for plotting training/validation loss
    - tsne_embeddings() for plotting embeddings using t-SNE
    - pca_embeddings() for plotting embeddings using PCA
    - umap_embeddings() for plotting embeddings using UMAP

Author: Tristan Hayes - 46969842
"""
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

def loss_plot(training_loss, val_losses):
    """
    Plot training and validation loss for the model over the epochs.

    Parameters:
        training_loss (List): A list of training losses at each epoch.
        val_losses (List): A list of validation losses at each epoch.
    """
    # Store loss per epoch graph
    plt.plot(training_loss, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    os.makedirs('outputs/', exist_ok=True)
    plt.savefig("outputs/epoch_losses.png")
    plt.close()

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
