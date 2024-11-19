"""
Author: Yucheng Wang
Student ID: 47914111
This script handles the t-SNE visualization of the output from a trained Graph 
Convolutional Network (GCN) model. It includes functions to:

1. Perform t-SNE dimensionality reduction on the model's output.
2. Plot the results using Matplotlib and save the plot.
3. Load the preprocessed data, the trained GCN model, and perform a forward pass.
4. Visualize the t-SNE projection of node embeddings with labels.

The main function `visulize()` orchestrates the complete process of loading data, 
performing inference with the model, and plotting the t-SNE visualization.
"""
import dataset
import modules
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def perform_tsne(data, device):
    """
    Perform t-SNE dimensionality reduction on the input data.
    """
    data = data.to('cpu')  # t-SNE works on CPU
    return TSNE(n_components=2).fit_transform(data.detach().numpy())

def plot_scatter(z, labels, num_classes):
    """
    Create a scatter plot of the t-SNE output.
    """
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    # Plot each class separately
    for i in range(num_classes):
        indices = np.where(labels == i)
        plt.scatter(z[indices, 0], z[indices, 1], label=i)

    plt.title("TSNE Visualized")
    plt.legend()

def save_plot(directory, name):
    """
    Save the generated plot to the specified directory with a given name.
    """
    os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, f"{name}.png")
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")  
    plt.close()  

def plot_tsne(name, data, labels, num_classes, device):
    """
    Perform t-SNE on the model output and save the scatter plot.
    """
    data = data.to(device)
    labels = labels.to('cpu')  

    # Perform t-SNE on the data
    z = perform_tsne(data, device)
    plot_scatter(z, labels, num_classes)

    # Save the plot in the specified directory
    plot_dir = "C:/Users/Wangyucheng/Desktop/comp3710a3/PatternAnalysis-2024/Multi-layer_GNN_47914111/plots/"
    save_plot(plot_dir, name)
    # Show the plot
    plt.show() 

def load_and_prepare_model(num_features, num_classes, device, model_path="GCN_Model.pt"):
    """
    Load the GCN model, initialize it, and load the saved weights.
    """
    # Load and move the model to the specified device
    model = modules.GCN(num_features, 16, num_classes, 0.5).to(device)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model

def process_and_forward_pass(model, feature_matrix, adjacency_matrix, device):
    """
    Perform a forward pass through the model to obtain the output (node embeddings).
    """
    # Move input data to the specified device
    feature_matrix = feature_matrix.to(device)
    adjacency_matrix = adjacency_matrix.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations for inference
    with torch.no_grad():  
        output = model(feature_matrix, adjacency_matrix)

    return output

def visulize():
    """
    Main function to orchestrate the loading of data, model inference, and t-SNE visualization.

    This function loads the preprocessed graph data, the trained GCN model, and performs a 
    forward pass to get the node embeddings. It then applies t-SNE to these embeddings and 
    saves the visualized plot.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the preprocessed data
    adjacency_matrix, _, _, _, feature_matrix, labels = dataset.preprocess_adjacency_matrix()

    # Extract the number of features and number of classes
    num_features = feature_matrix.size(1)
    num_classes = len(torch.unique(labels))

    # Load the model and move it to the specified device
    model = load_and_prepare_model(num_features, num_classes, device)

    # Perform the forward pass to get the model output
    output = process_and_forward_pass(model, feature_matrix, adjacency_matrix, device)

    # Plot and save the t-SNE visualization
    plot_tsne("TSNE_train_plot", output, labels, num_classes, device)

visulize()
