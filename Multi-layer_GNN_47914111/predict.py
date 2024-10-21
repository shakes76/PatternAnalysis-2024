import dataset
import modules
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

def perform_tsne(data, device):
    data = data.to('cpu')  # t-SNE works on CPU
    return TSNE(n_components=2).fit_transform(data.detach().numpy())

def plot_scatter(z, labels, num_classes):
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    for i in range(num_classes):
        indices = np.where(labels == i)
        plt.scatter(z[indices, 0], z[indices, 1], label=i)

    plt.title("TSNE Visualized")
    plt.legend()

def save_plot(directory, name):
    os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, f"{name}.png")
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")  # Added a print statement to confirm the plot was saved
    plt.close()  # Ensure the plot is closed after saving to avoid overlap with future plots

def plot_tsne(name, data, labels, num_classes, device):
    data = data.to(device)
    labels = labels.to('cpu')  # Ensure labels are on CPU for plotting

    # Perform t-SNE on the data
    z = perform_tsne(data, device)

    # Create the scatter plot
    plot_scatter(z, labels, num_classes)

    # Save the plot in the specified directory
    plot_dir = "C:/Users/Wangyucheng/Desktop/comp3710a3/PatternAnalysis-2024/Multi-layer_GNN_47914111/plots/"
    save_plot(plot_dir, name)

    plt.show()  # Ensure the plot is displayed if running interactively

def load_and_prepare_model(num_features, num_classes, device, model_path="GCN_Model.pt"):
    # Load and move the model to the specified device
    model = modules.GCN(num_features, 16, num_classes, 0.5).to(device)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model

def process_and_forward_pass(model, feature_matrix, adjacency_matrix, device):
    # Move input data to the specified device
    feature_matrix = feature_matrix.to(device)
    adjacency_matrix = adjacency_matrix.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform the forward pass
    with torch.no_grad():  # Disable gradient calculations for inference
        output = model(feature_matrix, adjacency_matrix)

    return output

def visulize():
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
