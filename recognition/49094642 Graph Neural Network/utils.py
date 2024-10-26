import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap.umap_ as umap


# Function to plot UMAP embeddings from the graph network
def plot_umap(data, model):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        x, edge_index = data.x, data.edge_index
        x = F.relu(model.bn1(model.conv1(x, edge_index)))
        x = F.dropout(x, p=0.5, training=model.training)
        x = F.relu(model.bn2(model.conv2(x, edge_index)))
        x = F.dropout(x, p=0.5, training=model.training)
        x = F.relu(model.bn3(model.conv3(x, edge_index)))
        x = F.dropout(x, p=0.5, training=model.training)
        embeddings = model.conv4(x, edge_index).cpu().numpy()

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot UMAP results with ground truth labels
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=data.y.cpu().numpy(), cmap="Spectral", s=5)
    plt.colorbar()
    plt.title("UMAP Projection ")
    plt.show()


# Function to plot loss and accuracy curves
def plot_metrics(train_loss_values, test_loss_values, train_acc_values, test_acc_values):
    plt.figure(figsize=(12, 6))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_values, label='Train Loss', color='red')
    plt.plot(test_loss_values, label='Test Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_values, label='Train Accuracy', color='red')
    plt.plot(test_acc_values, label='Test Accuracy', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

