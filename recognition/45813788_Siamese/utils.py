import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualise_embedding(embeddings, labels, epoch, current_dir):
    """
    Function to Visualize embeddings using t-SNE and PCA.
    """
    # Convert embeddings and labels to numpy arrays
    embeddings_np = embeddings.numpy()
    labels_np = np.array(labels)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_np)

    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_np)

    # Plot t-SNE
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels_np, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Benign', 'Malignant'])
    plt.title(f't-SNE Visualization at Epoch {epoch}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Plot PCA
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=labels_np, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Benign', 'Malignant'])
    plt.title(f'PCA Visualization at Epoch {epoch}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.tight_layout()
    save_dir = os.path.join(current_dir,'images')
    file_name = f"epoch{epoch}.jpg"
    save_path = os.path.join(save_dir, file_name)

    # Save the figure
    plt.savefig(save_path)
    plt.close()


def plot_loss(train_losses, val_losses):
    '''
    Function for Plotting the Loss Curves
    '''
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()
  
 
def plot_accuracy(train_accuracies, val_accuracies):
    '''
    Function that plots accuracy of the model.
    '''
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_auc(train_aurocs, val_aurocs):
    '''
    Function to plot Receiver Operator Partial Under the Area Curve
    '''
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_aurocs) + 1)
    plt.plot(epochs, train_aurocs, label='Training AUROC')
    plt.plot(epochs, val_aurocs, label='Validation AUROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.title('Training and Validation AUROC over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()