from torch.utils.data import *
from dataset import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utiles import *
from modules import *
from sklearn.manifold import TSNE

def plot_TSNE(output, y_true, classes):
    # Plotting t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(output)

    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):  
        idx = y_true == i 
        plt.scatter(reduced_embeddings[idx, 0], 
                    reduced_embeddings[idx, 1], label=classes[i], alpha = 0.7)  
    plt.legend()
    plt.title("TSNE Plot")
    plt.savefig("TSNE_plot.png")

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 6))

    plt.plot(train_loss, label='Train Loss', color='red')
    plt.plot(val_loss, label='Validation Loss', color='blue')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss graph')
    plt.savefig("Training and Validation Loss graph.png")

def plot_accuracy(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 6))

    plt.plot(train_accuracies, label='Train Accuracy', color='red')
    plt.plot(val_accuracies, label='Validation Accuracy', color='blue')
        
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig("Training and Validation Accuracy.png")