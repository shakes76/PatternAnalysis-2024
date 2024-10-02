import torch
from modules import GCN  
from dataset import load_facebook_data  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize(embeddings, labels):
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='Spectral')
    plt.show()

def predict():
    # Load data
    data = load_facebook_data('recognition/gnn-node-classification-TsungTseTu/data/facebook.npz')

    # Initialize the model (same as in training)
    model = GCN(input_dim=128, hidden_dim=64, output_dim=data.num_classes)
                        
    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get predictions
    with torch.no_grad():
    out = model(data)

    # Visualize embeddings using TSNE
    embeddings = out.numpy()
    labels = data.y.numpy()
    visualize(embeddings, labels)

if __name__ == '__main__':
    predict()
