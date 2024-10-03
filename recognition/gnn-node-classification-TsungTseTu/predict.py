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
    data = load_facebook_data()

    # Extract features, edges, and target
    features = torch.tensor(data['features'], dtype=torch.float32)
    edge = torch.tensor(data['edges'].T, dtype=torch.long)
    target = torch.tensor(data['target'], dtype=torch.long)
    
    # Set input nad output based on feature and target
    input = features.shape[1] #Number of features per node
    output = len(set(target.numpy())) # number of unique classes

    # Initialize the model (same as in training)
    model = GCN(input_dim=input, hidden_dim=64, output_dim=output)

    # Load pre-trained model
    model.load_state_dict(torch.load("gnn_model.pth", weights_only=True))
                        
    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get predictions
    with torch.no_grad():
        out = model(features, edge)

    # Visualize embeddings using TSNE
    embeddings = out.numpy()
    labels = target.numpy()
    visualize(embeddings, labels)

if __name__ == '__main__':
    predict()
