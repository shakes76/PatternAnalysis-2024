import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from modules import GNN
from dataset import load_data

def visualize_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model(data).detach().numpy()
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embeddings)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data.y, cmap='Spectral')
    plt.title('TSNE Visualization of Node Embeddings')
    plt.show()

if __name__ == "__main__":
    # Load data
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # Load model (ensure it is properly trained)
    model = GNN(in_channels=data.num_features, out_channels=6)
    # Assuming you have a saved model state_dict
    model.load_state_dict(torch.load("model.pth"))

    # Visualize embeddings
    visualize_embeddings(model, data)