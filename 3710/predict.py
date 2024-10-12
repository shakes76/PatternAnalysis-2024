import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from modules import GNN
from dataset import load_data
from sklearn.metrics import accuracy_score

def visualize_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model(data).detach().numpy()
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embeddings)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data.y, cmap='Spectral')
    plt.title('TSNE Visualization of Node Embeddings')
    plt.show()

def calculate_accuracy(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1)
        accuracy = accuracy_score(data.y.cpu(), predictions.cpu())
    return accuracy

if __name__ == "__main__":
    # Load data
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # Load model (ensure it is properly trained)
    model = GNN(in_channels=data.num_features, out_channels=6)
    # Assuming you have a saved model state_dict
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    # Calculate and print accuracy
    accuracy = calculate_accuracy(model, data)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Visualize embeddings
    visualize_embeddings(model, data)