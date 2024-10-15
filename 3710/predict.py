import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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

def calculate_accuracy(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1)
        # Use the specified mask to calculate accuracy
        accuracy = accuracy_score(data.y[mask].cpu(), predictions[mask].cpu())
    return accuracy

if __name__ == "__main__":
    # Load data
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # Load model (ensure it is properly trained)
    model = GNN(in_channels=data.num_features, out_channels=4)
    # Assuming you have a saved model state_dict
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    # Calculate and print accuracy on validation and test nodes
    val_accuracy = calculate_accuracy(model, data, data.val_mask)
    test_accuracy = calculate_accuracy(model, data, data.test_mask)
    print(f'Accuracy on validation nodes: {val_accuracy * 100:.2f}%')
    print(f'Accuracy on test nodes: {test_accuracy * 100:.2f}%')

    # Visualize embeddings (useful for all nodes)
    visualize_embeddings(model, data)