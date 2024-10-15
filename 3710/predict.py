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

def calculate_accuracy(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1)
        # 使用测试掩码来计算准确率
        accuracy = accuracy_score(data.y[data.test_mask].cpu(), predictions[data.test_mask].cpu())
    return accuracy

if __name__ == "__main__":
    # Load data
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # Create a test mask (the nodes not in train_mask will be used for testing)
    num_nodes = data.num_nodes
    test_mask = ~data.train_mask  # 训练节点之外的节点为测试节点
    data.test_mask = test_mask

    # Load model (ensure it is properly trained)
    model = GNN(in_channels=data.num_features, out_channels=4)
    # Assuming you have a saved model state_dict
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    print(f"Number of testing nodes: {data.test_mask.sum().item()}")
    # Calculate and print accuracy on test nodes only
    accuracy = calculate_accuracy(model, data)
    print(f'Accuracy on test nodes: {accuracy * 100:.2f}%')

    # Visualize embeddings (useful for all nodes)
    visualize_embeddings(model, data)
