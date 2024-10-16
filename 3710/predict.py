import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from modules import GNN
from dataset import load_data

def visualize_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model(data).detach().cpu().numpy()
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embeddings)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data.y.cpu().numpy(), cmap='Spectral')
    plt.title('TSNE Visualization of Node Embeddings')
    plt.show()

def calculate_accuracy(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1)
        # 使用指定的掩码计算准确率
        accuracy = accuracy_score(data.y[mask].cpu(), predictions[mask].cpu())
    return accuracy

if __name__ == "__main__":
    # 加载数据
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 加载模型（确保模型已正确训练）
    model = GNN(in_channels=data.num_features, out_channels=4)
    model.load_state_dict(torch.load("model.pth"))
    model = model.to(device)

    # 计算并打印验证集和测试集上的准确率
    val_accuracy = calculate_accuracy(model, data, data.val_mask)
    test_accuracy = calculate_accuracy(model, data, data.test_mask)
    print(f'Accuracy on validation nodes: {val_accuracy * 100:.2f}%')
    print(f'Accuracy on test nodes: {test_accuracy * 100:.2f}%')

    # 可视化嵌入（对所有节点）
    visualize_embeddings(model, data)
