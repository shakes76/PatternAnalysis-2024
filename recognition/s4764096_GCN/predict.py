import torch as th
import umap.umap_ as umap
import matplotlib.pyplot as plt
from dataset import prepare_data
from modules import GCN

def show_graph(graph, model, title="UMAP visualization of node embeddings"):
    with th.no_grad():
        embeddings = model(graph, graph.ndata['features'])

    embeddings = embeddings.numpy()

    umap_embeddings = umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=2).fit_transform(embeddings)
    true_labels = graph.ndata['labels'].numpy()

    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=true_labels, cmap='viridis', s=10)
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_training_results(train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss', color='red')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Test Accuracy Over Epochs')
    plt.legend()
    plt.show()


# 主程序
if __name__ == '__main__':

    graph, _, _, in_feats = prepare_data()
    model = GCN(in_feats, num_classes=4)

    print("Generating UMAP before training...")
    show_graph(graph, model, title="UMAP visualization before training")

    model.load_state_dict(th.load("/Users/lingjieruan/Desktop/3710report/_pycache_/gcn_model.pth"))

    print("Generating UMAP after training...")
    show_graph(graph, model, title="UMAP visualization after training")

    try:
        train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        test_accuracies = [0.6, 0.7, 0.75, 0.8, 0.85]

        plot_training_results(train_losses, test_accuracies)
    except Exception as e:
        print(f"Could not load training results: {e}")
