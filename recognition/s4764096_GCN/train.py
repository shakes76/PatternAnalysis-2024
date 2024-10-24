import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap.umap_ as umap
from dataset import prepare_data
from modules import GCN

def train():

    graph, train_mask, test_mask, in_feats = prepare_data()

    model = GCN(in_feats, num_classes=4)
    optimizer = th.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    num_epochs = 100

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        logits = model(graph, graph.ndata['features'])

        loss = F.cross_entropy(logits[train_mask], graph.ndata['labels'][train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with th.no_grad():
            test_logits = model(graph, graph.ndata['features'])
            test_preds = test_logits[test_mask].argmax(dim=1)
            test_labels = graph.ndata['labels'][test_mask]
            accuracy = (test_preds == test_labels).float().mean()

            test_accuracies.append(accuracy.item())

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Test Accuracy: {accuracy.item()}')

    plot_results(train_losses, test_accuracies)

    show_graph(graph, model)

def plot_results(train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss', color='red')
    plt.plot(epochs, test_accuracies, label='Test Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Test Accuracy Over Epochs')
    plt.legend()

    plt.show()

def show_graph(graph, model):
    with th.no_grad():

        embeddings = model(graph, graph.ndata['features'])

    embeddings = embeddings.numpy()

    umap_embeddings = umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=2).fit_transform(embeddings)
    true_labels = graph.ndata['labels'].numpy()

    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=true_labels, cmap='viridis', s=10)
    plt.colorbar()

    plt.show()

if __name__ == '__main__':
    train()


