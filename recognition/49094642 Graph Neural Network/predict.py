import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt
from modules import GCN
from dataset import load_data

def plot_umap(data, model):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        x = F.relu(model.bn1(model.conv1(x, edge_index)))
        x = F.dropout(x, p=0.5, training=model.training)
        x = F.relu(model.bn2(model.conv2(x, edge_index)))
        x = F.dropout(x, p=0.5, training=model.training)
        x = F.relu(model.bn3(model.conv3(x, edge_index)))
        x = F.dropout(x, p=0.5, training=model.training)
        embeddings = model.conv4(x, edge_index).cpu().numpy()

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=data.y.cpu().numpy(), cmap="Spectral", s=5)
    plt.colorbar()
    plt.title("UMAP")
    plt.show()

def main():
    data, _, num_classes = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=128, hidden_dim=256, num_classes=num_classes).to(device)
    data = data.to(device)

    # Load the best trained model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Perform UMAP and plot
    plot_umap(data, model)

if __name__ == '__main__':
    main()
