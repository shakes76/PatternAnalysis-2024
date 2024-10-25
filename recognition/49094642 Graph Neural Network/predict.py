import torch
import matplotlib.pyplot as plt
import umap.umap_ as umap
from dataset import DatasetPreparation
from modules import GCN
from torch_geometric.data import Data

def plot_predictions_umap(data, model):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        x = torch.relu(model.bn1(model.conv1(x, edge_index)))
        x = torch.dropout(x, p=0.5, training=model.training)
        x = torch.relu(model.bn2(model.conv2(x, edge_index)))
        x = torch.dropout(x, p=0.5, training=model.training)
        x = torch.relu(model.bn3(model.conv3(x, edge_index)))
        x = torch.dropout(x, p=0.5, training=model.training)
        embeddings = model.conv4(x, edge_index).cpu().numpy()

        # UMAP embedding
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
        embedding_2d = reducer.fit_transform(embeddings)

    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=data.y.cpu().numpy(), cmap="Spectral", s=5)
    plt.colorbar()
    plt.title("UMAP Projection ")
    plt.show()


def predict():
    # File paths
    edges_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    labels_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"

    # Data preparation
    dataset = DatasetPreparation(edges_path, labels_path, features_path)
    edges, labels, features = dataset.load_data()
    node_features = dataset.prepare_node_features(features)
    edge_index = dataset.convert_to_undirected(edges)
    y = dataset.prepare_labels(labels)

    # Create Data object
    data = Data(x=node_features, edge_index=edge_index, y=y)
    data.test_mask = torch.rand(data.num_nodes) < 0.2

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=128, hidden_dim=256, num_classes=len(labels['page_type'].unique())).to(device)

    # Load trained model
    model.load_state_dict(torch.load('model.pth'))

    # Evaluate and print predictions
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        print("Predictions for test nodes:")
        print(pred[data.test_mask])

    # Plot UMAP visualization of predictions
    plot_predictions_umap(data, model)


if __name__ == "__main__":
    predict()
