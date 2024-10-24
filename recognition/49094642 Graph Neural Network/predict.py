import torch
from modules import GCN
from dataset import DataLoader
import umap
import matplotlib.pyplot as plt

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# UMAP visualization function
def visualize_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model.conv1(data.x, data.edge_index)

    umap_model = umap.UMAP(n_components=2)
    umap_embeds = umap_model.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=data.y.cpu().numpy(), cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title('UMAP of GCN Node Embeddings')
    plt.show()

def main():
    # File paths
    edge_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"
    target_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"

    data_loader = DataLoader(edge_path, features_path, target_path)
    data = data_loader.create_data()

    # Load the trained model
    model = GCN(in_channels=data.num_node_features, hidden_channels=64, out_channels=len(torch.unique(data.y)))
    model = load_model(model, 'gcn_model.pth')

    # Visualize embeddings with UMAP
    visualize_embeddings(model, data)

if __name__ == "__main__":
    main()

