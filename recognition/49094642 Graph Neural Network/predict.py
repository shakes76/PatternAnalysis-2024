import torch
import matplotlib.pyplot as plt
from modules import GCN
from dataset import DataLoader
from sklearn.manifold import UMAP

def visualize_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model1.conv1(data.x, data.edge_index)
        
    umap = UMAP(n_components=2)
    umap_embeds = umap.fit_transform(embeddings.cup().numpy())
    
    plt.figure(figsize)=(10, 8))
    scatter = plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=data.y.cpu().numpy(), cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title('UMAP')
    plt.show()

def main():
    edge_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"
    target_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"

    data_loader = DataLoader(edge_path, features_path, target_path)
    data = data_loader.create_data()

    visualize_embeddings(model, data)
    
if __name__ == "__main__":
    main()
