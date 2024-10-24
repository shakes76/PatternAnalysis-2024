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

    for epoch in range(100):
        train(model, data, optimizer)
        acc = test(model, data)
        print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
    
    visualize_embeddings(model,data)
