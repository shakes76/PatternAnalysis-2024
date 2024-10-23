def visualize(model, deviece, data):
    model.eval()
    with torch.no_grad():
        embeddings = model1.conv1(data.x, data.edge_index)
    plt.figure(figsize)=(10, 8))
    scatter = plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=data.y.cpu().numpy(), cmap='Spectral', s=5)
    plt.colorbar(scatter)
    plt.title('UMAP')
    plt.show()
