def visualize(model, data):
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
