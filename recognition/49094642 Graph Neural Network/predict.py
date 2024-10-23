def visualize(model, deviece, data):
  model.eval()
  plot_2d_umap(features, labels=None, n_neighbors=15, min_dist=0.1, metric='euclidean'):
  reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric)
  embedding = reducer.fit_transform(features)
  plt.figure(figsize)=(10, 8))
  scatter = plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=data.y.cpu().numpy(), cmap='Spectral', s=5)
  plt.colorbar(scatter)
  plt.title('UMAP')
  plt.show()
