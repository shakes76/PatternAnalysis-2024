def plot_2d_umap(features, labels=None, n_neighbors=15, min_dist=0.1, metric='euclidean'):
  reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric)
  embedding = reducer.fit_transform(features)
  plt.figure(figsize=(10, 8))
  if labels is not None:
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.colorbar(boundaries=range(len(set(labels))+1)) 
  else:
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
    plt.title('UMAP')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.show()
