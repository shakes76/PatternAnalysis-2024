"""
UMAP embedding for ADNI dataset StyleGAN models.
"""


from utils import *


def generate_dataset(path):
    """
    Generate dataset as 2D Array for UMAP embedding.
    """
    dataset = []
    for i in range(1000):
        img_path = path + f"/img_{i}.png"
        img = Image.open(img_path).convert('L')
        dataset.append(np.array(img).flatten())
    return np.array(dataset)


def perform_UMAP():
    """
    Plot UMAP embeddings for the model generations.
    """
    # load datasets into one big array
    ad_dataset = generate_dataset(f"{SRC}/generated_images/12 Epoch AD/images")
    nc_dataset = generate_dataset(f"{SRC}/generated_images/12 Epoch NC/images")
    all_dataset = np.concatenate((ad_dataset, nc_dataset))

    reducer = umap.UMAP(random_state=42)
    reducer.fit(all_dataset)
    embedding = reducer.transform(all_dataset)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))

    # label AD as 0 and NC as 1
    labels = np.zeros(2000)
    labels[1000:] = 1

    # plot UMAP embeddings
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(3)-0.5).set_ticks(np.arange(2))
    plt.title('UMAP projection', fontsize=24)
    plt.show()


if __name__ == "__main__":
    perform_UMAP()
