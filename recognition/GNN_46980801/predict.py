import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from modules import GCNNet
from dataset import graph_data


def makeTSNE(model, data, num):
    model.eval()
    with torch.no_grad():
        # Get the embeddings from the second to last layer
        embeddings = model(data).cpu().numpy()

    #Ensure TSNE is 2d for ease of plotting
    reducer = TSNE(n_components=2)
    
    embeddings = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=data.y.cpu(), s=10, legend='full')
    plt.title(f"TSNE Visualization of Node Embeddings with Ground Truth Labels")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Ground Truth Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('TSNE' + str(num) + '.png')
    plt.show()


