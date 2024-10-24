import torch
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from modules import GCN
from dataset import FacebookDatasetLoader
from torch_geometric.data import Data


# Function to visualize UMAP embeddings
def plot_umap(data, model):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        x = model.bn1(model.conv1(x, edge_index))
        x = torch.relu(x)
        x = model.bn2(model.conv2(x, edge_index))
        x = torch.relu(x)
        embeddings = model.conv3(x, edge_index).cpu().numpy()

    # Reduce dimensionality using UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot UMAP
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=data.y.cpu().numpy(), cmap="Spectral", s=10)
    plt.colorbar(scatter)
    plt.title("UMAP of GCN Embeddings")
    plt.show()


# Load the dataset and model, visualize predictions with UMAP
def predict():
    # File paths
    edge_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    target_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"
    feature_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"

    # Load dataset
    loader = FacebookDatasetLoader(edge_path, target_path, feature_path)
    node_features, edge_index, y, labels = loader.load_data()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels['page_type'])

    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=128, hidden_dim=256, num_classes=len(labels['page_type'].unique())).to(device)
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(y_encoded)).to(device)

    # Model prediction
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)

    # Classification report
    print("Classification Report:")
    print(classification_report(data.y.cpu(), pred.cpu(), target_names=le.classes_))

    # Visualize UMAP embeddings
    plot_umap(data, model)


if __name__ == "__main__":
    predict()

