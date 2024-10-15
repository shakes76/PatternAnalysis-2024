import torch
from modules import GNNModel
from dataset import load_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re

def predict():
    data, classes = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = GNNModel(in_channels=data.num_features, hidden_channels=64, out_channels=len(classes)).to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        _, pred = out.max(dim=1)

    # Compute TSNE embeddings
    embeddings = out.cpu().numpy()
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot embeddings visualization
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=data.y.cpu(), cmap='tab10', alpha=0.7)

    # Get legend elements
    handles, labels = scatter.legend_elements(prop="colors")

    # Extract integer labels from strings like '$\\mathdefault{0}$'
    int_labels = []
    for label in labels:
        # Use regular expression to extract digits
        match = re.search(r'\d+', label)
        if match:
            int_labels.append(int(match.group()))
        else:
            int_labels.append(None)  # Handle cases where no digits are found

    # Map integer indices to class names
    class_labels = [str(classes[i]) if i is not None else 'Unknown' for i in int_labels]

    # Plot legend
    plt.legend(handles=handles, labels=class_labels, title='Classes')
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.savefig('tsne_embeddings.png')
    plt.show()

    # Output sample predictions
    for i in range(10):
        print(f'Node {i}: Predicted Label: {classes[pred[i]]}, True Label: {classes[data.y[i]]}')

if __name__ == '__main__':
    predict()
