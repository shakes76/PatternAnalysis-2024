'''
Author: Kangqi Wang
Student Number: 48300588

This script is about predicting the labels for nodes 
in Facebook Large Page-Page Network dataset and 
visualizes the learned node embeddings using t-SNE. 
'''

import torch
from modules import GNNModel
from dataset import load_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def predict():
    # Load data and classes
    data, classes = load_data()
    device = torch.device('cuda')
    data = data.to(device)

    # Load the trained model
    model = GNNModel(in_channels=data.num_features, hidden_channels=64, out_channels=len(classes)).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # Get model outputs
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        _, pred = out.max(dim=1)

    # Compute TSNE embeddings using the learned node representations
    embeddings = out.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=48300588)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot TSNE embeddings with ground truth labels
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=data.y.cpu(), cmap='tab10', alpha=0.7)
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Create legend with class names
    class_labels = [str(cls) for cls in classes]
    plt.legend(handles=scatter.legend_elements()[0], labels=class_labels, title='Classes')

    plt.savefig('tsne_embeddings.png')
    plt.show()

    # Print sample predictions
    for i in range(10):
        print(f'Node {i}: Predicted Label: {classes[pred[i]]}, True Label: {classes[data.y[i]]}')

    # Calculate and print test accuracy
    test_mask = data.test_mask
    test_correct = pred[test_mask] == data.y[test_mask]
    test_acc = int(test_correct.sum()) / int(test_mask.sum())
    print(f'Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    predict()
