import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from modules import GNN
from dataset import load_data

def visualize_embeddings(model, data):
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Obtain the node embeddings from the model
        embeddings = model(data).detach().cpu().numpy()
    
    # Apply t-SNE to reduce the embeddings to 2 dimensions for visualization
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(embeddings)
    
    # Plot the t-SNE result
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data.y.cpu().numpy(), cmap='Spectral')
    plt.title('TSNE Visualization of Node Embeddings')
    plt.savefig('../plot/TSNE_Visualization.png')
    plt.show()

def calculate_accuracy(model, data, mask):
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Perform a forward pass to obtain model outputs
        out = model(data)
        # Get the predicted labels by taking the argmax over the output probabilities
        predictions = out.argmax(dim=1)
        # Calculate accuracy using the specified mask (e.g., validation or test nodes)
        accuracy = accuracy_score(data.y[mask].cpu(), predictions[mask].cpu())
    return accuracy

if __name__ == "__main__":
    # Load the data from the given .npz file
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Load the model (ensure the model has been properly trained)
    model = GNN(in_channels=data.num_features, out_channels=4)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model = model.to(device)

    # Calculate and print the accuracy on validation and test sets
    val_accuracy = calculate_accuracy(model, data, data.val_mask)
    test_accuracy = calculate_accuracy(model, data, data.test_mask)
    print(f'Accuracy on validation nodes: {val_accuracy * 100:.2f}%')
    print(f'Accuracy on test nodes: {test_accuracy * 100:.2f}%')

    # Visualize the embeddings (for all nodes)
    visualize_embeddings(model, data)

