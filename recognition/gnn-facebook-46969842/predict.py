"""
predict.py

A file for predicting results from a give model. Consists
of a function test() to test the saved model.

Author: Tristan Hayes - 46969842
"""
import torch
import time
from torch_geometric.loader import DataLoader
from dataset import FacebookDataset
from modules import GCN
from utils import tsne_embeddings, pca_embeddings, umap_embeddings

def test(model, loader):
    """
    Testing loop for the GCN model.

    Parameters:
        model (torch.nn.Module): The GCN model to be tested.
        loader (DataLoader): DataLoader for the test dataset.
    """
    startTime = time.time()
    print("> Testing")
    model.eval()  # Set model to evaluation mode
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            data = batch[0] #Get the data from the batch
            
            outputs, _ = model(data)  # Forward pass
            _, predicted = torch.max(outputs[data.test_mask], 1)  # Get predicted classes
            
            total += data.y[data.test_mask].size(0)  # Total number of test examples
            correct += (predicted == data.y[data.test_mask]).sum().item()  # Count correct predictions

    accuracy = 100 * correct / total
    print("Test Accuracy: {:.2f}%".format(accuracy))

    endTime = time.time()
    runTime = endTime - startTime
    print("Testing Time: {:.2f} seconds".format(runTime))


if __name__ == "__main__":
    dataset = FacebookDataset(path='facebook.npz')
    data = dataset.get_data()
    model = GCN(dataset)
    model.load_state_dict(torch.load("model.pth")) # Loads the model from the saved file
    loader = DataLoader([data], batch_size=32, shuffle=True)
    test(model, loader)
    
    tsne_embeddings(model, data)
    pca_embeddings(model, data)
    umap_embeddings(model, data)