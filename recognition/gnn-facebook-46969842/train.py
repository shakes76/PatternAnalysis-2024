"""File for training, validating, testing"""
import time
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from dataset import FacebookDataset
from modules import GCN

def train(model, loader, criterion, optimizer, numEpochs):
    """
    Training loop for the GCN model.

    Parameters:
        model (torch.nn.Module): The GCN model to be trained.
        loader (DataLoader): DataLoader for batching the dataset.
        criterion (torch.nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        numEpochs (int): The number of epochs to train the model.
    """
    startTime = time.time()
    model.train()  # Set model to training mode
    print("> Training")
    for epoch in range(numEpochs):
        for batch in loader:
            optimizer.zero_grad()  # Clear gradients
            
            # Forward pass
            out, _ = model(batch)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])  # Compute loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f"Epoch [{epoch + 1}/{numEpochs}], Loss: {loss.item():.5f}")

    endTime = time.time()
    runTime = endTime - startTime
    print("Training Time: " + str(runTime) + " seconds")

if __name__ == "__main__":
    # Define Parameters
    learningRate = 0.01
    numEpochs = 200

    # Load dataset
    dataset = FacebookDataset(path='facebook.npz')
    data = dataset.get_data()  # Get the data object

    # Create DataLoader for batching
    loader = DataLoader([data], batch_size=32, shuffle=True)

    # Create the model
    model = GCN(dataset)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learningRate)
    train(model, loader, criterion, optimizer, numEpochs)
