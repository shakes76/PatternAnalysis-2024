"""File for training, validating, testing"""
import time
import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from dataset import FacebookDataset
from modules import GCN
import matplotlib.pyplot as plt
import os

def train(model, loader, criterion, optimizer, numEpochs):
    """
    Training loop for the GCN model, including validation.

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

    epoch_losses = []  # List to store the loss for each epoch
    val_losses = []  # List to store the validation loss for each epoch
    
    for epoch in range(numEpochs):
        epoch_loss = 0  # Initialize the loss
        for batch in loader:
            optimizer.zero_grad()
            
            # Forward pass
            out, _ = model(batch)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])  # Compute training loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Update the loss
            epoch_loss += loss.item()
        
        # Average the loss for the epoch
        epoch_loss /= len(loader)
        epoch_losses.append(epoch_loss)  # Store average training loss

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch in loader:
                out, _ = model(batch)
                loss = criterion(out[batch.val_mask], batch.y[batch.val_mask])  # Compute validation loss
                val_loss += loss.item()
            val_loss /= len(loader)
            val_losses.append(val_loss)  # Store average validation loss

        # Set model back to training mode for the next epoch
        model.train()

        # Print status every 10 epochs or on the final
        if ((epoch + 1) % 10 == 0) or (epoch == numEpochs - 1):
            print(f"Epoch [{epoch + 1}/{numEpochs}], "
                  f"Loss: {epoch_loss:.5f}, "
                  f"Validation Loss: {val_loss:.5f}")
            # Save the model every 10 epochs or on the final
            torch.save(model.state_dict(), f"model.pth")

    endTime = time.time()
    runTime = endTime - startTime
    print("Training Time: " + str(runTime) + " seconds")

    # Store loss per epoch graph
    plt.plot(epoch_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    os.makedirs('outputs/', exist_ok=True)
    plt.savefig("outputs/epoch_losses.png")
    plt.close()



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
    # Define Parameters
    learningRate = 0.01
    numEpochs = 400

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

    #Test the Data
    test_model = GCN(dataset)
    test_model.load_state_dict(torch.load("model.pth"))
    test(model, loader) 
