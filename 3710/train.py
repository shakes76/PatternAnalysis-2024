import torch
from modules import GNN
from dataset import load_data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train(data, model, optimizer, loss_fn, epochs=100, patience=10):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    # Initialize variables to track the best validation loss and early stopping
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0  # Track the number of epochs without improvement

    # Lists to record losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        out = model(data)  # Forward pass
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])  # Compute training loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        # Record training loss
        train_losses.append(loss.item())

        # Calculate training accuracy
        train_predictions = out[data.train_mask].argmax(dim=1)
        train_accuracy = accuracy_score(data.y[data.train_mask].cpu(), train_predictions.cpu())
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        with torch.no_grad():
            out = model(data)  # Recompute outputs
            val_loss = loss_fn(out[data.val_mask], data.y[data.val_mask])  # Compute validation loss
            val_predictions = out[data.val_mask].argmax(dim=1)
            val_accuracy = accuracy_score(data.y[data.val_mask].cpu(), val_predictions.cpu())

        # Record validation loss and accuracy
        val_losses.append(val_loss.item())
        val_accuracies.append(val_accuracy)

        # Print progress
        print(f'Epoch {epoch}, Training Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy * 100:.2f}%, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

        # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model state
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1

        # Check if early stopping is needed
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # After training, load the best model state if it exists
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, "model.pth")
    else:
        torch.save(model.state_dict(), "model.pth")

    # Plot and save loss curves
    epochs_range = range(len(train_losses))

    plt.figure()
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.savefig('../plot/loss_over_epochs.png')
    plt.close()

    # Plot and save accuracy curves
    plt.figure()
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.savefig('../plot/accuracy_over_epochs.png')
    plt.close()

if __name__ == "__main__":
    # Load the data from the specified .npz file
    npz_file_path = '/Users/zhangxiangxu/Downloads/3710_report/facebook.npz'
    data = load_data(npz_file_path)

    # Initialize model, optimizer, and loss function
    model = GNN(in_channels=data.num_features, out_channels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model with early stopping
    train(data, model, optimizer, loss_fn, epochs=100, patience=10)
    print(f"Number of training nodes: {data.train_mask.sum().item()}")
