from torch.utils.data import DataLoader
from dataset import upload_dataset
from modules import GCNModel
import matplotlib.pyplot as plt
import torch.optim as optim
import torch

# Check if the CUDA && MPS for our laptop is available
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CPU usage.")

# Upload our dataset from DataLoader object
tensor_edges, train_set, test_set = upload_dataset(device)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

def evaluate_accuracy(model, loader, edges, criterion, device):
    """
    Evaluates the model's accuracy and loss on a given dataset.

    Parameters:
        - model: The trained GCN model used to make predictions.
        - loader: A DataLoader object containing the dataset to evaluate (training and test set with 0.7 and 0.3).
        - edges: Graph edge information, used in graph neural networks.
        - criterion: The loss function used to calculate the error between predicted results and actual datas.
        - device: The computing device, we use "mps" on MAC devices.

    Returns:
        - accuracy: The accuracy of the predictions on the dataset.
        - loss_function: The average loss computed over the dataset.
    """

    model.eval()
    total = 0
    correct = 0
    test_loss = 0.0

    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        outputs = model(features, edges)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = correct / total
    loss_function = test_loss / len(loader)
    return accuracy, loss_function


def train_evaluate_model(model, train_loader, test_loader, edges, device, learning_rate, num_epochs):
    """
        Trains the model and evaluates its performance on both the training and test sets after each epoch.
        Implement early stopping if the test accuracy does not improve for a specified number of epochs.

        Parameters:
            - model: The GCN neural network model to be trained.
            - train_loader: DataLoader object containing the training dataset.
            - test_loader: DataLoader object containing the test dataset.
            - edges: The graph edges information for the neural network.
            - device: The computing device ('mps' for MAC there) used for training and evaluation.
            - num_epochs: The total number of epochs to train the model.

        Returns:
            - train_losses: A list of training losses recorded after each epoch.
            - train_accuracies: A list of training accuracies recorded after each epoch.
            - test_losses: A list of test losses recorded after each epoch.
            - test_accuracies: A list of test accuracies recorded after each epoch.
    """

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Lists for recording losses and accuracies for both training and test set
    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    best_accuracy = 0
    unchanged_count = 0

    # The iteration of training process
    for epoch in range(num_epochs):

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features, edges)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluate on train and test set after each epoch
        train_accuracy, train_loss = evaluate_accuracy(model, train_loader, edges, criterion, device)
        test_accuracy, test_loss = evaluate_accuracy(model, test_loader, edges, criterion, device)

        # List train and test losses and accuracies for final plotting
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Print training and test loss and accuracy values
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

        # Early stopping based on test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            unchanged_count = 0
        else:
            unchanged_count += 1

        # Check if the accuracy unchanged in specified epochs
        if unchanged_count >= 10:
            print(f"Early stopping at epoch {epoch + 1}. Best Test Accuracy: {best_accuracy:.4f}")
            break

    return train_losses, train_accuracies, test_losses, test_accuracies


# Define the GCN model
model = GCNModel(classes=4, features=128).to(device)

# Train the model, plot the results
train_losses, train_accuracies, test_losses, test_accuracies = \
    train_evaluate_model(model, train_loader, test_loader, tensor_edges, device, learning_rate=0.0005, num_epochs=75)

# Plotting losses and accuracy
plt.figure(figsize=(10, 6))

# Plot the train and test loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the train and test accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
