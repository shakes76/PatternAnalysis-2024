# Contains source code for training, validating, testing, and saving the model.
# Requires dataset.py and modules.py to run.

import os
import torch
import torch.optim as optim
import torch.nn as nn

# For setting up matplotlib correctly
os.environ['MPLCONFIGDIR'] = '/home/Student/s4699165/Project'
import matplotlib.pyplot as plt

from dataset import get_data_loaders
from modules import get_vit_model

# Move model to the appropriate device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model and return losses and accuracies
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    model.train()  # Set the model to training mode
    train_loss = []
    train_acc = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            epoch_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)  # Get predictions
            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()  # Correct predictions

        accuracy = 100 * correct / total  # Calculate accuracy
        train_loss.append(epoch_loss / len(train_loader))  # Record average loss
        train_acc.append(accuracy)  # Record accuracy
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
        
        # Step the scheduler based on the average loss
        scheduler.step(epoch_loss / len(train_loader))

    return train_loss, train_acc

# Evaluate the model on the test set and return loss and accuracy
def test(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU if available
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    return avg_test_loss, accuracy

# Plot and save the loss and accuracy curves
def plot_metrics(train_loss, train_acc, test_loss, test_acc):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot training and test loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss During Training")

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, [test_acc] * len(epochs), label="Final Test Accuracy", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy During Training")

    # Save the plot as a PNG file
    plt.savefig("training_metrics_plot.png")
    plt.close()

def main():
    print("Running main loop now.")

    # Get data loaders with augmentations for the training set
    train_loader, test_loader = get_data_loaders(batch_size=32)
    
    # Initialize Model, Loss, Optimizer
    class_weights = torch.tensor([1.0, 1.2]).to(device)
    model = get_vit_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print("Beginning training now.")
    # Train the model and get training loss and accuracy
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, num_epochs=50)

    print("Beginning testing now.")
    # Test the model and get test loss and accuracy
    test_loss, test_accuracy = test(model, test_loader, criterion)

    # Plot and save the training and testing metrics
    plot_metrics(train_loss, train_acc, test_loss, test_accuracy)

    # Save the model
    torch.save(model.state_dict(), 'gfnet_adni_model.pth')
    print(f'Model saved with Test Accuracy: {test_accuracy:.2f}%')

if __name__ == '__main__':
    main()




