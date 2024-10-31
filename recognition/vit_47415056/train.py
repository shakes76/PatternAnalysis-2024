import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from modules import create_model
from dataset import get_dataloaders

def train_model(data_dir='/home/groups/comp3710/ADNI/AD_NC', num_epochs=20, batch_size=128, learning_rate=1e-4, num_classes=2, model_path='best_model.pth'):
    """
    Train a Vision Transformer model on a given dataset.

    Parameters:
    - data_dir (str): Path to the dataset directory.
    - num_epochs (int): Number of epochs for training.
    - batch_size (int): Number of samples per batch.
    - learning_rate (float): Learning rate for the optimizer.
    - num_classes (int): Number of output classes.
    - model_path (str): Path to save the best model.

    Returns:
    - model_path (str): Path of the saved model with the best accuracy.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data using custom DataLoader function
    dataloaders, dataset_sizes = get_dataloaders(data_dir, batch_size)

    # Initialize the model and move it to the selected device
    model = create_model(num_classes)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize lists to store losses and accuracies
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    best_acc = 0.0
    since = time.time()

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over training data
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss and correct predictions
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate average loss and accuracy for the training set
        train_loss = running_loss / dataset_sizes['train']
        train_acc = (running_corrects.double() / dataset_sizes['train']) * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Accumulate loss and correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        # Calculate average loss and accuracy for the validation set
        test_loss = running_loss / dataset_sizes['test']
        test_acc = (running_corrects.double() / dataset_sizes['test']) * 100
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # Save the model if it has the best accuracy so far
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)

        # Print epoch results
        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate total training time
    total_time = time.time() - since
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)
    print(f"Training complete in {total_minutes}m {total_seconds}s")
    print(f"Best Test Accuracy: {best_acc:.2f}%")

    return model_path