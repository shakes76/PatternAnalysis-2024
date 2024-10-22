from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import time
import torch.optim as optim
from modules import *
from dataset import *
from utils import *

# define the device used to train
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('Device:', device)

def evaluate_model_on_loader(model, data_loader):
    """Evaluates the model on a given DataLoader.

    Args:
        model : The PyTorch model to evaluate.
        data_loader : The DataLoader for evaluation data.

    Returns:
        tuple: Average loss and accuracy of the model on the provided dataset.
    """

    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=5):
    """
    Function to handle the training model using defined hyperparameters

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        criterion: Loss function.
        optimizer: Optimizer for updating the model parameters.
        scheduler: Learning rate scheduler.
        num_epochs: Number of epochs to train the model (default: 100).
        patience: Number of epochs to wait before early stopping (default: 5).

    Returns:
        tuple: Lists of training and validation losses and accuracies for each epoch.
    """

    early_stopping = EarlyStopping(patience=patience)
    train_losses = []
    val_losses = []

    train_accs= []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        # Calculate average training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate_model_on_loader(model, val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        end = time.time()

        print(f'Epochs-[{epoch+1}/{num_epochs}]'
            f'\tTrain Loss: {train_loss:.4f}, \tTrain Accuracy: {train_accuracy:.4f}%, '
              f'\tValidation Loss: {val_loss:.4f}, \tValidation Accuracy: {val_accuracy:.4f}%, '
              f'\tTime: {end - start:.2f} seconds')

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return train_losses, val_losses, train_accs, val_accs

class EarlyStopping:
    """
    Implements early stopping to stop training when validation loss does not improve.

    Args:
        patience : Number of epochs to wait before stopping (default: 5).
        min_delta : Minimum change in the validation loss to qualify as improvement (default: 0).
    """
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """Checks whether to stop training based on validation loss.

        Args:
            val_loss: The current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

if __name__ == "__main__":

    """Main function to train, evaluate, and save the CvT model."""

    model = CvT(in_channels=3, num_classes=2, embed_dim=32, num_heads=8, mlp_dim=128, num_transformer_blocks=4, dropout=0.3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print('Start training')
    start_time = time.time()
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    end_time = time.time()

    total_training_time = end_time - start_time
    minutes, seconds = divmod(total_training_time, 60)
    print(f'Training time: {minutes} minutes, {seconds} seconds')

    print('End training')

    # visualizae train and val loss and accuracy
    show_plot_loss(train_losses, val_losses)
    show_plot_accuracy(train_accuracies, val_accuracies)

    # save th model
    torch.save(model.state_dict(), 'cvt_model.pth')

    # testing
    print('Eval start!')
    test_accuracy = evaluate_model_on_loader(model, test_loader)[1]
    print('Test accuracy:', test_accuracy)
    print('Eval end!')



