import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import get_adni_dataloader  # Import your data loader
from modules import GFNet  # Import your model

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 40
    step_size = 5  # Number of epochs to wait before reducing the learning rate
    gamma = 0.1    # Factor by which the learning rate is reduced

    # Load data
    train_loader, val_loader = get_adni_dataloader(batch_size=batch_size, train=True)

    # Initialize the model, loss function, and optimizer
    model = GFNet().to(device)  # Adjust model initialization as needed
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=3, verbose=True)


    # Lists to store loss and accuracy values
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Training and validation loop
    for epoch in range(num_epochs):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer, train_losses, train_accuracies, device)
        validate_one_epoch(epoch, model, val_loader, criterion, val_losses, val_accuracies, device)

        # Step the learning rate scheduler
        scheduler.step(val_losses[-1])

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}, '
              f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

    # Save the model
    torch.save(model.state_dict(), 'gfnet_model.pth')

    # Plot the training and validation losses and accuracies
    plot_metrics(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies)

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, train_losses, train_accuracies, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)

def validate_one_epoch(epoch, model, val_loader, criterion, val_losses, val_accuracies, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(running_loss / len(val_loader))
    val_accuracies.append(100 * correct / total)

def plot_metrics(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"Training_vs_validation.png")

if __name__ == '__main__':
    main()
