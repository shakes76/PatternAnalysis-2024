import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from dataset import ADNIDataset, get_dataloader
from modules import ViTClassifier

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """
    Train the Vision Transformer model for Alzheimer's classification.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    stopping_epoch = epochs
    down_consec = 0

    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping check
        if epoch > 0 and val_acc < val_accs[-2]:
            down_consec += 1
        else:
            down_consec = 0
        if down_consec >= 4:
            stopping_epoch = epoch + 1
            break

    # Save the best model
    torch.save(model, "adni_vit.pt")

    # Plot training and validation metrics
    plot_metric(stopping_epoch, 'loss', train_losses, val_losses)
    plot_metric(stopping_epoch, 'accuracy', train_accs, val_accs)

    return model

def plot_metric(stopping_epoch: int, metric_type: str, train_data: list, val_data: list):
    """
    Helper function to plot a given metric
    """
    plt.figure()
    plt.plot(range(1, stopping_epoch+1), train_data, label = f"Training {metric_type}")
    plt.plot(range(1, stopping_epoch+1), val_data, label=f"Validation {metric_type}", color='orange')
    plt.xlabel('Epoch')
    plt.ylabel(metric_type)
    plt.legend()
    plt.title(f"Training {metric_type} vs validation {metric_type}")
    plt.savefig(f"Training_vs_validation_{metric_type}_{int(time.time())}.png")

def main():
    """
    Main execution function.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialise hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 20
    LR = 1e-4

    # Initialise data loaders
    train_loader, val_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)

    # Initialise model
    model = ViTClassifier().to(device)

    # Run training
    trained_model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

if __name__ == "__main__":
    main()