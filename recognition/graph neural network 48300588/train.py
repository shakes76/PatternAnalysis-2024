import torch
from torch.optim import SGD  # Changed import
from torch.optim.lr_scheduler import StepLR  # Import scheduler
import matplotlib.pyplot as plt
from modules import GNNModel
from dataset import load_data
import time

def train():
    # Load data and classes
    data, classes = load_data()
    device = torch.device('cuda')
    data = data.to(device)

    # Initialize model, optimizer, and loss function
    model = GNNModel(in_channels=data.num_features, hidden_channels=64, out_channels=len(classes)).to(device)
    optimizer = SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.95)  # Changed to SGD with adjusted lr and momentum
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # Initialize scheduler
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize variables for tracking training progress
    epochs = 200
    patience = 10  # For early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update scheduler

        # Record training loss
        train_losses.append(loss.item())

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            # Calculate losses
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss.item())

            # Get predictions
            _, pred = out.max(dim=1)

            # Calculate accuracies
            train_correct = pred[data.train_mask] == data.y[data.train_mask]
            train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
            train_accuracies.append(train_acc)

            val_correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
            val_accuracies.append(val_acc)

        print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the best model
    torch.save(model.state_dict(), 'best_model.pth')

    # Plot loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss.png')

    # Plot accuracy curves
    plt.figure()
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')

    # Evaluate test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        _, pred = out.max(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    train()
