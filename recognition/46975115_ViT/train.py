import torch
from modules import VisionTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_dataloaders
from utils import get_hyperparameters, get_optimizer
import matplotlib.pyplot as plt

def train(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device for computation (CPU or GPU).

    Returns:
        tuple: (average loss, accuracy) for the epoch.
    """

    model.train()
    # Track parameters for total loss and correct number of predictions
    total_loss = 0
    correct = 0

    for inputs, labels in dataloader:
        # Move data to device
        inputs, labels = inputs.to(device), labels.long().to(device)
        # Reset gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        # Compute loss and backpropogate
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss 
        total_loss += loss.item()
        # Get predicted class
        predictions = outputs.argmax(dim=1)
        # Count correct predictions
        correct += (predictions == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on validation data.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device for computation.

    Returns:
        tuple: (average loss, accuracy) on the validation set.
    """
        
    model.eval()
    total_loss = 0
    correct = 0

    # Disable gradient calculations
    with torch.no_grad():
        # Same as train function
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def evaluate_test(model, test_loader, criterion, device):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (Loss): Loss function.
        device (torch.device): Device for computation.

    Returns:
        tuple: (average loss, accuracy) on the test dataset.
    """
        
    model.eval()
    total_loss = 0
    correct = 0

    #Disable gradient calculations
    with torch.no_grad():
        # Same as above functions
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(test_loader), correct / len(test_loader.dataset)

def plot_metrics(train_acc, val_acc, train_loss, val_loss):
    """
    Plots the training and validation accuracy and loss over epochs.

    Args:
        train_acc (list): List of training accuracies per epoch.
        val_acc (list): List of validation accuracies per epoch.
        train_loss (list): List of training losses per epoch.
        val_loss (list): List of validation losses per epoch.
    """
        
    epochs = len(train_acc)
    plt.figure(figsize=(12, 5))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_acc, label='Train Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Plot train and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_loss, label='Train Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('accuracy_loss_plot.png')

def main():
    """
    Main function to initialize, train, evaluate, and plot results of the Vision Transformer model.
    """
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = get_hyperparameters() # Load hyparameters from utils.py
    model = VisionTransformer().to(device)
    
    # Load training, validation, and test data
    base_data_dir = '/home/groups/comp3710/ADNI/AD_NC'
    train_loader, val_loader, test_loader = get_dataloaders(base_data_dir, batch_size=params['batch_size'])
    
    criterion = torch.nn.CrossEntropyLoss() # Cross-entropy loss for classification
    optimizer = get_optimizer(model, params) # Get optimizer with specified parameters
    # Reduce learning rate on plateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    # Get number of training epochs
    epochs = params['num_epochs']
    
    # Lists to store the training progress for plotting later
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store metrics for plotting
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
    
    # Save best model
    torch.save(model.state_dict(), 'best_model.pth')

    # After training, plot the metrics
    plot_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = evaluate_test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()
