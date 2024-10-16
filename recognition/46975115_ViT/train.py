import torch
from modules import VisionTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_dataloaders
from utils import get_hyperparameters, get_optimizer
import matplotlib.pyplot as plt

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.long().to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def evaluate_test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)  # Convert labels to long (integer type)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # For binary classification, use argmax to get the predicted class
            correct += (outputs.argmax(dim=1) == labels).sum().item()
    # Return average loss and accuracy
    return total_loss / len(test_loader), correct / len(test_loader.dataset)

def plot_metrics(train_acc, val_acc, train_loss, val_loss):
    epochs = len(train_acc)
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_acc, label='Train Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_loss, label='Train Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('accuracy_loss_plot.png')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = get_hyperparameters()
    
    model = VisionTransformer().to(device)
    
    base_data_dir = '/home/groups/comp3710/ADNI/AD_NC'
    train_loader, val_loader, test_loader = get_dataloaders(base_data_dir, batch_size=params['batch_size'])
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, params)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_acc = 0
    early_stopping_patience = 3
    patience_counter = 0

    epochs = params['num_epochs']
    
    # To store the training progress for plotting later
    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

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
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

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
