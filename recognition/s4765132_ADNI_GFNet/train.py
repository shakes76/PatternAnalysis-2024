import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from modules import GFNet 
from dataset import train_loader, val_loader, test_loader  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, Optimizer
model = GFNet(num_classes=2).to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_and_validate(model, train_loader, val_loader, num_epochs=10, patience=3):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    best_val_loss = float('inf')  # Initialize the best validation loss
    epochs_no_improve = 0  # Count epochs with no improvement

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  
            torch.save(model.state_dict(), './result/best_model.pth')  
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break


    # Plot training loss and accuracy
    plt.figure(figsize=(10, 6))
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.savefig("./result/train_loss.png")
    plt.show()

    plt.figure()
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.savefig("./result/train_val_accuracy.png") 
    plt.show()

 
train_and_validate(model, train_loader, val_loader, num_epochs=100, patience=5)
