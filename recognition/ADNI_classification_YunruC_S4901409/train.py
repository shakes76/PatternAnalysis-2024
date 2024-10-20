import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from modules import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_data_loaders


def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Visualize the training and validaton performance of GFNet across epochs.
    """

    #Assuming the number of epochs is equal to the number of loss entries
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))


    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()


    plt.tight_layout()
    plt.show()


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss= 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", unit="batch")

    for images, labels in pbar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion (outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim = 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss /len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

def validate (model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", unit="batch")
        for images, labels in pbar:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

if __name__ == "__main__":

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model, loss function, and optimizer
    model = GFNet(
            img_size=512, 
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_rate = 0.1
        )
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    model.to(device)

    # Load data
    """
    change the zip_path to the path you want and don't forget to modify the directory
    in datset.py
    """
    zip_path = "ADNI_AD_NC_2D.zip"
    extract_to = "data"
    train_loader, val_loader, test_loader = get_data_loaders(zip_path, extract_to, batch_size=32, train_split = 0.80)

    # Training loop
    best_val_accuracy = 0.0
    epochs = 20

    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        test_loss, test_accuracy = validate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'model.pth')

    print("Training complete.")

    # plot the curves
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)