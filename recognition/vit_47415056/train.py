import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from modules import initialize_model
from dataset import get_train_val_loaders, get_test_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_validate(train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler, num_epochs=11, model_path="model_weights.pth"):
    """
    Train, validate, and test a model, saving the best model based on validation accuracy.

    """
    best_val_accuracy = 0
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        total_correct, total_samples = 0, len(train_loader.dataset)

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()

        train_accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}: Training Accuracy: {train_accuracy:.2%}")

        # Validation phase
        model.eval()
        val_correct, val_loss = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += outputs.argmax(dim=1).eq(labels).sum().item()

        val_accuracy = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}: Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2%}")
        scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at Epoch {epoch + 1} with Validation Accuracy: {best_val_accuracy:.2%}\n")

    # Testing logic
    model.eval()
    all_preds, all_labels = [], []
    print("\nTesting the model on the test dataset...\n")
    for images, labels in test_loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Confusion Matrix:\n{conf_matrix}\nTest Accuracy: {accuracy:.2%}")

    # Plotting metrics
    plt.figure()
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("Training Metrics")
    plt.show()

def main():
    """
    Main function to set up data loaders, initialize the model and training parameters, 
    and call the training and validation process.

    """
    # Paths to data directories
    train_data_dir = "/home/groups/comp3710/ADNI/AD_NC/train"
    test_data_dir = "/home/groups/comp3710/ADNI/AD_NC/test"

    # Load data loaders with 90-10 split
    train_loader, val_loader = get_train_val_loaders(train_data_dir)
    test_loader = get_test_loader(test_data_dir)

    # Initialize model, loss, optimizer, and scheduler
    model = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # Train, validate, and test the model
    train_and_validate(train_loader, val_loader, test_loader, model, criterion, optimizer, scheduler)

if __name__ == "__main__":
    main()