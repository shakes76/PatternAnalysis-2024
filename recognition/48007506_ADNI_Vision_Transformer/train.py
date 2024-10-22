"""
train.py

Source code for training, validating, testing and saving the vision transformer.

Author: Chiao-Yu Wang (Student No. 48007506)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from modules import GFNet
from dataset import load_data
from constants import NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, WEIGHT_DECAY
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model():
    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    train_loader, val_loader, _ = load_data()

    # Initialize the model
    model = GFNet(num_classes=2).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler to reduce learning rate if validation accuracy stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.8)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Backpropagation and optimize
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # Validation step
        val_accuracy = validate_model(model, val_loader)

        # Step the scheduler based on validation accuracy
        scheduler.step(val_accuracy)

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

def validate_model(model, val_loader):
    """
    Validates the model on the validation set.
    """
    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device).float(), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == '__main__':
    train_model()
