from dataset import get_dataloaders
from modules import GFNet
import torch
import torch.nn as nn
import torch.optim as optim

# Select the GPU is available otherwise select the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Train the specified model returning the average loss and accuracy """
def train(model, dataloader, criterion, optimiser):
    model.train()
    num_loss = 0.0
    num_correct = 0

    # Training loop
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimiser.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        num_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        num_correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy
    avg_loss = num_loss / len(dataloader.dataset)
    accuracy = num_correct / len(dataloader.dataset)

    return avg_loss, accuracy

""" Evaluate the generated model returning the avarage loss and accuracy """
def evaluate(model, dataloader, criterion):
    model.eval()
    num_loss = 0.0
    num_correct = 0

    with torch.no_grad():
        # Evaluation loop
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            num_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            num_correct += (predicted == labels).sum().item()

    # Calculate the average loss and accuracy
    avg_loss = num_loss / len(dataloader.dataset)
    accuracy = num_correct / len(dataloader.dataset)

    return avg_loss, accuracy

# Get the dataloaders from dataset.py
train_dataloader, test_dataloader = get_dataloaders()

# Create the components nessesary for training
model = GFNet().to(device)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
# Loop through each of the epochs
for epoch in range(epochs):
    # Train the model in the current epoch
    train_loss, train_accuracy = train(model, train_dataloader, criterion, optimiser)

    # Print the results of the epoch
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train - Loss: {train_loss:.4f}, Accuracy: {100*train_accuracy:.2f}%")

# Evaluate trained model on the test data
test_loss, test_accuracy = evaluate(model, test_dataloader, criterion)

# Print the results of the evaluation
print(f"Test - Loss: {test_loss:.4f}, Accuracy: {100*test_accuracy:.2f}%")