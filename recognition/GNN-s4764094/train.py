from torch.utils.data import DataLoader
from dataset import upload_dataset
from modules import GCNModel
import matplotlib.pyplot as plt
import torch.optim as optim
import torch

device = torch.device("cpu")
# Check if the CUDA && MPS for our laptop is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CPU usage.")

# Upload our dataset
tensor_edges, train_set, test_set = upload_dataset(device)

# Load our data
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

def evaluate_accuracy(model, loader, edges, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        outputs = model(features, edges)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = correct / total
    epoch_loss = test_loss / len(loader)
    return accuracy, epoch_loss

def train_model(model, train_loader, edges, criterion, optimizer, device, test_loader, num_epochs):
    model.train()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            # Training process
            optimizer.zero_grad()
            outputs = model(features, edges)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluate on train and test set after each epoch
        train_accuracy, train_loss = evaluate_accuracy(model, train_loader, edges, device)
        test_accuracy, test_loss = evaluate_accuracy(model, test_loader, edges, device)

        # List train and test losses and accuracies
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Print training and test loss and accuracy
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_losses, train_accuracies, test_losses, test_accuracies


# Define GCN model
model = GCNModel(classes=4, features=128).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.02, weight_decay=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Train the model, plot the results
train_losses, test_losses, train_accuracies, test_accuracies = train_model(
    model, train_loader, tensor_edges, criterion, optimizer, device, test_loader, num_epochs=50
)

# Plotting losses and accuracy
plt.figure(figsize=(10, 6))

# Plot train and test loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot train and test accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
