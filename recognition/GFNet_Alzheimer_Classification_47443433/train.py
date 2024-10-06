import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ADNIDataset, data_transform
from modules import GFNet  # Import GFNet directly

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 32  # You might need to adjust this depending on your GPU memory
learning_rate = 0.001

# Load the dataset
train_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/train', transform=data_transform)
test_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/test', transform=data_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and optimizer
model = GFNet(img_size=224, patch_size=16, num_classes=2, embed_dim=768, depth=12).to(device)  # Initialize GFNet
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

# Save the model checkpoint
torch.save(model.state_dict(), 'gfnet_model.ckpt')