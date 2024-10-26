import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ADNIDataset, data_transform
from modules import GFNet  
from torch.cuda.amp import GradScaler, autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
batch_size = 32
learning_rate = 0.001
accumulation_steps = 2

train_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/train', transform=data_transform)
test_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/test', transform=data_transform)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = GFNet(img_size=224, patch_size=16, num_classes=2, embed_dim=768, depth=8).to(device)  # Initialize GFNet
criterion = nn.CrossEntropyLoss()
optimiser = optim.AdamW(model.parameters(), lr=learning_rate)
scaler = GradScaler() 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

    # Testin
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    model.train()

torch.save(model.state_dict(), 'gfnet_model.ckpt')