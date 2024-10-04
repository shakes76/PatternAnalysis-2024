import torch
import torch.nn as nn
import torch.optim as optim
from dataset import dataloader_train, dataloader_test
from modules import GFNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GFNet(embed_dim=384, img_size=256, patch_size=16, num_classes=1000).to(device)
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
model.train()
for epoch in range(epochs):
    running_loss = 0.0

    for input, label in dataloader_train:
        input, label = input.to(device), label.to(device)

        optimiser.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {running_loss/len(dataloader_train)}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in dataloader_train:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')