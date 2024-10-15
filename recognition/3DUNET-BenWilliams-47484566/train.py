import torch.optim as optim

import modules.py

# Example setup
model = UNet3D(in_channels=3, out_channels=3)

criterion = crossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]))
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1

device='cuda'

# Training loop (simplified)
def train_model(model, dataloader, criterion, optimizer):
    model.train()  # Set model to training mode
    model = model.to(device)
    for epoch in range(epochs):
        loss = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()

            outputs = model(inputs
            loss = criterion(outputs, labels))

            loss.backward()
            optimizer.step()

            loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}')
