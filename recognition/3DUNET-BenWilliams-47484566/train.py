import torch.optim as optim

import modules.py

# Example setup
model = UNet3D(in_channels=3, out_channels=3)


optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1

device='cuda'

# Training loop (simplified)
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()  # Set model to training mode
    model = model.to(device)
    for epoch in range(num_epochs):
        loss = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()

            outputs = model(inputs)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs}')
