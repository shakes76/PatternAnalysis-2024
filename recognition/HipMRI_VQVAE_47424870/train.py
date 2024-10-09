import torch
import torch.nn as nn
import torch.optim as optim
from modules import Encoder, Decoder  # Import components from modules.py
from dataset import get_data_loader   # Import data loader from dataset.py
import matplotlib.pyplot as plt

# Initialize model, optimizer, and loss function
encoder = Encoder(in_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32)
decoder = Decoder(in_channels=128, out_channels=1)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
loss_fn = nn.MSELoss()

# Set up training parameters
num_epochs = 20
image_dir = ''  # Update with actual path
train_loader = get_data_loader(image_dir, batch_size=32)

# Training loop
train_loss_list = []
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    epoch_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        encoded = encoder(batch)
        decoded = decoder(encoded)
        loss = loss_fn(decoded, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss_list.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Plot the training loss
plt.plot(range(1, num_epochs + 1), train_loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

