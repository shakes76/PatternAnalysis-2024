import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from modules import Encoder, Decoder, VQVAE  # Import components from modules.py
from dataset import get_dataloader  # Import data loader from dataset.py


def train_vqvae(num_epochs=20, batch_size=32, lr=1e-3, device='cpu'):
    # Initialize model components
    input_dim = 1  # Number of input channels
    hidden_dim = 128  # Hidden dimension size
    num_embeddings = 64  # Number of embeddings for the Vector Quantizer
    embedding_dim = 128  # Dimension of each embedding

    # Instantiate the VQVAE model
    vqvae = VQVAE(input_dim=input_dim, hidden_dim=hidden_dim, num_embeddings=num_embeddings, embedding_dim=embedding_dim, device=device)
    vqvae.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(vqvae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Set up training parameters
    current_dir = os.path.dirname(__file__)
    image_dir  = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_train")
    train_loader = get_dataloader(image_dir, batch_size=batch_size)

    for (batch, _) in train_loader:
        print(type(batch), batch.shape)  # This should print <class 'torch.Tensor'>
        break

    # Training loop
    train_loss_list = []
    for epoch in range(num_epochs):
        vqvae.train()
        epoch_loss = 0

        for (batch, _) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, quantisation_loss = vqvae(batch)  # Forward pass
            loss = loss_fn(reconstructed, batch) + quantisation_loss  # Combine losses
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss_list.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the model weights after training
    torch.save(vqvae.encoder.state_dict(), 'encoder.pth')
    torch.save(vqvae.decoder.state_dict(), 'decoder.pth')
    print("Model weights saved as 'encoder.pth' and 'decoder.pth'.")

    # Plot the training loss
    plt.plot(range(1, num_epochs + 1), train_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig("training_loss.png")

# If you want to run this script directly, you can include this check:
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_vqvae(device=device)