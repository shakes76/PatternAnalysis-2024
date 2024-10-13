import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from modules import Encoder, Decoder, VQVAE  # Import components from modules.py
from dataset import get_dataloader  # Import data loader from dataset.py


def train_vqvae(num_epochs=50, batch_size=32, lr=1e-4, device='cpu'):
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
    train_image_dir = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_train")
    val_image_dir = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_seg_validate")  # Update this path for validation data

    # Load training and validation data
    train_loader = get_dataloader(train_image_dir, batch_size=batch_size)
    val_loader = get_dataloader(val_image_dir, batch_size=batch_size)  # Use the same get_dataloader function for validation

    # Print out a sample batch shape
    for (batch, _) in train_loader:
        print(type(batch), batch.shape)  # This should print <class 'torch.Tensor'>
        break

    # Training loop
    train_loss_list = []
    val_loss_list = []  # To keep track of validation loss
    for epoch in range(num_epochs):
        vqvae.train()
        epoch_loss = 0

        # Training phase
        for (batch, _) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, quantisation_loss = vqvae(batch)  # Forward pass
            loss = loss_fn(reconstructed, batch) + quantisation_loss  # Combine losses
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Validation phase
        vqvae.eval()  # Set the model to evaluation mode
        epoch_val_loss = 0
        
        with torch.no_grad():  # No need to compute gradients during validation
            for (val_batch, _) in val_loader:
                val_batch = val_batch.to(device)
                reconstructed_val, quantisation_loss_val = vqvae(val_batch)  # Forward pass on validation data
                val_loss = loss_fn(reconstructed_val, val_batch) + quantisation_loss_val  # Combine losses
                epoch_val_loss += val_loss.item()

        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save the model weights after training
    torch.save(vqvae.encoder.state_dict(), 'encoder.pth')
    torch.save(vqvae.decoder.state_dict(), 'decoder.pth')
    print("Model weights saved as 'encoder.pth' and 'decoder.pth'.")

    # Plot the training and validation loss
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("training_validation_loss.png")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_vqvae(device=device)
