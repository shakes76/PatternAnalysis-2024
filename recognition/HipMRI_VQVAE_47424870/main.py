from dataset import get_dataloader
import os
import matplotlib.pyplot as plt
from modules import Encoder, Decoder, VQVAE
import torch

def visualize_batch(batch, num_images=4):
    """
    Visualizes a batch of images.
    
    Args:
        batch (torch.Tensor): Batch of images to visualize.
        num_images (int): Number of images to display from the batch.
    """
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, img in enumerate(batch[:num_images]):
        # Convert tensor to numpy and squeeze to remove any singleton dimensions
        axes[i].imshow(img.squeeze().numpy(), cmap='gray')
        axes[i].axis('off')
    plt.show()

def test_encoder():
    """Tests the Encoder module by passing a sample input and printing the output shape."""
    print("Testing Encoder...")

    # Adjusted parameter names to match the Encoder class definition
    encoder = Encoder(input_dim=1, hidden_dim=128)  # Using input_dim and hidden_dim
    sample_input = torch.randn(16, 1, 64, 64)  # Create a random batch of 16, 1-channel, 64x64 images
    encoded_output = encoder(sample_input)  # Pass through encoder
    print(f"Encoded Output Shape: {encoded_output.shape}")  # Output should have shape [16, latent_dim]

def test_decoder():
    """Tests the Decoder module by passing a sample encoded input and printing the output shape."""
    print("Testing Decoder...")

    # Adjusted parameter names to match the Decoder class definition
    embedding_dim = 512  # This should match the output channels of the encoder (hidden_dim * 4)
    hidden_dim = 128     # This should match the hidden_dim value used in the encoder
    output_dim = 1       # Output dimension should be 1, as we want a single channel output for grayscale images

    decoder = Decoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)  # Instantiate decoder

    # Create a sample encoded input with shape matching the Encoder's output
    sample_encoded_input = torch.randn(16, embedding_dim, 8, 8)  # (Batch size, Channels, Height, Width)
    decoded_output = decoder(sample_encoded_input)  # Pass through decoder

    print(f"Decoded Output Shape: {decoded_output.shape}")  # Output should have shape [16, 1, 64, 64]

def test_vqvae():
    """Tests the VQVAE model by passing a sample input and printing the output shapes and losses."""
    print("Testing VQVAE...")

    # Adjust parameters to match the VQVAE class definition
    input_dim = 1              # Input dimension should match the number of channels in the input image (1 for grayscale)
    hidden_dim = 128           # Hidden dimension used in the Encoder/Decoder
    num_embeddings = 512       # Number of embeddings (you can adjust this based on your VQVAE configuration)
    embedding_dim = 64         # Dimension of each embedding vector

    # Instantiate the VQVAE model with the correct parameters
    vqvae = VQVAE(input_dim=input_dim, hidden_dim=hidden_dim, num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    # Create a sample input with shape matching the input image shape
    sample_input = torch.randn(16, input_dim, 64, 64)  # (Batch size, Channels, Height, Width)
    
    # Pass through VQVAE
    reconstructed_output, quantisation_loss = vqvae(sample_input)

    print(f"Reconstructed Output Shape: {reconstructed_output.shape}")  # Expected shape: (16, 1, 64, 64)
    print(f"Quantisation Loss: {quantisation_loss.item():.4f}")    # Output should have shape corresponding to latent space

if __name__ == '__main__':
    # Define data directory
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "keras_slices", "keras_slices_train")
    print(f"Data Directory: {data_dir}")

    # Create dataloader using custom function
    dataloader = get_dataloader(root_dir=data_dir, batch_size=16, image_size=64, shuffle=True)

    # Iterate through the dataloader and visualize the first batch
    for batch_idx, (images, _) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1} | Image Batch Shape: {images.shape}")
        if batch_idx == 0:  # Visualize only the first batch
            visualize_batch(images)
            break

    # Run tests for the custom modules
    print("\n=== Testing Modules ===")
    test_encoder()
    test_decoder()
    test_vqvae()