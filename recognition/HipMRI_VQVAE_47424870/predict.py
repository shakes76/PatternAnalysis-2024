import torch
import matplotlib.pyplot as plt
import os
from modules import VQVAE
from dataset import get_dataloader

# Define the function for loading the VQVAE model
def load_vqvae_model(encoder_path, decoder_path, input_dim=1, hidden_dim=128, num_embeddings=64, embedding_dim=128, device='cpu'):
    vqvae = VQVAE(input_dim=input_dim, hidden_dim=hidden_dim, num_embeddings=num_embeddings, embedding_dim=embedding_dim, device=device)
    
    # Load the pre-trained weights
    vqvae.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    vqvae.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    vqvae.to(device)
    vqvae.eval()  # Set to evaluation mode
    return vqvae

# Define the function to predict and visualize results
def predict_vqvae(model, image_dir, save_images=True, output_dir='reconstructed_images', num_images=5, device='cpu'):
    # Load test data
    test_loader = get_dataloader(image_dir, batch_size=1, shuffle=False, device=device)
    
    # Create directory for saving images if needed
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Predict and visualize results
    with torch.no_grad():
        for i, (image, _) in enumerate(test_loader):
            if i >= num_images:
                break

            image = image.to(device)

            # Get the reconstructed image
            reconstructed, _ = model(image)
            
            # Visualize the original and reconstructed images
            plt.figure(figsize=(10, 5))
            
            # Original image
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(image.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
            
            # Reconstructed image
            plt.subplot(1, 2, 2)
            plt.title('Reconstructed Image')
            plt.imshow(reconstructed.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
            
            filename = os.path.join(output_dir, f'reconstructed_{i}.png')
            plt.savefig(filename)
            print(f"Saved reconstructed image {i} to {filename}.")

# Main function
if __name__ == "__main__":
    # Define paths for model weights and image directory
    encoder_path = 'encoder.pth'  # Path to encoder weights
    decoder_path = 'decoder.pth'  # Path to decoder weights
    image_dir = "path_to_test_images"  # Path to test image directory

    # Set up the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the VQVAE model
    vqvae_model = load_vqvae_model(encoder_path, decoder_path, device=device)

    # Run the prediction and visualization process
    predict_vqvae(vqvae_model, image_dir, device=device)
