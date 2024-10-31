import torch
import matplotlib.pyplot as plt
import os
from modules import VQVAE
from dataset import get_dataloader
from skimage.metrics import structural_similarity as ssim 

# Load a pre-trained VQVAE model from file
def load_vqvae_model(model_path, in_channels=1, hidden_channels=128, res_channels=32, 
                     nb_res_layers=2, nb_levels=3, embed_dim=128, nb_entries=1024, 
                     scaling_rates=[8, 4, 2], device='cpu'):
    """
    Load a pre-trained VQVAE model from file.
    
    Parameters:
        model_path (str): Path to the model file (.pth) containing saved weights.
        in_channels (int): Number of input channels (default=1 for grayscale images).
        hidden_channels (int): Number of hidden channels in the model.
        res_channels (int): Number of residual channels in the model.
        nb_res_layers (int): Number of residual layers in the model.
        nb_levels (int): Number of hierarchical levels in the VQ-VAE.
        embed_dim (int): Embedding dimension in the quantization layers.
        nb_entries (int): Number of embeddings in the quantization layers.
        scaling_rates (list of int): List of scaling rates for each level in the model.
        device (str): Device to load the model on ('cpu' or 'cuda' for GPU).
        
    Returns:
        vqvae (VQVAE): The VQVAE model loaded with pre-trained weights and set to evaluation mode.
    """
    # Initialize VQVAE model with specified parameters
    vqvae = VQVAE(in_channels=in_channels, hidden_channels=hidden_channels, 
                  res_channels=res_channels, nb_res_layers=nb_res_layers, 
                  nb_levels=nb_levels, embed_dim=embed_dim, nb_entries=nb_entries, 
                  scaling_rates=scaling_rates)
    
    # Load the pre-trained model weights
    vqvae.load_state_dict(torch.load(model_path, map_location=device))  
    vqvae.to(device)
    vqvae.eval()  # Set to evaluation mode to disable dropout, etc.
    return vqvae

# Predict and reconstruct images with VQVAE, and save reconstructed images
def predict_vqvae(model, image_dir, save_images=True, output_dir='reconstructed_images', 
                  num_images=30, device='cpu'):
    """
    Predict and reconstruct images with VQVAE, and save reconstructed images if specified.
    
    Parameters:
        model (VQVAE): The VQVAE model to use for reconstruction.
        image_dir (str): Path to the directory containing images for reconstruction.
        save_images (bool): If True, saves reconstructed images to output_dir.
        output_dir (str): Directory where reconstructed images will be saved.
        num_images (int): Number of images to reconstruct and save (default=30).
        device (str): Device to use for inference ('cpu' or 'cuda' for GPU).
        
    Returns:
        None. Saves reconstructed images and prints SSIM scores.
    """
    # Load test data as a PyTorch DataLoader object
    test_loader = get_dataloader(image_dir, batch_size=1, shuffle=False, device=device)
    
    # Create directory to save reconstructed images, if saving is enabled
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loaded {len(test_loader)} images for testing.")
    ssim_list = []  # To store SSIM scores for each image
    
    # Predict and visualize results
    with torch.no_grad():  # Disable gradient tracking for faster inference
        for i, (image, _) in enumerate(test_loader):
            if i >= num_images:  # Stop if reached the number of images to predict
                break

            image = image.to(device)

            # Reconstruct the image using the VQVAE model
            reconstructed, _ = model(image)

            # Convert images to NumPy arrays for SSIM calculation
            original_image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HWC format
            reconstructed_image_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HWC format

            # Calculate SSIM with a dynamically adjusted window size
            min_dim = min(original_image_np.shape[:2])  
            win_size = min(5, min_dim) if min_dim >= 5 else min_dim  # Ensure win_size <= min_dim and is odd

            # Compute SSIM score for image quality assessment
            ssim_value = ssim(
                original_image_np, reconstructed_image_np,
                channel_axis=-1,  # For multichannel images
                data_range=original_image_np.max() - original_image_np.min(),
                win_size=win_size
            )
            ssim_list.append(ssim_value)

            # Visualize and save original and reconstructed images
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(original_image_np, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title('Reconstructed Image')
            plt.imshow(reconstructed_image_np, cmap='gray')
            plt.axis('off')
            
            filename = os.path.join(output_dir, f'reconstructed_{i}.png')
            plt.savefig(filename)
            plt.close()
            print(f"Saved reconstructed image {i} to {filename}.")
            print(f"SSIM for image {i}: {ssim_value:.4f}")

    # Calculate and print the average SSIM across all images
    avg_ssim = sum(ssim_list) / len(ssim_list)
    print(f"Average SSIM over {len(ssim_list)} images: {avg_ssim:.4f}")

if __name__ == "__main__":
    # Select device for computation, using GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define paths for the images and the model
    current_dir = os.path.dirname(__file__)
    image_dir = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_train")
    model_path = os.path.join(current_dir, 'vqvae_model.pth')
    
    # Load the pre-trained model
    model = load_vqvae_model(model_path=model_path, device=device)
    
    # Run the prediction and save reconstructed images
    predict_vqvae(model=model, image_dir=image_dir, device=device)
