import torch
import matplotlib.pyplot as plt
import os
from modules import VQVAE
from dataset import get_dataloader
from skimage.metrics import structural_similarity as ssim 

# Define the function for loading the VQVAE model
def load_vqvae_model(model_path, in_channels=1, hidden_channels=128, res_channels=32, nb_res_layers=2, nb_levels=3, embed_dim=64, nb_entries=512, scaling_rates=[8, 4, 2], device='cpu'):
    vqvae = VQVAE(in_channels=in_channels, hidden_channels=hidden_channels, res_channels=res_channels, nb_res_layers=nb_res_layers, nb_levels=nb_levels, embed_dim=embed_dim, nb_entries=nb_entries, scaling_rates=scaling_rates)
    
    # Load the pre-trained model weights
    vqvae.load_state_dict(torch.load(model_path, map_location=device))  
    
    vqvae.to(device)
    vqvae.eval()  # Set to evaluation mode
    return vqvae

def predict_vqvae(model, image_dir, save_images=True, output_dir='reconstructed_images', num_images=20, device='cpu'):
    # Load test data
    test_loader = get_dataloader(image_dir, batch_size=1, shuffle=False, device=device)
    
    # Create directory for saving images if needed
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loaded {len(test_loader)} images for testing.")
    ssim_list = []
    
    # Predict and visualize results
    with torch.no_grad():
        for i, (image, _) in enumerate(test_loader):
            if i >= num_images:
                break

            image = image.to(device)

            # Get the reconstructed image
            reconstructed, _ = model(image)

            # Convert images to NumPy arrays for SSIM calculation
            original_image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HWC format
            reconstructed_image_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()  # HWC format

            # Calculate SSIM with dynamically adjusted win_size
            min_dim = min(original_image_np.shape[:2])  # smallest of height or width
            win_size = min(5, min_dim) if min_dim >= 5 else min_dim  # Ensure win_size <= min_dim and is odd

            ssim_value = ssim(
                original_image_np, reconstructed_image_np,
                channel_axis=-1,  # for multichannel images
                data_range=original_image_np.max() - original_image_np.min(),
                win_size=win_size
            )
            ssim_list.append(ssim_value)

            # Visualize and save images as before
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

    avg_ssim = sum(ssim_list) / len(ssim_list)
    print(f"Average SSIM over {len(ssim_list)} images: {avg_ssim:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.path.dirname(__file__)
    image_dir = os.path.join(current_dir, "keras_slices", "keras_slices", "keras_slices_train")
    model_path = os.path.join(current_dir,'vqvae_model.pth')
    model = load_vqvae_model(model_path=model_path, device=device)
    predict_vqvae(model=model, image_dir=image_dir, device=device)