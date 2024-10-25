### Testing the VQVAE Model working fine

import torch
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compute_ssim
import numpy as np
import matplotlib.pyplot as plt




# Main code to load model, generate image, and calculate SSIM
if __name__ == '__main__':
    model_path = './Model/Vqvae.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = Model(HIDDEN_DIM, RESIDUAL_HIDDEN_DIM, NUM_RESIDUAL_LAYER, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST).to(device)
    
    # Load the saved model
    model = load_model(model, model_path)
    
    # Load your test dataloader (make sure you define your dataloader)
    # dataloader = get_dataloader("HipMRI_study_keras_slices_data")
    
    # Generate the reconstructed image
    original_img, recon_img = generate_image(model, dataloader)

    # Calculate SSIM score
    ssim_score = calculate_ssim(original_img[0], recon_img[0])  # Comparing the first image in the batch

    print(f'SSIM Score: {ssim_score}')

    # Save original and reconstructed images for visual comparison
    save_image(original_img[0], 'original_img.png')
    save_image(recon_img[0], 'reconstructed_img.png')

    # Display the original and reconstructed images
    original_np = original_img[0].cpu().numpy().squeeze()  # [H, W]
    recon_np = recon_img[0].cpu().numpy().squeeze()  # [H, W]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_np, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(recon_np, cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')

    plt.show()
