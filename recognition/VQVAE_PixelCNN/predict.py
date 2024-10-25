import torch
import torch.nn.functional as F
import numpy as np
from train_PixelCNN import *
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compute_ssim
import numpy as np
import matplotlib.pyplot as plt


# Function to generate reconstructed image from VQ-VAE model
def generate_image(model, dataloader):
    with torch.no_grad():
        for batch in dataloader:
            original_img = batch.to(device)  # Take one batch of images
            loss, recon_img, perplexity, _ = model(original_img)
            return original_img, recon_img

# Function to calculate SSIM score
def calculate_ssim(original_img, recon_img):
    # Move tensors to CPU and convert to NumPy
    original_img = original_img.squeeze(0).cpu().numpy()  # [H, W]
    recon_img = recon_img.squeeze(0).cpu().numpy()  # [H, W]
    
    # Calculate SSIM, note that pixel values should be in the range [0, 1] or [-1, 1]
    ssim_score = compute_ssim(original_img, recon_img, data_range=2.0, channel_axis=None)

    return ssim_score


## Comparing the generated image with the dataset images
def calculate_ssim_with_dataset(model, generated_sample, dataloader, threshold=0.7):
    # Decode the generated sample without gradient tracking
    with torch.no_grad():
        re2 = model.decode(generated_sample)

    # Convert to numpy
    re2_np = re2.cpu().numpy().squeeze()  # [H, W] format

    # Save the generated image (re2)
    save_image(torch.tensor(re2_np), "generated_image.png")

    # Initialize variables for saving the first found image with SSIM > threshold
    first_match_img = None
    first_ssim_score = 0.0

    # Loop through the dataset to find the first image with SSIM > threshold
    for idx, data in enumerate(dataloader):
        # Assuming data[0] contains the image tensor
        dataset_img = data[0].numpy().squeeze()  # [H, W] format

        # Compute SSIM between the generated image and the dataset image
        ssim_score = compute_ssim(re2_np, dataset_img, data_range=2.0, channel_axis=None)

        if ssim_score > threshold:
            first_match_img = dataset_img
            first_ssim_score = ssim_score

            # Save the matching original image
            save_image(torch.tensor(first_match_img), f"first_match_image_{idx}.png")

            print(f"Found matching image at index {idx} with SSIM score: {ssim_score:.2f}")
            break

    if first_match_img is None:
        print(f"No image found with SSIM score > {threshold}.")
        return

    # Display the generated (reconstructed) image and the original found image side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(first_match_img, cmap='gray')
    axs[0].set_title('Found Original Image')
    axs[0].axis('off')

    axs[1].imshow(re2_np, cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')

    plt.show()




def display_original_recon_image(model, dataloader):
    # Generate the reconstructed image
    original_img, recon_img = generate_image(model, dataloader)

    # Calculate SSIM score
    ssim_score = calculate_ssim(original_img[0], recon_img[0])  # Comparing the first image in the batch

    print(f'SSIM Score: {ssim_score}')

    # Save original and reconstructed images for visual comparison
    save_image(original_img[0], f'Output/original_img.png')
    save_image(recon_img[0], f'Output/reconstructed_img.png')

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
    plt.close()



# Function to generate an output image from PixelCNN
def generate_latent_space(pixelcnn_model, shape, device):
    """
    Generates an image sample from a trained PixelCNN model.

    Args:
        pixelcnn_model: Trained PixelCNN model.
        shape: Tuple (batch_size, channels, height, width) representing the shape of the output.
        device: The device (CPU/GPU) to run the model on.

    Returns:
        A tensor of generated images with shape (batch_size, channels, height, width).
    """
    # Set model to evaluation mode
    pixelcnn_model.eval()

    # Initialize an empty image tensor of zeros
    batch_size, channels, height, width = shape
    generated_latent = torch.zeros((batch_size, channels, height, width), device=device)

    # Generate pixels row by row, column by column
    for i in range(height):
        for j in range(width):
            # Forward pass through PixelCNN
            with torch.no_grad():
                output = pixelcnn_model(generated_latent)

            # Get the output for the current pixel and sample from the distribution
            # Use softmax to get probabilities, then sample from the probabilities
            pixel_probabilities = F.softmax(output[:, :, i, j], dim=1)
            pixel_values = torch.multinomial(pixel_probabilities, 1).float()

            # Assign the generated pixel to the image
            generated_latent[:, :, i, j] = pixel_values.squeeze(-1)

    return generated_latent


if __name__ == "__main__":

    dataloader = get_dataloader("HipMRI_study_keras_slices_data", batch_size = BATCH_SIZE)

    # Initialize VQVAE model and load weights
    model = Model(HIDDEN_DIM, RESIDUAL_HIDDEN_DIM, NUM_RESIDUAL_LAYER, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST).to(device)
    model.load_state_dict(torch.load('./Model/Vqvae.pth'))

    # Initialize PixelCNN
    pixelcnn_model = PixelCNN().to(device)
    pixelcnn_model.load_state_dict(torch.load('./Model/pixelCNN.pth'))

    # Generating Image from VQVAE Model
    display_original_recon_image(model, dataloader)

    # Generate the Image using the pixelCNN model
    shape = (1, 128, 64, 32)  # batch size, channels, height, width
    generated_sample = generate_latent_space(pixelcnn_model, shape, device)
    calculate_ssim_with_dataset(model, generated_sample, dataloader)