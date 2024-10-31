import os
import torch
from torchvision import transforms
from modules import VQVAE  # Import your VQVAE model
from dataset import MedicalImageDataset, get_dataloaders  # Import your dataset classes
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories for the dataset and model
test_dir = "/Users/bairuan/Documents/uqsem8/comp3710/report/cloned/PatternAnalysis-2024/recognition/VQVAE_Bairu/HipMRI_study_keras_slices_data/keras_slices_test"
model_path = "vqvae_final_model.pth"  # Change if necessary
save_dir = "reconstructed_images"  # Directory to save reconstructed images

# Hyperparameters
batch_size = 1  # Adjust based on your requirements

# Pre-processing transformation
input_transf = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.Normalize((0.5,), (0.5,))
])

def main():
    # Load the model
    model = VQVAE(1, 64, 512, 64, 2).to(device)  # Adjust parameters as necessary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create directory to save images if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load test data
    test_loader = get_dataloaders(test_dir, test_dir, test_dir, batch_size=batch_size)[2]  # Using the test loader

    # SSIM Scores
    ssim_scores = []

    # Inference and Visualization
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch = batch.to(device)
            
            # Forward pass through the model
            reconstructed_data, quantization_loss = model(batch)

            # Denormalize images
            original_image = (batch.cpu().numpy().squeeze() * 0.5 + 0.5)
            reconstructed_image = (reconstructed_data.cpu().numpy().squeeze() * 0.5 + 0.5)

            # Calculate SSIM
            ssim_score = ssim(batch, reconstructed_data, data_range=1.0).item()
            ssim_scores.append(ssim_score)

            # Save images
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title(f'Original Image {batch_idx + 1}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image, cmap='gray')
            plt.title(f'Reconstructed Image {batch_idx + 1}\nSSIM: {ssim_score:.4f}')
            plt.axis('off')

            plt.savefig(os.path.join(save_dir, f'image_{batch_idx + 1}.png'))
            plt.close()

    # SSIM statistics
    average_ssim = np.mean(ssim_scores)
    print(f"Average SSIM on test set: {average_ssim:.4f}")

if __name__ == "__main__":
    main()
