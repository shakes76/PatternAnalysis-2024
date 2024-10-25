# Import necessary libraries and modules
import torch
import numpy as np
import argparse
import utils
import matplotlib.pyplot as plt
from modules import VQVAE
import os
from skimage.metrics import structural_similarity as ssim
from utils import predict_and_reconstruct
import dataset

# Define command-line arguments for dataset directory and model save path
parser = argparse.ArgumentParser()
epochs = 100
learning_rate = 1e-3
batch_size = 16
weight_decay = 1e-5

# Model architecture parameters
n_hiddens = 512
n_residual_hiddens = 512
n_residual_layers = 32
embedding_dim = 512
n_embeddings = 1024
beta = 0.1

# Dataset and model save path arguments, for easier readability and storage
# previous parameters were changes often, therefore variables were chosen to store them
parser.add_argument("--dataset_dir", type=str, default='HipMRI_study_keras_slices_data')
parser.add_argument("--save_path", type=str, default="vqvae_data.pth")
parser.add_argument("-save", action="store_true")

# Parse command-line arguments and set device
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with specified checkpoint path
# had to include load_model function within predict.py since it used many variables
def load_model(model_path):
    model = VQVAE(n_hiddens, n_residual_hiddens, n_residual_layers,
                  n_embeddings, embedding_dim, beta).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model

def main():
    # Load test data from dataset directory
    test_path = os.path.join(args.dataset_dir, 'keras_slices_test')
    nii_files_test = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.endswith(('.nii', '.nii.gz'))]
    x_test = dataset.load_data_2D(nii_files_test, normImage=False, categorical=False)
    x_test_tensor = torch.from_numpy(x_test).float().unsqueeze(1)  # Add channel dimension

    # Create DataLoader for test data and load saved model
    test_loader = torch.utils.data.DataLoader(x_test_tensor, batch_size=batch_size)
    path = 'results/' + args.save_path
    model = load_model(path)

    # Generate reconstructions and display comparison images
    for original, reconstructed in predict_and_reconstruct(model, test_loader):
        print(f"Original shape: {original.shape}, Reconstructed shape: {reconstructed.shape}")

        # Create a single figure for displaying all images horizontally
        fig, axs = plt.subplots(5, 2, figsize=(15, 10))  # 5 rows, 2 columns

        # Display the first 5 original and reconstructed images
        for i in range(min(5, len(original))):
            # Original image
            original_img = np.squeeze(original[i], axis=0)
            axs[i, 0].imshow(original_img, cmap='gray')
            axs[i, 0].set_title(f'Original #{i}')
            axs[i, 0].axis('off')  # Hide axis for clarity

            # Reconstructed image
            if reconstructed[i].shape[0] == 1:
                reconstructed_img = np.squeeze(reconstructed[i], axis=0)
            else:
                reconstructed_img = np.mean(reconstructed[i], axis=0)

            # Compute SSIM score and plot
            ssim_score = ssim(original_img, reconstructed_img, data_range=reconstructed_img.max() - reconstructed_img.min())
            axs[i, 1].imshow(reconstructed_img, cmap='gray')
            axs[i, 1].set_title(f'#{i} SSIM Score: {ssim_score:.4f}')
            axs[i, 1].axis('off')  # Hide axis for clarity

            # Save each individual reconstructed image separately
            if args.save:
                plt.imsave(f"reconstructed_{i}.png", reconstructed_img, cmap='gray')  # Save each reconstructed image

        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()  # Show all images together

        break  # Stop after first batch for demo

if __name__ == "__main__":
    main()
