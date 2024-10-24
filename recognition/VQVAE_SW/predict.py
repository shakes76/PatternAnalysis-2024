import torch
import matplotlib.pyplot as plt
from modules import VQVAE  # import VQVAE
from dataset import get_data_loader  # import data loader
from skimage.metrics import structural_similarity as ssim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/content/data/HipMRI_study_keras_slices_data'
batch_size = 1

model = VQVAE(in_channels=1, hidden_channels=256, num_embeddings=1024, embedding_dim=128).to(device)
model.load_state_dict(torch.load('vqvae_hipmri.pth', map_location=device))
model.eval()

test_loader = get_data_loader(data_dir, subset='test', batch_size=batch_size, shuffle=False, normImage=True, target_size=(256, 256))

def predict_and_evaluate(data_loader):
    """
    Make predictions on the test set and evaluate the reconstruction quality
    """
    with torch.no_grad(): # Turn off gradient calculation to reduce memory usage
        ssim_scores = [] # Store the SSIM scores of all images
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)

            # forward propagatino
            reconstructed, _ = model(batch)

            # transfer to numpy array
            original_image = batch[0, 0, :, :].cpu().numpy()
            reconstructed_image = reconstructed[0, 0, :, :].cpu().numpy()

            # ensure data in [0,1]
            original_image = np.clip(original_image, 0, 1)
            reconstructed_image = np.clip(reconstructed_image, 0, 1)

            # calculate SSIM
            ssim_value = ssim(original_image, reconstructed_image, data_range=1.0)
            ssim_scores.append(ssim_value)

            # visualization
            if batch_idx < 5:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(original_image, cmap='gray')
                axs[0].set_title("Original Image")
                axs[0].axis('off')

                axs[1].imshow(reconstructed_image, cmap='gray')
                axs[1].set_title(f"Reconstructed Image\nSSIM: {ssim_value:.4f}")
                axs[1].axis('off')

                plt.show()

        # print SSIM
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        print(f"Average SSIM: {avg_ssim:.4f}")

# inference and visualization function
if __name__ == "__main__":
    predict_and_evaluate(test_loader)