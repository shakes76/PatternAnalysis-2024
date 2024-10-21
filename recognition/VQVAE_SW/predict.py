import torch
import matplotlib.pyplot as plt
from modules import VQVAE  # import VQVAE
from dataset import get_data_loader  # import data loader
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/content/data/HipMRI_study_keras_slices_data'
batch_size = 1

model = VQVAE(in_channels=1, hidden_channels=128, num_embeddings=512, embedding_dim=64).to(device)
model.load_state_dict(torch.load('vqvae_hipmri.pth'))
model.eval()

test_loader = get_data_loader(data_dir, subset='test', batch_size=batch_size, shuffle=False, normImage=True)

def predict_and_evaluate(data_loader):
    with torch.no_grad():
        ssim_scores = []
        for batch in data_loader:
            batch = batch.to(device)

            # forward
            reconstructed, _ = model(batch)

            # transfer to numpy array
            original_image = batch[0, 0, :, :].cpu().numpy()
            reconstructed_image = reconstructed[0, 0, :, :].cpu().numpy()

            # calculate SSIM
            ssim_value = ssim(original_image, reconstructed_image, data_range=reconstructed_image.max() - reconstructed_image.min())
            ssim_scores.append(ssim_value)

            # visualization
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(original_image, cmap='gray')
            axs[0].set_title("Original Image")
            axs[0].axis('off')

            axs[1].imshow(reconstructed_image, cmap='gray')
            axs[1].set_title(f"Reconstructed Image\nSSIM: {ssim_value:.4f}")
            axs[1].axis('off')

            plt.show()

            # only represent one batch
            break

        # print SSIM
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        print(f"Average SSIM: {avg_ssim:.4f}")

# inferrence and visualization funciton
if __name__ == "__main__":
    predict_and_evaluate(test_loader)