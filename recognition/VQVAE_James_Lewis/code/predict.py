from torchvision.utils import save_image
import matplotlib.pyplot as plt
from modules import VQVAE
from dataset import load_data_2D, DataLoader
import torch
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    return model


def construct_images(original_imag_list, reconstructed_data_list, section):
    """
    Construct and save the original and reconstructed images for visualization.

    @param original_imag_list: list of torch.Tensor, the original images
    @param reconstructed_data_list: list of torch.Tensor, the reconstructed images
    @param epoch: int, the current epoch
    @param section: str, the section of the dataset (e.g., 'train', 'val', 'test')
    """
    save_dir = f'reconstructed_images/{section}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Ensure both lists are of equal length
    num_images = min(len(original_imag_list), len(reconstructed_data_list), 5)  # Limit to 5 images for clarity
    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(num_images * 3, 6))

    for i in range(num_images):
        original_imag = original_imag_list[i].detach().cpu().squeeze((0, 1))
        reconstructed_data = reconstructed_data_list[i].detach().cpu().squeeze((0, 1))


        # Original Image
        axes[0, i].imshow(original_imag, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {i + 1}')

        # Reconstructed Image
        axes[1, i].imshow(reconstructed_data, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Reconstructed {i + 1}')

    plt.suptitle(f'{section.capitalize()} Images')

    # Save the image grid to a file
    save_path = os.path.join(save_dir, f'{section}_reconstructed_images.png')
    plt.savefig(save_path)
    plt.close()








def main():

    input_dim = 1
    out_dim = 128
    n_res_block = 2
    n_res_channel = 64
    stride = 2
    n_embed = 256
    embedding_dims = 128
    commitment_cost = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(input_dim,
                  out_dim,
                  n_res_block,
                  n_res_channel,
                  stride,
                  n_embed,
                  commitment_cost,
                  embedding_dims).to(device)

    model_path = "saved_models/vqvae_model.pth"
    model = load_model(model, model_path, device)

    test_image_directory = '/Users/jameslewis/PatternAnalysis-2024/recognition/VQVAE_James_Lewis/data/HipMRI_study_keras_slices_data/keras_slices_test'
    test_image_names = [os.path.join(test_image_directory, f) for f in os.listdir(test_image_directory) if f.endswith('.nii.gz')]
    test_images = load_data_2D(test_image_names, normImage=True)
    test_loader = DataLoader(test_images, batch_size=1, shuffle=False)

    ssim_scores = []
    total_test_ssim = 0
    model.eval()

    with torch.no_grad():
        for data in test_loader:

            # Convert the numpy array to a PyTorch tensor
            data = data.unsqueeze(1)  # Add channel dimension for grayscale [batch_size, 1, height, width]
            data = data.to(device)  # Move the data to the appropriate device (e.g., GPU)

            # Forward pass through the model
            reconstructed_data, _, embeddings = model(data)

            # Compute SSIM between reconstructed and original data
            ssim_score = ssim_metric(reconstructed_data, data)

            # Accumulate SSIM score
            total_test_ssim += ssim_score.item()

            ssim_scores.append({'ssim': ssim_score.item(),
                                'reconstructed': reconstructed_data,
                                'original': data})

    avg_test_ssim = total_test_ssim / len(test_loader)

    print(f"Average SSIM on test set: {avg_test_ssim}")

    # Sort the list by SSIM score
    ssim_scores.sort(key=lambda x: x['ssim'])

    # Save 4 worst SSIM score images (side by side)
    worst_images = ssim_scores[:4]
    construct_images([x['original'] for x in worst_images],
                             [x['reconstructed'] for x in worst_images],
                             'worst')

    # Save 4 best SSIM score images (side by side)
    best_images = ssim_scores[-4:]
    construct_images([x['original'] for x in best_images],
                             [x['reconstructed'] for x in best_images],
                             'best')

    print("Images saved for the 4 best and 4 worst SSIM scores.")

if __name__ == "__main__":
    main()