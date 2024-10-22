import torch
import matplotlib.pyplot as plt
from itertools import cycle
from dataset import load_test_data
from modules import VQVAE
import utils

# Initialise the device
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def generate_samples(data_loader, model, output_loc, ssim_score=None, image_loc=None, epoch=-1):
    """
        Generates a 2x8 grid of images from a VQVAE where the top row contains the original images and the bottom row
        the reconstructed images. This function is either called for a specific epoch or the final model, as such the
        model can simply be passed through as is or loaded from the final model checkpoint. While the images have only
        one channel and are thus greyscale, they greyscale colourmap is not used as the colour provides better contrast.

        Input:
            data_loader: the DataLoader that contains the images we want to generate samples from
            model: the VQVAE model
            epoch: the epoch number, if -1 then the final model is loaded
    """
    print("Generating")
    # If epoch is not specified, then the final model is loaded in
    if epoch == -1:
        model.load_state_dict(torch.load(output_loc + 'final_vqvae.pt'))
    model.eval()  # Set model to evaluation model so it does not train while we generate
    data_loader_iter = cycle(data_loader)  # Allows infiite cycling if not enough data is passed through
    ims = next(data_loader_iter)  # Get a batch of samples from the cycle iterator
    
    # Ensure there are 16 images if the batch size is less than 16
    while ims.shape[0] < 16:
        next_ims = next(data_loader_iter)
        ims = torch.cat((ims, next_ims), dim=0)
        
    ims = ims[:16].float().unsqueeze(1).to(device)  # Ensures correct number of images are correct shape for model

    generated_ims, _, _, _ = model(ims)  # We only want the image

    # Define a 2x8 grid for images
    fig, axes = plt.subplots(2, 8, figsize=(16, 8))
    axes = axes.flatten()
    if ssim_score is not None:
        fig.suptitle(f'Generated Images with SSIM Score of {ssim_score:.5f}', fontsize=20)

    # Plot 8 original images in the first row
    for i in range(8):
        img = ims[i, :, :].squeeze().cpu().numpy()  # Remove the extra dimension
        axes[i].imshow(img)
        axes[i].axis("off")

    # Plot the 8 corresponding reconstructed images in the second row
    for i in range(8):
        img_tilde = generated_ims[i, :, :].squeeze().detach().cpu().numpy()  # Detach from computation graph
        axes[i + 8].imshow(img_tilde)
        axes[i + 8].axis("off")
        
    # Handle axes titling
    for i in [0, 8]:
        axes[i].axis("on")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
    axes[0].set_ylabel("Original", fontsize=20)
    axes[8].set_ylabel("Reconstructed", fontsize=20)

    plt.tight_layout()

    # Determine where to save the image based on what purpose it was called for
    if epoch == -1:
        plt.savefig(output_loc + f'final_reconstruction.png')
    else:
        plt.savefig(image_loc + f'epoch{epoch}.png')
    plt.close()


if __name__ == "__main__":
    OUTPUT_LOCATION = "./outputs/"
    utils.folder_check(output_loc=OUTPUT_LOCATION)
    test_loader = load_test_data()
    model = VQVAE(128, 32, 5, 512, 64)
    generate_samples(test_loader, model, OUTPUT_LOCATION)
