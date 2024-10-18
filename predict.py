import torch
import matplotlib.pyplot as plt
from itertools import cycle

# Initialise the device
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def generate_samples(data_loader, model, epoch=-1):
    """
        Generates a 4x8 grid of images from a VQVAE where the top 2 rows are the original images and the bottom 2 rows
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
        model.load_state_dict(torch.load(f'final_vqvae.pt'))
    model.eval()  # Set model to evaluation model so it does not train while we generate
    data_loader_iter = cycle(data_loader)  # Allows infiite cycling if not enough data is passed through
    ims = next(data_loader_iter)  # Get a batch of samples from the cycle iterator
    ims = ims[:16].float().unsqueeze(1).to(device)  # Ensures correct number of images are correct shape for model

    generated_ims, _, _, _ = model(ims)  # We only want the image

    # Define a 4x8 grid for images (2 rows for inputs, 2 rows for outputs)
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()

    # Plot the 16 original images in the first two rows
    for i in range(16):
        img = ims[i, :, :].squeeze().cpu().numpy()  # Remove the extra dimension
        # axes[i].imshow(img, cmap='gray')
        axes[i].imshow(img)
        axes[i].axis("off")

    # Plot the 16 corresponding reconstructed images in the last two rows
    for i in range(16):
        img_tilde = generated_ims[i, :, :].squeeze().detach().cpu().numpy()  # Detach from computation graph
        # axes[i+16].imshow(img_tilde, cmap='gray')
        axes[i + 16].imshow(img_tilde)
        axes[i + 16].axis("off")

    plt.tight_layout()
    plt.show()

    # Determine where to save the image based on what purpose it was called for
    if epoch == -1:
        plt.savefig(f'./outputs/final_reconstruction.png')
    else:
        plt.savefig(f'./epoch_reconstructions/epoch{epoch}.png')
    plt.close()
