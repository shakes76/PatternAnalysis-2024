import torch
import matplotlib.pyplot as plt
from itertools import cycle

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def generate_samples(test_loader, model, epoch=-1):
    """Generates and saves reconstructed samples for a given epoch."""
    print("Generating")
    if epoch == -1:
        model.load_state_dict(torch.load(f'final_vqvae.pt'))
    model.eval()
    test_loader_iter = cycle(test_loader)  # Initialize cycling iterator here
    x = next(test_loader_iter)  # Get a batch of samples using the cycling iterator
    x = x[:32].float().unsqueeze(1).to(device)

    x_tilde, _, _, _ = model(x)

    # Define a 4x8 grid for images (2 rows for inputs, 2 rows for outputs)
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()

    # Plot 16 input images in the first two rows
    for i in range(16):
        img = x[i, :, :].squeeze().cpu().numpy()  # Remove the extra dimension
        # axes[i].imshow(img, cmap='gray')
        axes[i].imshow(img)
        axes[i].axis("off")

    # Plot 16 corresponding output images in the next two rows
    for i in range(16):
        img_tilde = x_tilde[i, :, :].squeeze().detach().cpu().numpy()  # Detach from computation graph
        # axes[i+16].imshow(img_tilde, cmap='gray')
        axes[i + 16].imshow(img_tilde)
        axes[i + 16].axis("off")

    plt.tight_layout()
    plt.show()

    if epoch == -1:
        plt.savefig(f'./outputs/final_image.png')
    else:
        plt.savefig(f'./epoch_reconstructions/epoch{epoch}.png')
    plt.close()