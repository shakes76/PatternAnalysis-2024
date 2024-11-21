import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torchvision.utils import make_grid

# Calculate SSIM
def calculate_ssim(original, reconstructed):
    ssim_scores = []
    for i in range(original.size(0)):
        orig = original[i].cpu().detach().numpy().squeeze()
        recon = reconstructed[i].cpu().detach().numpy().squeeze()
        ssim_score = ssim(orig, recon, data_range=recon.max() - recon.min())
        ssim_scores.append(ssim_score)
    return np.mean(ssim_scores)

# Display images
def show_img(img):
    npimg = img.numpy()
    npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

# Combined display of Original and Reconstructed Images
def show_combined(originals, reconstructions, average_ssim):
    # Combine originals and reconstructions
    combined = torch.cat([originals[:8], reconstructions[:8]], dim=0)
    
    npimg = make_grid(combined, nrow=8, normalize=True).numpy()
    npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())  # Normalize for display
    
    # Display with SSIM
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    ax.set_title(f"Originals (Top) and Reconstructions (Bottom) - Avg SSIM: {average_ssim:.3f}", fontsize=14)
    ax.axis('off')
    plt.show()

# Plot metrics over epochs
def plot_metrics(train, validation, metric_name):
    epochs = range(1, len(train) + 1)
    plt.plot(epochs, train, 'bo-', label=f'Train {metric_name}')
    plt.plot(epochs, validation, 'ro-', label=f'Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Epochs')
    plt.legend()
    plt.show()


