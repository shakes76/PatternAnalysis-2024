import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def generate(epoch, model, dataloader, directory, ssim_score, final=False):
    """ 
    Plots test images against generated images.
    """
    print("---Generating Images---")
    
    model.eval()

    # Collect images from validation set
    x = next(iter(dataloader))  
    batch_size = len(x)
    if batch_size > 8:  # Plot at most 8 images from a batch
        batch_size = 8
    x = x[:batch_size].float().unsqueeze(1).to(device)
    
    # Generate images 
    with torch.no_grad():
        _, x_hat, _ = model(x)
    
    # Plot 2 rows of images, first is original, second is generated
    fig, axes = plt.subplots(2, batch_size, figsize=(16, 8))
    axes = axes.flatten()
    if not final:
        fig.suptitle(f"Generated Images at Epoch {epoch} (SSIM: {ssim_score:.5f})", fontsize=20)
    else:
        fig.suptitle(f"Final Model Generated Images (SSIM: {ssim_score:.5f})", fontsize=20)

    # Plot the original image batch on first row
    for i in range(batch_size):
        im = x[i, :, :].squeeze().cpu().numpy()  
        axes[i].imshow(im, cmap='gray')         
        axes[i].axis("off")

    # Plot the generated image batch on second row
    for i in range(batch_size):
        im_hat = x_hat[i, :, :].squeeze().detach().cpu().numpy()  
        axes[i + batch_size].imshow(im_hat, cmap='gray')
        axes[i + batch_size].axis("off")

    # Label each row as 'Original' or 'Generated'
    for i in [0, batch_size]:
        axes[i].axis("on")
        axes[i].set_xticks([])  
        axes[i].set_yticks([])  
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
    axes[0].set_ylabel("Original", fontsize=16)
    axes[batch_size].set_ylabel("Generated", fontsize=16)
    
    plt.tight_layout()
    if not final:
        plt.savefig(f'{directory}/imgs_epoch_{epoch}.png')
    else:
        plt.savefig(f'{directory}/final_imgs.png')
