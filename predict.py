import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def generate(epoch, model, test_loader):
    """ Plots test images against generated images """
    print("---Generating Images---")
    model.eval()

    # Collect images from test images
    x = next(iter(test_loader))  
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

    plt.tight_layout()
    plt.savefig(f'./Project/Images/imgs_epoch_{epoch}.png')