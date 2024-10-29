import matplotlib.pyplot as plt
import torch
import os
from torchvision.utils import save_image
from PIL import Image

from modules import *
from config import *
from utils import *

def plot_loss(G_loss,D_loss):
    """
    Plots and saves the Generator and Discriminator losses separately as two images.

    Args:
        G_loss (list of float): Generator loss values over iterations.
        D_loss (list of float): Discriminator loss values over iterations.
    """

    plt.figure(figsize=(10,5))
    plt.title("Generator Loss During Training")
    plt.plot(G_loss, label="G", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gen_loss.png')

    plt.figure(figsize=(10,5))
    plt.title("Discriminator Loss During Training")
    plt.plot(D_loss, label="D", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('disc_loss.png')
    plt.close()

def plot_loss_epoch(G_loss, D_loss, epoch):
    """
    Plots and saves Generator and Discriminator losses for each epoch on the same plot.

    Args:
        G_loss (list of float): Generator loss values per epoch.
        D_loss (list of float): Discriminator loss values per epoch.
        epoch (int): Current epoch number, used in the plot title and filename.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(G_loss, label="Generator Loss", color="blue")
    plt.plot(D_loss, label="Discriminator Loss", color="red")
    plt.title(f"Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'loss_epoch_{epoch}.png')
    plt.close()

def generate_examples(gen, mapping_net,epoch, n=5):
    """
    Generates and saves example images from the Generator model at a specific training epoch.

    Args:
        gen (nn.Module): Generator model.
        mapping_net (nn.Module): Mapping network for generating latent vectors.
        epoch (int): Current epoch number, used to name the output directory.
        n (int): Number of example images to generate.
    """
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1, mapping_net)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")



    save_dir = "/home/Student/s4653241/StyleGAN2/recognition/StyleGAN_s4653241/StyleGAN/saved_examples"
    for i in range(5):
        images= next(data_iter)  # Get a batch of images and labels

        
        # Save the first few images in each batch
        for j in range(len(images)):
            # Convert the image tensor to a numpy array and scale properly
            img = images[j].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) and numpy

            if img.max() <= 1.0:  # If normalized between 0-1, scale to 0-255
                img = (img * 255).astype(np.uint8)
            
            # Ensure it has three channels; convert grayscale to RGB
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)  # Convert single channel to RGB

            # Create PIL image
            pil_img = Image.fromarray(img)

            # Save the image
            pil_img.save(os.path.join(save_dir, f"batch_{i+1}_img_{j+1}.png"))

        print(f"Images saved to {save_dir}")