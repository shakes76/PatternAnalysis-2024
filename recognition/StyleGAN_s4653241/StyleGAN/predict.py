import matplotlib.pyplot as plt
import torch
import os
from torchvision.utils import save_image

from torch.utils.data import Dataset
import random


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


    gen.train()

'''
Generate samples from the dataset
'''
def generate_samples(Dataset, n=5):

    indices = random.sample(range(len(Dataset)), n)
    # Display the images
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i, index in enumerate(indices):
        raw_img = Dataset.get_raw_image(index)
        axes[i].imshow(raw_img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.show()

 
