# Importing the required libraries
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

def generate_examples(gen, mapping_net,epoch, n=5):

    gen.eval()
    alpha = 1.0
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
