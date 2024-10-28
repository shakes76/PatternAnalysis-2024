"""
Shows example usage of your trained model. Print out any results and / or provide visu-
alisations where applicable
"""

# Importing the required libraries
import matplotlib.pyplot as plt
import torch
import os
from torchvision.utils import save_image

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

def generate_examples(gen, epoch, n=100):

    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1, gen.mapping_net)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

    gen.train()
