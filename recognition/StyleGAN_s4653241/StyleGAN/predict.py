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
    plt.close()

def plot_loss_epoch(G_loss, D_loss, epoch):
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

    for i in range(n):
        with torch.no_grad():
            w     = get_w(1, mapping_net)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

