from __future__ import print_function
#%matplotlib inline
import os
import torch
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision.utils import save_image



import modules
import utils
import dataset
import train

###################################################
# Loss visualisation and Generated images plotting

def plot_loss(G_Loss, D_Loss):
    """Plots Graphs of Discriminator and Generator Loss over iterations"""
    plt.figure(figsize=(10,5))
    plt.title("Generator Loss During Training")
    plt.plot(G_Loss, label="G", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gen_loss.png')

    plt.figure(figsize=(10,5))
    plt.title("Discriminator Loss During Training")
    plt.plot(D_Loss, label="D", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('disc_loss.png')


def generate_examples(gen, mapping_network, epoch, device):
    """Generates images of brains using trained StyleGAN2 model"""
    n = 10
    for i in range(n):
        with torch.no_grad():
            w = utils.get_w(1, mapping_network, device)
            noise = utils.get_noise(1, device)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples_{utils.save}'):
                os.makedirs(f'saved_examples_{utils.save}')
            save_image(img*0.5+0.5, f"saved_examples_{utils.save}/epoch{epoch}_img_{i}.png")
