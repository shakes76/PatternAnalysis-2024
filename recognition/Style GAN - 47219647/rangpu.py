import matplotlib.pyplot as plt
from modules import *

def plot_losses(disc_losses, gen_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.plot(gen_losses, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Discriminator and Generator Losses Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

