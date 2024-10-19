"""
Shows example usage of your trained model. Print out any results and / or provide visu-
alisations where applicable
"""

# Importing the required libraries
import matplotlib.pyplot as plt


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
