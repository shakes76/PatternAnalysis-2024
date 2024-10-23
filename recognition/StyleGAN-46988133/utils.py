"""
utils.py created by Matthew Lockett 46988133
"""
import os
import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib
import matplotlib.pyplot as plt
import hyperparameters as hp


def save_resolution(disc, gen, D_losses, G_losses, current_resolution, current_depth, device):
    """
    This function is called at the end of training the StyleGAN on a certain image resolution. Thus
    this function will save the most up to date images from the generator and also a loss plot. 
    Both models will also be saved for later use in inference.
    
    Param: A discriminator model.
    Param: A generator model.
    Param: D_losses: The amount of losses the discriminator has occured so far.
    Param: G_losses: The amount of losses the generator has occured so far.
    Param: current_resolution: The current resolution/image size that the gen and disc have been trained on.
    Param: current_depth: Also indicates the current resolution/image size.
    Param: device: The Cuda or CPU device that the models were trained on.
    """
    # Generate some noise for the generator to produce images
    fixed_noise = torch.randn(64, hp.LATENT_SIZE, device=device)

    # Save the last images produced at the current resolution 
    with torch.no_grad():
        fake = gen(fixed_noise, depth=current_depth).detach().cpu()
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(f"Final Images Produced at Resolution {current_resolution}x{current_resolution}")
    plt.imshow(np.transpose(vutils.make_grid(fake[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(os.path.join(hp.SAVED_OUTPUT_DIR, f"{current_resolution}x{current_resolution}_final_images.png"), pad_inches=0)
    plt.close()

    # Output a training loss plot based on all previous losses
    plt.figure(figsize=(10,5))
    plt.title(f"Generator and Discriminator Loss During Training for Resolution {current_resolution}x{current_resolution}")
    plt.plot(G_losses,label="Generator", color='blue', linewidth='2')
    plt.plot(D_losses,label="Discriminator", color='red', linewidth='2')
    plt.xlabel("Iterations")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.savefig(os.path.join(hp.SAVED_OUTPUT_DIR, f"{current_resolution}x{current_resolution}_training_loss_plot.png"), pad_inches=0)
    plt.close()

    # Save the models at the current resolution for later use in inference 
    torch.save(gen.state_dict(), os.path.join(hp.SAVED_OUTPUT_DIR, f"{current_resolution}x{current_resolution}_gen_final_model.pth"))
    torch.save(disc.state_dict(), os.path.join(hp.SAVED_OUTPUT_DIR, f"{current_resolution}x{current_resolution}_disc_final_model.pth"))