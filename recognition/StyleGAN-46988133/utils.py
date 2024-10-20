"""
utils.py created by Matthew Lockett 46988133
"""
import os
import torch
import torchvision.utils as vutils
import matplotlib
import matplotlib.pyplot as plt
import hyperparameters as hp


def save_resolution(disc, gen, current_resolution, current_depth, device):
    """
    This function is called at the end of training the StyleGAN on a certain image resolution. Thus
    this function will save the most up to date images from the generator and this resolution and 
    also save the models for inference later.
    
    Param: A discriminator model.
    Param: A generator model.
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

    # Save the models at the current resolution for later use in inference 
    torch.save(gen.state_dict(), os.path.join(hp.SAVED_OUTPUT_DIR, f"{current_resolution}x{current_resolution}_gen_final_model.pth"))
    torch.save(disc.state_dict(), os.path.join(hp.SAVED_OUTPUT_DIR, f"{current_resolution}x{current_resolution}_disc_final_model.pth"))