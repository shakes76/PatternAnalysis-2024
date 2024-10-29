import torch
import warnings
import config
from config import *



def save_model(epoch,G,D,optimizer_G,optimizer_D,optim_mapping, gen_loss, disc_loss):
    """
    Saves the model checkpoint including the state of models, optimizers, and training parameters.

    Parameters:
    - epoch (int): The current epoch number.
    - G (nn.Module): Generator model.
    - D (nn.Module): Discriminator model.
    - optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
    - optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
    - optim_mapping (torch.optim.Optimizer): Optimizer for the mapping network.
    - gen_loss (float): Generator loss.
    - disc_loss (float): Discriminator loss.
    """
    checkpoint = {
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        "optim_mapping_state": optim_mapping.state_dict(),
        'random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all(), # Save the random state of the GPU
        'gen_loss': gen_loss,
        'disc_loss': disc_loss,

    }
    path = config.save_path + f'checkpoint_StyleGAN2_Epoch{epoch+1}.pth'
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch+1}.", flush =True)

def load_checkpoint(path, G, D, optimizer_G, optimizer_D,optim_mapping):
    """
    Loads a saved checkpoint to resume training.

    Parameters:
    - path (str): Path to the checkpoint file.
    - G (nn.Module): Generator model to load state into.
    - D (nn.Module): Discriminator model to load state into.
    - optimizer_G (torch.optim.Optimizer): Generator optimizer.
    - optimizer_D (torch.optim.Optimizer): Discriminator optimizer.
    - optim_mapping (torch.optim.Optimizer): Optimizer for the mapping network.

    Returns:
    - epoch (int): Epoch to resume from.
    - gen_loss (float): Saved generator loss.
    - disc_loss (float): Saved discriminator loss.
    """
    checkpoint = torch.load(path)
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    optim_mapping.load_state_dict(checkpoint['optim_mapping_state'])
    epoch = checkpoint['epoch']
    gen_loss = checkpoint['gen_loss']
    disc_loss = checkpoint['disc_loss']
    torch.set_rng_state(checkpoint['random_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

    return epoch, gen_loss, disc_loss



def devicer():
    """
    Selects the best available device (CUDA, MPS, or CPU) for computation.

    Returns:
    - device (torch.device): The computation device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # print(f'Using device: {device}', flush=True)
    return device

def get_w(batch_size, mapping_network,device=devicer()):
    """
    Generates the latent vector `w` for StyleGAN2, expanded to match the required shape.

    Parameters:
    - batch_size (int): Number of samples in the batch.
    - mapping_network (nn.Module): Network that maps latent `z` to latent `w`.
    - device (torch.device): Device for computation.

    Returns:
    - w (torch.Tensor): Expanded `w` tensor for generator blocks.
    """
    z = torch.randn(batch_size, W_DIM).to(device)
    w = mapping_network(z)
    # Expand w from the generator blocks
    return w[None,:,:].expand(config.log_resolution, -1, -1)

def get_noise(batch_size,device=devicer()):
    """
    Generates a list of noise tensors of increasing resolution for each generator block in StyleGAN2.

    Parameters:
    - batch_size (int): Number of samples in the batch.
    - device (torch.device): Device for computation.

    Returns:
    - noise (list of torch.Tensor): List of noise tensors for generator layers.
    """
    noise = []
    #noise res starts from 4x4
    resolution = 4

    # For each gen block
    for i in range(config.log_resolution):
        # First block uses 3x3 conv
        if i == 0:
            n1 = None
        # For rest of conv layer
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device= device)
        n2 = torch.randn(batch_size, 1, resolution, resolution,device= device)

        # add the noise tensors to the lsit
        noise.append((n1, n2))
        # subsequent block has 2x2 res
        resolution *= 2

    return noise

def gradient_penalty(critic, real, fake,device=devicer()):
    """
    Computes the gradient penalty for WGAN-GP, helping improve discriminator stability.

    Parameters:
    - critic (nn.Module): Discriminator or critic model.
    - real (torch.Tensor): Real images.
    - fake (torch.Tensor): Generated (fake) images.
    - device (torch.device): Device for computation.

    Returns:
    - gradient_penalty (torch.Tensor): Computed gradient penalty to add to the loss.
    """
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # Reshape gradients to calculate the norm
    gradient = gradient.view(gradient.shape[0], -1)
    # Calculate the norm and then the loss
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
