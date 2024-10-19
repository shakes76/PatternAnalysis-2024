
'''Utils for the project'''

import torch
import config

def save_model(epoch,G,D,optimizer_G,optimizer_D, gen_loss, disc_loss,path = config.path + 'checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all(), # Save the random state of the GPU
        'gen_loss': gen_loss,
        'disc_loss': disc_loss,

    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch}.", flush =True)

def load_checkpoint(path, G, D, optimizer_G, optimizer_D):
    checkpoint = torch.load(path)
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    epoch = checkpoint['epoch']
    gen_loss = checkpoint['gen_loss']
    disc_loss = checkpoint['disc_loss']
    torch.set_rng_state(checkpoint['random_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

    return epoch, gen_loss, disc_loss

def devicer():
    # Prioritize CUDA, then MPS, and finally CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'Using device: {device}', flush=True)
    return device
