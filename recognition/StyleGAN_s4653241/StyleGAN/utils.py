
'''Utils for the project'''

import torch

def save_model(epoch,G,D,optimizer_G,optimizer_D, gen_loss, disc_loss,path):
    checkpoint = {
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all(),
        'gen_loss': gen_loss,
        'disc_loss': disc_loss,

    }
    torch.save(checkpoint, path)
