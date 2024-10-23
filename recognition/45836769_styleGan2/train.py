"""
REFERENCES:

(1) This code was developed with assistance from the Claude AI assistant,
    created by Anthropic, PBC. Claude provided guidance on implementing
    StyleGAN2 architecture and training procedures.

    Date of assistance: 8-21/10/2024
    Claude version: Claude-3.5 Sonnet
    For more information about Claude: https://www.anthropic.com

(2) GitHub Repository: stylegan2-ada-pytorch
    URL: https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main
    Accessed on: 29/09/24 - 21/10/24
    
(3) Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). 
    Analyzing and improving the image quality of StyleGAN.
    arXiv. https://arxiv.org/abs/1912.04958

(4) Karras, T., Laine, S., & Aila, T. (2019).
    A Style-Based Generator Architecture for Generative Adversarial Networks.
    arXiv. https://arxiv.org/abs/1812.04948
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from modules import StyleGAN2Generator, StyleGAN2Discriminator
from dataset import ADNIDataset
import os
import gc
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torchvision import transforms

# Hyperparams - mostly following StyleGAN2 paper, adjusted for smaller network
z_dim = 128 # Latent dims (z: input, w: intermediate)
w_dim = 128  
num_mapping_layers = 8
mapping_dropout = 0.0
label_dim = 2  # AD and NC
num_layers = 5
ngf = 64 # Num generator features
ndf = 64 # Num discriminator features
batch_size = 8
num_epochs = 100
lr_g = 0.001
lr_d = 0.00005
beta1 = 0.5
beta2 = 0.999  # Adam betas
r1_gamma = 50.0  # R1 regularisation weight
d_reg_interval = 16  # Discrim regularisation interval
max_grad_norm_d = 1.0  # Maximum norm for gradient clipping
max_grad_norm_g = 10.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ADA parameters
ada_start_p = 0  # start ADA aug %
ada_target = 0.6  # Target ADA aug %
ada_interval = 4  # ADA update interval
ada_kimg = 500  # ADA adjustment speed (lower -> faster)

class ADAStats:
    """
    Class to keep track of Adaptive Discriminator Augmentation (ADA) statistics.
    """
    def __init__(self, start_p=0):
        self.p = start_p  # Current aug %
        self.rt = 0  # total of sign(real_score)
        self.num = 0  # Num samples
        self.avg = 0  # Avg sign(real_score)

    def update(self, real_signs):
        """Update ADA statistics with new batch of real image signs."""
        self.rt += real_signs.sum().item()
        self.num += real_signs.numel()
        self.avg = self.rt / self.num if self.num > 0 else 0

# Create results dir
for dir in ["results/AD", "results/NC", "results/UMAP", "checkpoints"]:
    os.makedirs(dir, exist_ok=True)

# Init dataset and loader
dataset = ADNIDataset(root_dir="/home/groups/comp3710/ADNI/AD_NC", split="train") # CHANGE DIR
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# Init generator and discriminator
generator = StyleGAN2Generator(z_dim, w_dim, num_mapping_layers, mapping_dropout, label_dim, num_layers, ngf).to(device)
discriminator = StyleGAN2Discriminator(image_size=(256, 256), num_channels=1, ndf=ndf, num_layers=num_layers).to(device)

# Init optimisers
g_optim = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
d_optim = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

# Loss function
# criterion = nn.BCEWithLogitsLoss()

# Init GradScaler
scaler = amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)

# Helper funcs
def requires_grad(model, flag=True):
    """Enable/disable gradients for model params"""
    for p in model.parameters():
        p.requires_grad = flag

def d_r1_loss(real_pred, real_img):
    """R1 regularisation for discriminator"""
    grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

def save_images(generator, z, labels, epoch, batch=None):
    """Save generated images, categorise AD and NC"""
    with torch.no_grad(), amp.autocast(device_type='cuda'):
        fakes = generator(z, labels)
        for i, (img, lbl) in enumerate(zip(fakes, labels)):
            label_str = "AD" if lbl == 0 else "NC"
            filename = f"results/{label_str}/fake_e{epoch+1}_" + (f"b{batch}_" if batch else "") + f"s{i+1}.png"
            save_image(img.float(), filename, normalize=True)  # Ensure float32 for save_image

def plot_losses(d_losses, g_losses):
    """Plot and save the discriminator and generator losses."""
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator", alpha=0.5)
    plt.plot(d_losses, label="Discriminator", alpha=0.5)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/loss_plot.png")
    plt.close()

def clear_cache():
    """Clear CUDA cache and run garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()
    
# ADA augmentation function
def ada_augment(images, p):
    """Apply adaptive discriminator augmentation to images."""
    if p > 0:
        augment_pipe = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=(-30, 30), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-15, 15)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([transforms.ElasticTransform(alpha=250.0, sigma=10.0)], p=0.5),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
            transforms.RandomApply([transforms.RandomAutocontrast()], p=0.5),
        ])
        mask = torch.rand(images.size(0), 1, 1, 1, device=images.device) < p
        augmented = augment_pipe(images)
        images = torch.where(mask, augmented, images)
    return images

def update_ada(ada_stats, real_pred, ada_target, ada_interval, batch_idx, ada_kimg):
    """Update ADA statistics and adjust augmentation probability."""
    ada_stats.update(real_pred.sign().to(torch.float32))
    adjust_p = None
    if batch_idx % ada_interval == 0:
        ada_r = ada_stats.avg
        if ada_r > ada_target:
            adjust_p = min(ada_stats.p + (1 / ada_kimg), 1)
        else:
            adjust_p = max(ada_stats.p - (1 / ada_kimg), 0)
        ada_stats.p = adjust_p
    return ada_stats, adjust_p
    
def d_loss_fn(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()
    return real_loss + fake_loss

def g_loss_fn(fake_pred):
    return F.softplus(-fake_pred).mean()

# Training loop
total_batches = len(dataloader)
print_interval = 50
save_interval = 5 # Every 5 epochs save and gen progress images
fixed_z = torch.randn(16, z_dim).to(device)
fixed_labels = torch.cat([torch.zeros(8), torch.ones(8)], dim=0).long().to(device)
d_losses = []
g_losses = []
ada_stats = ADAStats(ada_start_p)
d_update_freq = 1 # Update D every x G updates 

for epoch in range(num_epochs):
    clear_cache() # Free up GPU mem
    for i, (real_images, labels) in enumerate(dataloader):
        real_images, labels = real_images.to(device), labels.to(device)
        
        ### Train Generator ### (every iteration)
        requires_grad(generator, True) # Unfreeze G
        requires_grad(discriminator, False) # Freeze D
        
        with amp.autocast(device_type='cuda'):
            z = torch.randn(real_images.size(0), z_dim).to(device) # Gen noise
            fake_images = generator(z, labels) # Gen fakes
            fake_pred = discriminator(fake_images) # D classify fakes
            g_loss = g_loss_fn(fake_pred) # Non-saturating GAN loss
            g_losses.append(g_loss.item()) # Record loss
        
        g_optim.zero_grad() # Reset grads
        scaler.scale(g_loss).backward() # Backward pass
        scaler.unscale_(g_optim) # Unscale
        clip_grad_norm_(generator.parameters(), max_grad_norm_g) # Clip grads
        scaler.step(g_optim) # Update G param
        scaler.update() # Step scaler state

        ### Train Discriminator ### (every d_update_freq iterations)
        if i % d_update_freq == 0:
            # Apply ADA to real images
            real_images_aug = ada_augment(real_images, ada_stats.p)
            
            requires_grad(generator, False) # Freeze G
            requires_grad(discriminator, True) # unfreeze D
            
            with amp.autocast(device_type='cuda'):
                z = torch.randn(real_images.size(0), z_dim).to(device) # Create noise
                fake_images = generator(z, labels) # Gen fakes
                fake_output = discriminator(fake_images.detach()) #D classify fakes
                real_output = discriminator(real_images_aug) # D classify reals
                d_loss = d_loss_fn(real_output, fake_output) # non-saturated GAN loss with R1 regularisation
                d_losses.append(d_loss.item()) # Record loss
            
            # D backprop    
            d_optim.zero_grad() # Reset grads
            scaler.scale(d_loss).backward() # Backward pass with scaling
            scaler.unscale_(d_optim) # Unscale for clipping and step
            clip_grad_norm_(discriminator.parameters(), max_grad_norm_d) #Clip grads
            scaler.step(d_optim) # Update D params
            scaler.update() # Step scaler state
            
            # Update ADA stats and prob
            ada_stats, adjust_p = update_ada(ada_stats, real_output, ada_target, ada_interval, i, ada_kimg)
            
            # D R1 regularisation 
            if i % d_reg_interval == 0:
                real_images.requires_grad = True # Enable grad calc
                with amp.autocast(device_type='cuda'):
                    real_pred = discriminator(real_images)
                    r1_loss = d_r1_loss(real_pred, real_images)
                d_optim.zero_grad() # Reset grads
                scaler.scale(r1_gamma / 2 * r1_loss * d_reg_interval).backward() # Backward pass
                scaler.unscale_(d_optim) # Unscale
                clip_grad_norm_(discriminator.parameters(), max_grad_norm_d) # Clip grads
                scaler.step(d_optim) # Update d params
                scaler.update() # update scaler state
                real_images.requires_grad = False  # Reset requires_grad

        # Print losses and check for NaN
        if i % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{total_batches}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                  f"ADA_p: {ada_stats.p:.4f}")
            
        if torch.isnan(d_loss) or torch.isnan(g_loss):
            print(f"NaN loss detected at Epoch {epoch+1}, Batch {i+1}. Break.")
            break
    
    print(f"Epoch [{epoch+1}/{num_epochs}] completed")
    
    # Save checkpoints and generate images every 5 epochs
    if (epoch + 1) % save_interval == 0:
        save_images(generator, fixed_z, fixed_labels, epoch)
        torch.save({
            'gen_state_dict': generator.state_dict(),
            'discrim_state_dict': discriminator.state_dict(),
        }, f"checkpoints/stylegan2_checkpoint_epoch_{epoch+1}.pth")

        plot_losses(d_losses, g_losses)

print("Training complete!")