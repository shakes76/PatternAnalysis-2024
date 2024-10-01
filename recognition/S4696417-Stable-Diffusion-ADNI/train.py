from torchvision import transforms
from dataset import get_dataloader
import torch, wandb, os
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from utils import get_linear_schedule_with_warmup, visualize_denoising_process
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from modules import StableDiffusion, UNet, CosineAnnealingWarmupScheduler, NoiseScheduler_Fast_DDPM, Lookahead, PerceptualLoss

# SETUP - Must Match Image Size of VAE
IMAGE_SIZE = 256
BATCH_SIZE = 8 # will affect training performance
method = 'Local'

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),    
    transforms.Normalize((0.5,), (0.5,)),
])

print("Loading data...")
if method == 'Local':
    os.chdir('recognition/S4696417-Stable-Diffusion-ADNI')
    train_loader, val_loader = get_dataloader('data/train/AD', batch_size=BATCH_SIZE, transform=image_transform)
elif method == 'Slurm':
    train_loader, val_loader = get_dataloader('/home/groups/comp3710/ADNI/AD_NC/train/AD', batch_size=BATCH_SIZE, transform=image_transform)

# Settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

print("Loading model...")
vae = torch.load(f'checkpoints/VAE/ADNI-vae_e100_b8_im{IMAGE_SIZE}_l16.pt')
vae.eval() 
for param in vae.parameters():
    param.requires_grad = False 

unet = UNet(in_channels=vae.latent_dim, hidden_dims=[64, 128, 256, 512], time_emb_dim=256)
noise_scheduler = NoiseScheduler_Fast_DDPM(num_timesteps=100).to(device)
model = StableDiffusion(unet, vae, noise_scheduler, image_size=IMAGE_SIZE).to(device)


#criterion = nn.MSELoss()
perceptual_weight = 0.2
perceptual_loss = PerceptualLoss(vae).to(device)
mse_loss = nn.MSELoss()
scaler = GradScaler()

lr = 1e-4
epochs = 100
steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * epochs
warmup_steps = int(0.1 * total_steps)

base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

#scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_steps, total_steps)

# initialise wandb
wandb.init(
    project="Stable-Diffusion-ADNI", 
    entity="s1lentcs-uq",
    config={
        "learning rate": lr,
        "epochs": epochs,
        "optimizer": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__,
        "perceptual loss": type(perceptual_loss).__name__,
        "mse loss": type(mse_loss).__name__,
        "scaler": type(scaler).__name__,
        "name": "SD-ADNI - VAE and Unet",
        "image size": IMAGE_SIZE,
        "batch size": BATCH_SIZE
    })

print("Training model...")
for epoch in range(epochs):
    model.train()
    train_loss, val_loss = 0, 0
    train_psnr, val_psnr = 0, 0
    train_ssim, val_ssim = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for i, batch in enumerate(loop):

        images, _ = batch # retrieve clean image batch
        images = images.to(device)

        # Encode images to latent space
        with torch.no_grad():
            mu, logvar = model.encode(images)
            latents = model.sample_latent(mu, logvar)
        
        # Sample noise and timesteps
        timesteps = torch.randint(0, noise_scheduler.num_timesteps, (latents.size(0),), device=device)
        noise = torch.randn_like(latents) 

        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        with torch.no_grad():
            denoised_images = model.vae.decode(noise_scheduler.step(model.unet, noisy_latents, timesteps))
            ssim = structural_similarity_index_measure(denoised_images, images)
            psnr = peak_signal_noise_ratio(denoised_images, images)

        # Train UNet
        optimizer.zero_grad()
        with autocast('cuda'):
            predicted_noise = model.predict_noise(noisy_latents, timesteps)
            mse = mse_loss(predicted_noise, noise)

        perceptual = perceptual_loss(denoised_images, images)
        loss = mse + perceptual_weight * perceptual

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
           
        # Update metrics
        train_loss += loss.item()
        train_psnr += psnr.item()
        train_ssim += ssim.item()

        wandb.log({
            'train_psnr': psnr.item(),
            'train_ssim': ssim.item(),
            'train_loss': loss.item(),
            'train_mse': mse.item(),
            'train_perceptual': perceptual.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        # Update progress bar
        loop.set_postfix(loss=loss.item(), mse=mse.item(), perceptual=perceptual.item())

    # Compute average metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_train_psnr = train_psnr / len(train_loader)
    avg_train_ssim = train_ssim / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_psnr = 0
    val_ssim = 0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)

            with torch.no_grad():
                mu, logvar = model.encode(images)
                latents = model.sample_latent(mu, logvar)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (images.size(0),), device=device)
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            with torch.no_grad():
                denoised_images = model.vae.decode(noise_scheduler.step(model.unet, noisy_latents, timesteps))
                ssim = structural_similarity_index_measure(denoised_images, images)
                psnr = peak_signal_noise_ratio(denoised_images, images)

            with autocast('cuda'):
                predicted_noise = model.predict_noise(noisy_latents, timesteps)
                mse = mse_loss(predicted_noise, noise)
                
            perceptual = perceptual_loss(denoised_images, images)
            loss = mse + perceptual_weight * perceptual

            val_loss += loss.item()
            val_psnr += psnr.item()
            val_ssim += ssim.item()
    
    # Compute average validation metrics
    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_loader)
    avg_val_ssim = val_ssim / len(val_loader)

    # Log epoch-level metrics
    wandb.log({
        'epoch': epoch,
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'avg_train_psnr': avg_train_psnr,
        'avg_val_psnr': avg_val_psnr,
        'avg_train_ssim': avg_train_ssim,
        'avg_val_ssim': avg_val_ssim
    })

    print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    print(f'Train PSNR: {avg_train_psnr:.4f}, Val PSNR: {avg_val_psnr:.4f}')
    print(f'Train SSIM: {avg_train_ssim:.4f}, Val SSIM: {avg_val_ssim:.4f}')

    # Generate and log sample images
    if (epoch) % 2 == 0: 
        sample_images = model.sample(BATCH_SIZE, device=device)
        ssim = structural_similarity_index_measure(sample_images, images)
        psnr = peak_signal_noise_ratio(sample_images, images)
        wandb.log({
            'Generated SSIM': ssim,
            'Generated PSNR': psnr
        })

    if epoch == epochs: 
        sample_images = model.sample(num_images=8, device=device)
    

print("Training complete")
path = os.path.join(os.getcwd(), f'checkpoints/Diffusion/ADNI_diffusion_e{epoch+1}_im{IMAGE_SIZE}.pt')
torch.save(model, path)
wandb.finish()





