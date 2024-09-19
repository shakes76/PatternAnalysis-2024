from torchvision import transforms
from dataset import get_dataloader
from modules import StableDiffusion
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch, wandb, os, io
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils import generate_images, calculate_gradient_norm, get_noise_schedule, add_noise
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

image_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),    
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST-specific
])

print("Loading data...")
# os.chdir('recognition/S4696417-Stable-Diffusion-ADNI')
# train_loader, val_loader = get_dataloader('data/train/AD', batch_size=8, transform=image_transform)

# IMport MNIST dataset
train_set = MNIST(root='./data', train=True, download=True, transform=image_transform)
test_set = MNIST(root='./data', train=False, download=True, transform=image_transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=2)

# Settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

lr = 0.0001
epochs = 50

print("Loading model...")
model = StableDiffusion(
    in_channels=1,
    out_channels=1,
    model_channels=128,
    num_res_blocks=2,
    attention_resolutions=[16,8],
    channel_mult=[1, 2, 4, 8],
    num_heads=8
)
model = model.to(device)

criterion = nn.MSELoss()
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
noise_schedule = get_noise_schedule().to(device)

# initialise wandb
wandb.init(
    project="Stable-Diffusion-ADNI", 
    entity="s1lentcs-uq",
    config={
        "learning rate": lr,
        "epochs": epochs,
        "optimizer": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__,
        "loss": type(criterion).__name__,
        "scaler": type(scaler).__name__,
        "name": "SD-MNIST",
    }
)


print("Training model...")
model.train()
for epoch in range(epochs):
    train_loss, val_loss = 0, 0
    train_psnr, val_psnr = 0, 0
    train_ssim, val_ssim = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for i, batch in enumerate(loop):

        # for MNIST
        images, _ = batch
        images = images.to(device)
        timestamps = torch.randint(0, 1000, (images.size(0),), device=device)
        noisy_images = add_noise(images, timestamps, noise_schedule)
       
        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(noisy_images, timestamps)
            loss = criterion(outputs, images)

        scaler.scale(loss).backward()
        grad_norm = calculate_gradient_norm(model)
        scaler.step(optimizer)
        scaler.update()

        # Get PSNR and SSIM metrics
        with torch.no_grad():
            # Convert outputs to float32 for metric calculation
            outputs_float = outputs.float()
            psnr = peak_signal_noise_ratio(outputs_float, images)
            ssim = structural_similarity_index_measure(outputs_float, images)


        train_loss += loss.item()
        train_psnr += psnr.item()
        train_ssim += ssim.item()

        # Log metrics
        wandb.log({
            'train_loss': loss.item(),
            'train_psnr': psnr.item(),
            'train_ssim': ssim.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_norm': grad_norm
        })

        # update progress bar
        loop.set_postfix(loss=loss.item(), psnr=psnr.item(), ssim=ssim.item())


    scheduler.step()

    # validation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # for MNIST
            images, _ = batch
            timestamps = torch.randint(0, 1000, (images.size(0),), device=device).long()

            #images, timestamps = batch
            images = images.to(device)
            timestamps = timestamps.to(device)

            # Mixed precision training
            with autocast():
                outputs = model(images, timestamps)
                loss = criterion(outputs, images)

            # Calculate PSNR and SSIM
            outputs_float = outputs.float()
            psnr = peak_signal_noise_ratio(outputs_float, images)
            sam = structural_similarity_index_measure(outputs_float, images)

            val_loss += loss.item()
            val_psnr += psnr.item()
            val_ssim += ssim.item()

            loop.set_postfix(loss=loss.item())

    # Log epoch-level metrics
    wandb.log({
        'epoch': epoch,
        'avg_train_loss': train_loss / len(train_loader),
        'avg_val_loss': val_loss / len(val_loader),
        'avg_train_psnr': train_psnr / len(train_loader),
        'avg_val_psnr': val_psnr / len(val_loader),
        'avg_train_ssim': train_ssim / len(train_loader),
        'avg_val_ssim': val_ssim / len(val_loader)
    })

    print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
    print(f'Train PSNR: {train_psnr/len(train_loader):.4f}, Val PSNR: {val_psnr/len(val_loader):.4f}')
    print(f'Train SSIM: {train_ssim/len(train_loader):.4f}, Val SSIM: {val_ssim/len(val_loader):.4f}')

    if (epoch + 1) % 10 == 0:
        generate_images(model, device, epoch+1)

print("Training complete")
torch.save(model.state_dict(), 'final_model.pth')
wandb.finish()





