from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.optim.lr_scheduler import LambdaLR 
import wandb, os
import torch
import numpy as np
from torchvision import transforms
from sklearn.manifold import TSNE
from dataset import get_dataloader
from torchvision.utils import make_grid

def calculate_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

@torch.no_grad()
def visualize_denoising_process(model, noise_scheduler, num_inference_steps=10, num_images=5):
    # Start from random noise
    latents = torch.randn(num_images, 8, 16, 16).to('cuda')
    
    images = []
    for i, t in enumerate(reversed(range(0, noise_scheduler.num_timesteps, noise_scheduler.num_timesteps // num_inference_steps))):
        # Create a batch of the same timestep
        timesteps = torch.full((num_images,), t, device='cuda', dtype=torch.long)
        
        # Update latents with predicted noise
        latents = noise_scheduler.step(model.unet, latents, timesteps)
        
        if i % (num_inference_steps // 10) == 0:  # Log every 10% of steps
            # Decode latents to image space
            with torch.no_grad():
                images.append(wandb.Image(model.vae.decode(latents).cpu(), caption=f"Step {i}"))
    
    wandb.log({"denoising_process": images})


def get_latent_representations(dataloader, device, vae, label):
    latent_reps = []
    labels = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            mu, logvar = vae.encode(images)
            latent = vae.sample(mu, logvar)
            print(latent.shape)
            # mu = mu.view(mu.size(0), -1).cpu().numpy()
            latent_reps.append(latent.cpu().numpy())
            labels.extend([label] * images.shape[0])
    return np.concatenate(latent_reps), np.array(labels)


def get_manifold():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.chdir('recognition/S4696417-Stable-Diffusion-ADNI')
    vae = torch.load('checkpoints/VAE/ADNI-vae_e80_b16_im128.pt').to(device)
    vae.eval()

    image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader_AD, _ = get_dataloader('data/train/AD', batch_size=16, transform=image_transform)
    train_loader_NC, _ = get_dataloader('data/train/NC', batch_size=16, transform=image_transform)
    train_latent_AD, train_labels_AD = get_latent_representations(train_loader_AD, device, vae, "AD")
    train_latent_NC, train_labels_NC = get_latent_representations(train_loader_NC, device, vae, "NC")
    train_latent = np.concatenate([train_latent_AD, train_latent_NC])
    train_labels = np.concatenate([train_labels_AD, train_labels_NC])

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(train_latent)

    wandb.init(project="Stable-Diffusion-ADNI-Manifold", name="TSNE Manifold")

    tsne_table = wandb.Table(columns=["t-SNE_1", "t-SNE_2", "Label"])
    for i in range(len(tsne_results)):
        tsne_table.add_data(tsne_results[i, 0], tsne_results[i, 1], train_labels[i])

    wandb.log({"tsne_table": tsne_table})

    # Create and log the t-SNE plot
    tsne_plot = wandb.plot.scatter(
        tsne_table, 
        x="t-SNE_1", 
        y="t-SNE_2", 
        title="t-SNE visualization of VAE latent space"
    )
    wandb.log({"tsne_plot": tsne_plot})

    print("t-SNE plot logged to wandb")

    # Finish the wandb run
    wandb.finish()


def log_threshold_visualization_to_wandb(perceptual_loss, images, num_thresholds=5):
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    
    all_images = []
    for image in images:
        image_row = [image.cpu()]
        for threshold in thresholds:
            perceptual_loss.threshold = threshold
            mask = perceptual_loss.create_brain_mask(image.unsqueeze(0))
            image_row.append(mask.cpu())
        
        all_images.extend(image_row)
    
    grid = make_grid(all_images, nrow=num_thresholds+1, normalize=True, scale_each=True)
    wandb.log({"threshold_visualization": wandb.Image(grid, caption="Threshold Visualization")})


