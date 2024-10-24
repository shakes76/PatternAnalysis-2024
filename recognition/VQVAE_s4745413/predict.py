import torch
from torchvision import transforms
from dataset import NiftiDataset
import os
import numpy as np
from tqdm import tqdm
from PIL import Image  
from pytorch_msssim import ssim
from modules import VQVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hipmri_train_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_train"
hipmri_val_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_validate"
hipmri_test_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/keras_slices_test"
save_dir = "PatternAnalysis-2024/recognition/VQVAE_s4745413/codebook_images2"
model_checkpoint = 'PatternAnalysis-2024/recognition/VQVAE_s4745413/vqvae_model.pth'

base_dir = os.getcwd()
hipmri_train_dir = os.path.join(base_dir, hipmri_train_dir)
hipmri_val_dir = os.path.join(base_dir, hipmri_val_dir)
hipmri_test_dir = os.path.join(base_dir, hipmri_test_dir)
hipmri_save_dir = os.path.join(base_dir, save_dir)
model_checkpoint = os.path.join(base_dir, model_checkpoint)

h_dim = 64
res_h_dim = 32
n_res_layers = 2
n_emb = 512
e_dim = 64
commit_cost = 0.25

input_transf = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.Normalize((0.5,), (0.5,))
])

train_set, val_set, test_set = NiftiDataset.get_dataloaders(
    train_dir=hipmri_train_dir,
    val_dir=hipmri_val_dir,
    test_dir=hipmri_test_dir,
    batch_size=1,
    num_workers=4,
    transform=input_transf
)

model = VQVAE(h_dim, res_h_dim, n_res_layers, n_emb, e_dim, commit_cost).to(device)
checkpoint = torch.load(model_checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
os.makedirs(save_dir, exist_ok=True)

ssim_scores = []
ssim_threshold = 0.6

def save_images(tensor_images, save_dir, prefix='recon', num_samples=10):
    # Make sure to clamp the images to the range [0, 1]
    tensor_images = torch.clamp(tensor_images, 0, 1)
    
    # Unnormalize and save each image
    for i in range(num_samples):
        img_tensor = tensor_images[i].cpu()
        
        img_pil = transforms.ToPILImage()(img_tensor)
        
        # Convert to PNG
        img_pil.save(os.path.join(save_dir, f"{prefix}_image_{i}.png"))

# SSIM Calculation
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_set), total=len(test_set)):
        batch = batch.to(device)

        # Forward pass 
        embed_loss, recon_batch, perplexity = model(batch)

        recon_batch = recon_batch * 0.5 + 0.5
        batch = batch * 0.5 + 0.5  # Denormalize the original image 

        ssim_score = ssim(batch, recon_batch, data_range=1.0).item()
        ssim_scores.append(ssim_score)

# For saving the actual reconstructions
with torch.no_grad():
    for batch_idx, batch in enumerate(test_set):
        if batch_idx >= 20: 
            break
        batch = batch.to(device)

        # Pass the batch through the model to get reconstructions
        _, x_hat, _ = model(batch)

        save_images(batch, save_dir, prefix=f'original_{batch_idx}', num_samples=1)
        save_images(x_hat, save_dir, prefix=f'reconstructed_{batch_idx}', num_samples=1)

ssim_mean = np.mean(ssim_scores)
ssim_min = np.min(ssim_scores)
ssim_max = np.max(ssim_scores)
ssim_above_threshold = np.sum(np.array(ssim_scores) >= ssim_threshold)
total_images = len(ssim_scores)
percentage_above_threshold = (ssim_above_threshold / total_images) * 100

print(f"\nSSIM mean: {ssim_mean:.4f}")
print(f"Number of images with SSIM >= {ssim_threshold}: {ssim_above_threshold}, {percentage_above_threshold:.2f}%.")

