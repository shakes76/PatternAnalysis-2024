import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from adni_dataset import ADNIDataset

def calculate_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
    sum_of_means = 0.
    sum_of_stds = 0.
    total_images = 0

    for batch_images, _ in loader:
        batch_size = batch_images.size(0) # Num images
        
        # Reshape (batch_size, n_channels, height * width)
        flattened_images = batch_images.view(batch_size, batch_images.size(1), -1)
        
        # Calc mean across height and width - sum over batch
        batch_means = flattened_images.mean(dim=2)
        sum_of_means += batch_means.sum(dim=0)
        
        # Calc std across height and width - sum over batch
        batch_stds = flattened_images.std(dim=2)
        sum_of_stds += batch_stds.sum(dim=0)
        
        total_images += batch_size

    mean = sum_of_means / total_images
    std = sum_of_stds / total_images

    return mean, std

# Create default dataset
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 240)),
    transforms.ToTensor(),
])

dataset = ADNIDataset(root_dir='/Users/hamishmacintosh/Uni Work/COMP3710/AD_NC', split='train', transform=transform)

# Calculate mean and stddev
mean, std = calculate_mean_std(dataset)

print(f"Dataset mean: {mean.item():.4f}")
print(f"Dataset std: {std.item():.4f}")