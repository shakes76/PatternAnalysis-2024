# “dataset.py" containing the data loader for loading and preprocessing your data
# we need to load the nmist datasetf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Variables to store total sum and total squared sum
_mean = 0.0
_std = 0.0
_total_images = 0

transform_complete = transforms.Compose([
    transforms.ToTensor(),
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_complete)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

for images, _ in loader:
    # Compute the mean and std per batch
    batch_samples = images.size(0)  # Get the number of images in the batch
    images = images.view(batch_samples, images.size(1), -1)  # Flatten the image
    _mean += images.mean(2).sum(0)  # Sum mean across batch
    _std += images.std(2).sum(0)    # Sum std across batch
    _total_images += batch_samples

# Final mean and std calculations
_mean /= _total_images
_std /= _total_images

# TODO: Add more transformations into this
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_mean, std=_std)  # Use computed mean and std
])


# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=_transform)

# Create data loaders for batching
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def get_data():
    return train_loader, test_loader

def get_meta():
    return {"total_images": _total_images, "mean": _mean, "std": _std} 
