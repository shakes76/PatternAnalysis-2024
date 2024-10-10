from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Variables to store total sum and total squared sum

class GFNetDataloader():
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self._mean = 0.0
        self._std = 0.0
        self._total_images = 0
        self.train_loader = None
        self.test_loader = None

    def load(self):
        transform_complete = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_complete)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        for images, _ in loader:
            # Compute the mean and std per batch
            batch_samples = images.size(0)  # Get the number of images in the batch
            images = images.view(batch_samples, images.size(1), -1)  # Flatten the image
            self._mean += images.mean(2).sum(0)  # Sum mean across batch
            self._std += images.std(2).sum(0)    # Sum std across batch
            self._total_images += batch_samples

        # Final mean and std calculations
        self._mean /= self._total_images
        self._std /= self._total_images

        # TODO: Add more transformations into this
        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)  # Use computed mean and std
        ])


        # Load MNIST dataset
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=_transform)

        # Create data loaders for batching
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_data(self):
        return self.train_loader, self.test_loader

    def get_meta(self):
        return {"total_images": self._total_images, "mean": self._mean, "std": self._std} 
