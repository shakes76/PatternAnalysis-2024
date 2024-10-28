from torchvision.datasets import ImageFolder
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
Initial preprocessing of data
"""

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=128, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std

class AutoCropBlack:
    def __init__(self, threshold=10):
        """
        cropping dfunction for getting rid of black border around mri data
        """
        self.threshold = threshold

    def __call__(self, img):
        # Convert to grayscale to find non-black pixels
        gray = img.convert('L')
        gray_np = np.array(gray)

        # Determine the data type and max pixel value
        max_pixel_value = np.iinfo(gray_np.dtype).max if np.issubdtype(gray_np.dtype, np.integer) else 1.0

        threshold_value = self.threshold

        # Create mask of pixels greater than threshold
        mask = gray_np > threshold_value

        # Check if the mask is empty (no non-black pixels found)
        if not np.any(mask):
            return img  # Return original image if its completely black or below threshold

        # Get coordinates of non-black pixels
        coords = np.argwhere(mask)

        # Get bounding box of non-black pixels
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        # Crop the image to the bounding box
        cropped_img = img.crop((x0, y0, x1, y1))

        return cropped_img


def process(colab=False, test=False, show_samples=False):
    preprocess = transforms.Compose([
        AutoCropBlack(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    if not test:
        if not colab:
            dataset = ImageFolder(root='C:\\Users\\User\\Desktop\\3710Aux\\ADNI_AD_NC_2D\\AD_NC\\train',
                                transform=preprocess)
        else:
            dataset = ImageFolder(root='/content/drive/MyDrive/ADNI/AD_NC/train', transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
    else:
        if not colab:
            dataset = ImageFolder(root='C:\\Users\\User\\Desktop\\3710Aux\\ADNI_AD_NC_2D\\AD_NC\\test',
                                transform=preprocess)
        else:
            dataset = ImageFolder(root='/content/drive/MyDrive/ADNI/AD_NC/test', transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    return dataset


if __name__ == "__main__":
    process(colab=False, test=False, show_samples=True)

