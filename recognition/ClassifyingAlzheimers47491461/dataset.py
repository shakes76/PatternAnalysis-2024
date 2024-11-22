from torchvision.datasets import ImageFolder
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def compute_mean_std(dataset):
    """Computes mean and std of colour channels in an image dataset"""
    # Loads the parsed dataset in batches of 128 parralleised with 4 worker threads
    loader = DataLoader(dataset, batch_size=128, num_workers=4)
    # Initializes some variables
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, _ in loader:
        # Gets number of samples in current batch
        batch_samples = images.size(0)
        # Spatially flatten image
        images = images.view(batch_samples, images.size(1), -1)
        # Compute running mean across all batches
        mean += images.mean(2).sum(0)
        # Compute STD across all batches
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    # Actually calculating the average
    mean /= total_images_count
    std /= total_images_count
    return mean, std

class AutoCropBlack:
    def __init__(self, threshold=10):
        """cropping dfunction for getting rid of black border around mri data"""
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
            return img

        # Get coordinates of non-black pixels
        coords = np.argwhere(mask)

        # Get bounding box of non-black pixels
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        # Crop the image to the bounding box
        cropped_img = img.crop((x0, y0, x1, y1))

        return cropped_img


def process(colab=False, test=False, show_samples=False):
    '''Preprocessing pipeline for brain MRI data'''
    preprocess = transforms.Compose([
        # Removes black border present in every MRI image in DB
        AutoCropBlack(),
        # Grayscales image (forces colour channels to be 1)
        transforms.Grayscale(num_output_channels=1),
        # Random rotation of 15 degrees to help generalization
        transforms.RandomRotation(degrees=15),
        # Random re-scaling of image to help generalization
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        # Distorts image randomly, increases variability in dataset (helps with generalizatoin)
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        # Resizes image to align with research paper (as default)
        transforms.Resize((224, 224)),
        # Transforms image into tensor
        transforms.ToTensor(),
        # Normalizes data to highlight 'focal' points in data
        transforms.Normalize(mean=[0.2670], std=[0.2657]),
        # Removes random chunks of data from image to help generalization
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    # If we are training
    if not test:
        # And in local environment
        if not colab:
            # Load images from this local folder
            dataset = ImageFolder(root='C:\\Users\\User\\Desktop\\3710Aux\\ADNI_AD_NC_2D\\AD_NC\\train',
                                transform=preprocess)
        # If we are in an HPC
        else:
            # Load images from googledrive
            dataset = ImageFolder(root='/content/drive/MyDrive/ADNI/AD_NC/train', transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)
    # If we are testing
    else:
        # And in local environment
        if not colab:
            # Load images from local folder
            dataset = ImageFolder(root='C:\\Users\\User\\Desktop\\3710Aux\\ADNI_AD_NC_2D\\AD_NC\\test',
                                transform=preprocess)
        # If we are in HPC
        else:
            # Load images from googledrive
            dataset = ImageFolder(root='/content/drive/MyDrive/ADNI/AD_NC/test', transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    # Return transformed dataset and dataloader
    return dataset, dataloader


if __name__ == "__main__":
    process(colab=False, test=False, show_samples=True)

