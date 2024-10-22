import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, test_size=0.2, random_state=42):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): If True, creates dataset from training set, otherwise creates from test set.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducibility.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.images, self.labels = self.load_data()

    def load_data(self):
        """
        Load the ADNI dataset from JPEG files.
        Returns:
            tuple: (image_paths, labels)
        """
        image_paths = []
        labels = []
        class_dirs = ['AD', 'NC']  # Alzheimer's Disease and Cognitive Normal

        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(self.root_dir, class_name)
            logger.info(f"Loading {class_name} images from {class_dir}")

            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    file_path = os.path.join(class_dir, filename)
                    image_paths.append(file_path)
                    labels.append(class_idx)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
        )

        if self.train:
            logger.info(f"Training set size: {len(X_train)}")
            return X_train, y_train
        else:
            logger.info(f"Test set size: {len(X_test)}")
            return X_test, y_test

    def pad_to_square(self, image):
        """
        Add padding to make the image square (256x256).
        Args:
            image (PIL.Image): Input image
        Returns:
            PIL.Image: Padded square image
        """
        width, height = image.size
        target_size = 256

        # Calculate padding
        delta_w = target_size - width
        delta_h = target_size - height
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

        # Add padding with black (0)
        padded_image = Image.new('L', (target_size, target_size), 0)
        padded_image.paste(image, (padding[0], padding[1]))
        return padded_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Open as grayscale

        # Add padding to make the image square
        image = self.pad_to_square(image)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transform():
    """
    Get the transformation pipeline for the images.
    Returns:
        transforms.Compose: The transformation pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # Remove Resize since we're padding instead
        transforms.Normalize((0.5,), (0.5,)),  # Normalize for grayscale images
    ])


def get_dataloader(root_dir, batch_size, train=True, num_workers=4):
    """
    Create a DataLoader for the ADNI dataset.
    Args:
        root_dir (string): Directory with all the images.
        batch_size (int): Size of each batch.
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        num_workers (int): Number of subprocesses to use for data loading.
    Returns:
        torch.utils.data.DataLoader: DataLoader for the ADNI dataset.
    """
    dataset = ADNIDataset(root_dir, transform=get_transform(), train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)


# Example usage and testing
if __name__ == "__main__":
    root_dir = "./dataset/ADNI/train/"
    train_loader = get_dataloader(root_dir, batch_size=32, train=True)
    test_loader = get_dataloader(root_dir, batch_size=32, train=False)

    # Print some information about the dataset
    for images, labels in train_loader:
        logger.info(f"Batch shape: {images.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Label distribution: {np.bincount(labels)}")

        # Save a few example images to verify padding
        for i in range(min(5, len(images))):
            img = images[i].cpu().numpy()
            plt.figure(figsize=(5, 5))
            plt.imshow(img[0], cmap='gray')
            plt.title(f"Sample {i + 1} - Shape: {img.shape}")
            plt.savefig(f"sample_{i + 1}_padded.png")
            plt.close()
        break