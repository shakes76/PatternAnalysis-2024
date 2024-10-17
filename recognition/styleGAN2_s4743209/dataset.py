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
        Load the ADNI dataset from NIFTI files.
        Returns:
            tuple: (images, labels)
        """
        images = []
        labels = []
        class_dirs = ['AD', 'CN']  # Alzheimer's Disease and Cognitive Normal

        for class_idx, class_name in enumerate(class_dirs):
            class_dir = os.path.join(self.root_dir, class_name)
            logger.info(f"Loading {class_name} images from {class_dir}")

            for filename in os.listdir(class_dir):
                if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                    file_path = os.path.join(class_dir, filename)
                    try:
                        image = self.load_nifti(file_path)
                        images.append(image)
                        labels.append(class_idx)
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {str(e)}")

        images = np.array(images)
        labels = np.array(labels)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
        )

        if self.train:
            logger.info(f"Training set size: {len(X_train)}")
            return X_train, y_train
        else:
            logger.info(f"Test set size: {len(X_test)}")
            return X_test, y_test

    def load_nifti(self, file_path):
        """
        Load a NIFTI file and preprocess the image.
        Args:
            file_path (string): Path to the NIFTI file.
        Returns:
            numpy.ndarray: The preprocessed image.
        """
        nifti_img = nib.load(file_path)
        image_data = nifti_img.get_fdata()

        # Preprocess the image
        preprocessed_image = self.preprocess_image(image_data)
        return preprocessed_image

    def preprocess_image(self, image):
        """
        Preprocess the loaded NIFTI image.
        Args:
            image (numpy.ndarray): The loaded image data.
        Returns:
            numpy.ndarray: The preprocessed image.
        """
        # Normalize the image
        image = (image - image.min()) / (image.max() - image.min())

        # Resize to a fixed size (e.g., 128x128x128)
        zoom_factors = np.array([128, 128, 128]) / np.array(image.shape)
        image = zoom(image, zoom_factors, order=1)

        # Take a central slice (for 2D StyleGAN2)
        central_slice = image[:, :, image.shape[2] // 2]

        return central_slice

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
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
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Resize((128, 128)),  # Adjust size as needed for StyleGAN2
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


# Example usage
if __name__ == "__main__":
    root_dir = "/home/groups/comp3710/ADNI"
    train_loader = get_dataloader(root_dir, batch_size=32, train=True)
    test_loader = get_dataloader(root_dir, batch_size=32, train=False)

    # Print some information about the dataset
    for images, labels in train_loader:
        logger.info(f"Batch shape: {images.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        break

    # Visualize a sample image
    plt.imshow(images[0][0], cmap='gray')
    plt.title(f"Label: {'AD' if labels[0] == 0 else 'CN'}")
    plt.savefig('sample_adni_image.png')
    logger.info("Sample image saved as 'sample_adni_image.png'")