import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
import numpy as np

class MRIDataset(Dataset):
    """
    Custom Dataset for loading 2D MRI slices.
    Args:
        root_dir (str): Directory containing the images.
        transform (callable, optional): Optional transform to be applied on an image sample.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.nii.gz')]

        # Debugging statement to check the number of images
        print(f"Number of images loaded: {len(self.image_files)}")

    def __len__(self):
        """Returns the total number of image files in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Loads and returns an image at the given index."""
        # Get the image file path
        img_path = self.image_files[idx]

        # Load the NIfTI file
        img_nii = nib.load(img_path)
        image = img_nii.get_fdata()  # Get the image data as a numpy array

        image = np.expand_dims(image, axis=0)  # Adding a channel dimension to get shape (1, H, W)

        # Ensure the image has a single channel
        assert image.shape[0] == 1, f"Expected image to have a single channel, but got {image.shape[0]} channels."

        # Convert to numpy array (required for the PIL image conversion)
        image = image.squeeze(0)  # Remove the channel dimension for PIL conversion, now shape (H, W)
        image = image.astype(np.float32)  # Ensure data is in float32 format

        # Apply the transforms (if any)
        if self.transform:
            image = self.transform(image)

        # Return the image and its corresponding index (can be adapted for labels if needed)
        return image, idx

def get_dataloader(root_dir, batch_size=32, image_size=64, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the MRI dataset.
    Args:
        root_dir (str): Directory containing the images.
        batch_size (int): Number of samples per batch.
        image_size (int): Size to resize the images.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: The DataLoader object for the dataset.
    """
    # Define the image transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),                     # Convert numpy array to PIL image
        transforms.Resize((image_size, image_size)), # Resize image to (image_size, image_size)
        transforms.ToTensor(),                       # Convert image to a PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize tensor values to [-1, 1]
    ])

    # Create the dataset object
    dataset = MRIDataset(root_dir=root_dir, transform=transform)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

# Example usage
if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "keras_slices", "keras_slices_train")
    print(f"Data Directory: {data_dir}")

    dataloader = get_dataloader(root_dir=data_dir, batch_size=16, image_size=64, shuffle=True)

    for batch_idx, (images, _) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1} | Image Batch Shape: {images.shape}")
