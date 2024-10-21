import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F  # Import functional transforms

class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir (str): Directory containing the images.
            mask_dir (str): Directory containing the corresponding masks.
            transform (callable, optional): Optional transform to be applied on both images and masks.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_filenames = os.listdir(img_dir)  # List of image filenames
        
        # Ensure filenames are sorted to align images with masks
        self.img_filenames.sort()

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])

        # Get the corresponding mask filename
        img_name = os.path.splitext(self.img_filenames[idx])[0]  # Extract the base filename (without extension)
        mask_filename = f"{img_name}_segmentation.png"  # Assuming the mask is in PNG format
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image {self.img_filenames[idx]} at {mask_path}")

        # Load the image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Load mask as grayscale

        # Apply transformations (to both image and mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask  # Return both the image and the mask

    def get_image_path(self, idx):
        """Retrieve the path of the image for the given index."""
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        return img_path

    def get_mask_path(self, idx):
        """Retrieve the path of the mask for the given index."""
        img_name = os.path.splitext(self.img_filenames[idx])[0]  # Base filename
        mask_filename = f"{img_name}_segmentation.png"  # Assuming mask format
        mask_path = os.path.join(self.mask_dir, mask_filename)
        return mask_path


class SegmentationTransform:
    def __init__(self, resize=(640, 640), normalize=True):
        self.resize = resize
        self.normalize = normalize

    def __call__(self, image, mask):
        # Resize both image and mask
        image = F.resize(image, self.resize)
        mask = F.resize(mask, self.resize, interpolation=Image.NEAREST)  # Use NEAREST for masks to preserve labels

        # Convert to tensor
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()  # Convert mask to tensor with integer labels

        # Normalize image (do not normalize the mask)
        if self.normalize:
            image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        return image, mask

# Instantiate the dataset with transforms
transform = SegmentationTransform()
train_dataset = ISICDataset(img_dir='../../../ISIC2018_Task1-2_Training_Input_x2', mask_dir='../../../ISIC2018_Task1_Training_GroundTruth_x2', transform=transform)

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

# Function to display the image and mask
def show_image_and_mask(image, mask):
    """
    Display a given image and mask side by side.
    
    Args:
        image (Tensor): Tensor of the image (C, H, W).
        mask (Tensor): Tensor of the mask (H, W) or (1, H, W).
    """
    # If mask is (1, H, W), squeeze it to (H, W)
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
    # Convert image back to numpy for plotting
    image_np = image.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    mask_np = mask.numpy()  # Convert mask tensor to numpy
    
    # Create a figure to display both the image and mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")
    
    plt.show()

# Load a sample from the dataset
sample_idx = 2593  # Change this to load different samples (up to 2593)
image, mask = train_dataset[sample_idx]

# Display the image and mask
show_image_and_mask(image, mask)