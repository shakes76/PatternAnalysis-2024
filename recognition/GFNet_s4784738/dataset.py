"""
Data loader for brain images from the ADNI brain data

Benjamin Thatcher 
s4784738    
"""

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialises the image dataset using ADNI brain data.

        Parameters
        root_dir: Directory with 'train' or 'test' folders containing 'AD' and 'ND' subfolders
        with images inside them.
        transform: Optional transform to be applied to an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse the directory to gather image paths and labels
        for label_name in ['AD', 'NC']:
            label_dir = os.path.join(root_dir, label_name)
            # Set the label to 1 for Alzheimer's (AD) and 0 for Normal (NC)
            label = 1 if label_name == 'AD' else 0
            
            for image_name in os.listdir(label_dir):
                # Ensure the images are all valid (they should all be .jpeg)
                if image_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(label_dir, image_name))
                    self.labels.append(label)

    def __len__(self):
        """
        Returns the length of the image dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Gets the image and label and a certain index in the dataset

        Parameters
        idx: The index in the dataset of the image to be retrieved
        Returns: The image (in RGB form) and its label
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]

        # Transform the image if transforms are being used
        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loader(root_dir, batch_size=32, shuffle=True, num_workers=1):
    """
    Gets a dataloader with characteristics specified by the user.

    Parameters
    root_dir: The root directory in which ADNI images can be found
    batch_size: The batch size for images 
    shuffle: True is images should be shuffled, and False otherwise
    num_workers: The number of 
    Returns a dataloader with the specified characteristics
    """
    transform = transforms.Compose([
        # Resize to 256x240 from the default size of 256x240 pixels
        transforms.Resize((256, 240)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # Pre-calculated mean and standard deviation pixel values
        transforms.Normalize([0.1155], [0.2224]),
    ])

    dataset = ADNIDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader