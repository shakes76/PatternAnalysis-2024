import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Loop through the train/test directory
        for class_name in ["NC", "AD"]:
            class_dir = os.path.join(root_dir, class_name)
            label = 0 if class_name == "NC" else 1  # 0 for NC, 1 for AD
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load an image and return it along with the corresponding label."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')
        # Apply transformations 
        if self.transform:
            image = self.transform(image)

        return image, label