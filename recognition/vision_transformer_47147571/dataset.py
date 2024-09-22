import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate through AD and NC folders
        for label_name in ['AD', 'NC']:
            folder_path = os.path.join(root_dir, label_name)
            label = 1 if label_name == 'AD' else 0  # Label AD as 1, NC as 0
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    # Return the total number of images
    def __len__(self):
        return len(self.image_paths)

    # Retrieve the image and its label at a given index
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load the image and convert it to grayscale
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        
        # Apply transformations if they are defined
        if self.transform:
            image = self.transform(image)

        return image, label
