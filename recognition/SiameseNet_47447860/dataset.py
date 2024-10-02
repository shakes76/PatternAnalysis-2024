# the data loader for loading and preprocessing your data
import torch
from torchvision.transforms import v2
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not Found. Using CPU")

images_path = "~/.kaggle/train-image/image"
csv_path = "~/.kaggle/train-metadata.csv"

# Setting up tranformer to transform images to rebalance the data set
H, W = 256, 256
img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)
augment_transform = v2.Compose([
    v2.RandomResizedCrop(size=(256, 256), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# transformer used for incoming belign data
transform_w_normalise = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet values)
])

# transformer used for incoming malignant data
transform_no_normalise = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# checking the disproportion
csv = pd.read_csv(csv_path)
total_row = csv.len()
malignant_count = csv['target'].sum()
benign_count = total_row - malignant_count

ratio = benign_count // malignant_count


class ISICDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        """
        Args:
            csv_file (string): Path to the csv file with image names and classifications.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_file)  # Load the CSV file
        self.img_dir = img_dir  # Path to the image directory

    def __len__(self):
        return len(self.data)  # Return the size of the dataset

    def __getitem__(self, idx):
        # Get the image filename and classification
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 3]  # classification is in 4th column

        # Load the image and apply appropriate transformation (no normalisation for malignant images
        # because we normalise them all after we augment)
        image = Image.open(img_name)
        if label:
            # malignant
            image = transform_no_normalise(image)
        else:
            # benign
            image = transform_w_normalise(image)

        return image, label

def filter_dataset_by_label(dataset, label):
    """
    Filter the dataset to only include samples with the given label.
    Args:
        dataset: The dataset to filter
        label: The label to filter by (e.g., 0 or 1)
    Returns:
        A filtered dataset containing only the specified label.
    """
    indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == label]
    return torch.utils.data.Subset(dataset, indices)

class AugmentedISICDataset(Dataset):
    def __init__(self, base_dataset, num_augmentations=1):
        """
        Args:
            base_dataset (Dataset): The original dataset with images to augment.
            transform (callable): The transformations to apply to each image.
            num_augmentations (int): How many times to apply transformations to augment the data.
        """
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations

    def __len__(self):
        # The length is the original dataset size multiplied by the number of augmentations
        return len(self.base_dataset) * self.num_augmentations

    def __getitem__(self, idx):
        # Use modulo to access the base dataset, and apply the transform
        original_idx = idx % len(self.base_dataset)
        image, label = self.base_dataset[original_idx]
        augmented_image = augment_transform(image)
        return augmented_image, label

# Load the images
dataset = ISICDataset(csv_file=csv_path, img_dir=images_path)

# Filter the dataset for each classification
benign_dataset = filter_dataset_by_label(dataset, label=0)
malignant_dataset = filter_dataset_by_label(dataset, label=1)

# Augment malignant multiple times to balance it
augmented_malignant_dataset = AugmentedISICDataset(malignant_dataset, augment_transform, num_augmentations=ratio)

# Create a data loaders
benign_loader = DataLoader(benign_dataset, batch_size=32, shuffle=True)
malignant_loader = DataLoader(augmented_malignant_dataset, batch_size=32, shuffle=True)


