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

H, W = 256, 256


# Define a ISICDataLoader class
class ISICDataLoader:
    def __init__(self, csv_file, img_dir, batch_size=32):
        """
        Args:
            csv_file (string): Path to the csv file containing image information.
            img_dir (string): Directory with all the images
            batch_size (int): Batch size for the data loaders
        """
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.batch_size = batch_size

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("Warning: CUDA not Found. Using CPU")

        # Setting up transformers
        self.augment_transform = v2.Compose([
            v2.RandomResizedCrop(size=(256, 256), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Transformer used for incoming benign data
        self.transform_w_normalise = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet values)
        ])

        # Transformer used for incoming malignant data (without normalization initially)
        self.transform_no_normalise = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        # Load the CSV and calculate the ratio for balancing -> number of times we need to augment each image
        self.csv = pd.read_csv(self.csv_file)
        total_rows = len(self.csv)
        malignant_count = self.csv['target'].sum()
        benign_count = total_rows - malignant_count
        self.num_augmentations = benign_count // malignant_count

        # Initialize the dataset
        self.dataset = self.ISICDataset(self.csv_file, self.img_dir)

    class ISICDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with image names and classifications.
                img_dir (string): Directory with all the images.
            """
            self.data = pd.read_csv(csv_file)  # Load the CSV file
            self.img_dir = img_dir  # Path to the image directory

        def __len__(self):
            return len(self.data)  # Return size of the dataset

        def __getitem__(self, idx):
            # For a particular index, get the image name and its classification
            img_name = os.path.join(self.img_dir, self.data.iloc[idx, 1])  # image names are 2nd column
            label = self.data.iloc[idx, 3]  # classification is in 4th column

            # Load the image and apply appropriate transformation
            image = Image.open(img_name)
            if label == 1:  # Malignant
                image = ISICDataLoader().transform_no_normalise(image)
            else:  # Benign
                image = ISICDataLoader().transform_w_normalise(image)

            return image, label

    def filter_dataset_by_label(self, dataset, label):
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
                num_augmentations (int): How many times to apply transformations to augment the data.
            """
            self.base_dataset = base_dataset
            self.num_augmentations = num_augmentations

        def __len__(self):
            # The length is the original dataset size multiplied by the number of augmentations
            return len(self.base_dataset) * self.num_augmentations  # creating a virtual extended dataset

        def __getitem__(self, idx):
            # When we want to access a new (augmented image),
            # use modulo to get a corresponding index to an existing image in the original dataset
            original_idx = idx % len(self.base_dataset)
            # We get the image and a label (which should always be 1 (malignant))
            image, label = self.base_dataset[original_idx]
            # We then augment this existing image and return it as a 'new' image
            augmented_image = ISICDataLoader().augment_transform(image)
            # This means we aren't pre-computing every augmented image we need,
            # but that we are doing the augmentation on the fly when we need the image
            return augmented_image, label

    def get_data_loaders(self):
        """
        Creates the data loaders for benign and augmented malignant datasets.
        Returns:
            Tuple of benign and malignant data loaders.
        """
        # Filter the dataset for each classification
        benign_dataset = self.filter_dataset_by_label(self.dataset, label=0)
        malignant_dataset = self.filter_dataset_by_label(self.dataset, label=1)

        # Augment malignant multiple times to balance it
        augmented_malignant_dataset = self.AugmentedISICDataset(malignant_dataset, self.num_augmentations)

        # Create data loaders
        benign_loader = DataLoader(benign_dataset, batch_size=self.batch_size, shuffle=True)
        malignant_loader = DataLoader(augmented_malignant_dataset, batch_size=self.batch_size, shuffle=True)

        return benign_loader, malignant_loader


