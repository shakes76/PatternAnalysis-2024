import os
import random
import pandas as pd
import kagglehub
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

# This class loads the ISIC dataset and applies augmentations to malignant images
class ISICDataset(Dataset):
    def __init__(self, dataset_path, metadata_path, transform=None, augment_transform=None, num_augmentations=5):
        self.dataset_path = dataset_path
        self.transform = transform
        self.augment_transform = augment_transform
        self.num_augmentations = num_augmentations
        self.labels = self.load_labels(metadata_path) 
        self.data, self.malignant_data = self.load_data()

    def load_labels(self, metadata_path):
        metadata = pd.read_csv(metadata_path)
        return {row['isic_id'] + '.jpg': row['target'] for _, row in metadata.iterrows()}

    def load_data(self):
        image_files = []
        malignant_files = []
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if (file.endswith('.jpg') or file.endswith('.jpeg')) and file in self.labels:
                    image_files.append(file)
                    if self.labels[file] == 1:  # Malignant images (label = 1)
                        malignant_files.append(file)

        return image_files, malignant_files

    def __len__(self):
        return len(self.data) + (self.num_augmentations * len(self.malignant_data))

    # Dynamic data loading based on index, as to not use too much memory
    def __getitem__(self, index):
        if index < len(self.data):
            # Return original image
            img_name = self.data[index]
            augmentation = False
        else:
            # Calculate malignant image and augmentation index
            aug_index = index - len(self.data)
            original_index = aug_index // self.num_augmentations
            augmentation = True
            img_name = self.malignant_data[original_index]

        img_path = os.path.join(self.dataset_path, img_name)

        img = Image.open(img_path).convert('RGB')

        if augmentation and self.augment_transform:
            img = self.augment_transform(img)

        if self.transform:
            img = self.transform(img)

        label = self.labels[img_name]

        return img, label


# This class loads data and creates pairs of images for the Siamese Network labelled as similar or different
class SiameseDataset(Dataset):
    def __init__(self, dataset, num_pairs=50000, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.num_pairs = num_pairs

    def generate_pair(self, index):
        index = index % len(self.dataset)
        img0, label0 = self.dataset[index]
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            # Positive pair: Find another image with the same label
            while True:
                img1, label1 = random.choice(self.dataset)
                if label0 == label1:
                    break
        else:
            # Negative pair: Find an image with a different label
            while True:
                img1, label1 = random.choice(self.dataset)
                if label0 != label1:
                    break

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        similarity_label = torch.tensor(int(label0 != label1), dtype=torch.float32)

        return img0, img1, similarity_label

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, index):
        return self.generate_pair(index)

if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
    dataset_image_path = os.path.join(dataset_path, "train-image/image")
    meta_data_path = os.path.join(dataset_path, "train-metadata.csv")

    # Standard transforms for image resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Augmentation transforms for malignant cases
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])

    isic_dataset = ISICDataset(
        dataset_path=dataset_image_path,
        metadata_path=meta_data_path,
        transform=transform,
        augment_transform=augment_transform,
        num_augmentations=5  # Number of augmentations per malignant case
    )