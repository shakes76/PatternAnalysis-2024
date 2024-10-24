# import os
# import random
# import pandas as pd
# import kagglehub
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset
# from PIL import Image

# class ISICDataset(Dataset):
#     def __init__(self, dataset_path, metadata_path, transform=None):
#         self.dataset_path = dataset_path
#         self.transform = transform
#         self.data = self.load_data()
#         self.labels = self.load_labels(metadata_path)

#         self.add_augmented_malignant_cases()

#     def load_data(self):
#         image_files = []
#         for root, _, files in os.walk(self.dataset_path):
#             for file in files:
#                 if file.endswith('.jpg') or file.endswith('.jpeg'):
#                     image_files.append(os.path.join(root, file))
#         return image_files

#     def load_labels(self, metadata_path):
#         metadata = pd.read_csv(metadata_path)
#         return {row['isic_id'] + '.jpg': row['target'] for _, row in metadata.iterrows()}

#     def get_label_from_filename(self, filename):
#         img_id = os.path.basename(filename)
#         return self.labels.get(img_id, None)

#     def add_augmented_malignant_cases(self, num_augmentations=5):
#         original_data = self.data.copy()
#         original_labels = self.labels.copy()

#         malignant_transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(30),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#         ])

#         for img_path in original_data:
#             label = original_labels.get(os.path.basename(img_path), None)
#             if label == 1:
#                 img = Image.open(img_path).convert("RGB")
#                 for _ in range(num_augmentations):
#                     augmented_img = malignant_transform(img)
#                     augmented_img_path = img_path.replace('.jpg', f'_augmented_{_}.jpg')
#                     augmented_img.save(augmented_img_path)
#                     self.data.append(augmented_img_path)
#                     self.labels[os.path.basename(augmented_img_path)] = label  # Add to labels

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         img = self.data[index]
#         label = self.get_label_from_filename(img)
#         return img, label

# class SiameseDataset(Dataset):
#     def __init__(self, dataset, num_pairs=50000, transform=None):
#         self.dataset = dataset
#         self.transform = transform
#         # self.pairs = self.generate_pairs(num_pairs)
#         self.len = num_pairs # Can be arbitrary since pairs can keep being generated

#     def generate_pair(self, index):
#         index = index % len(self.dataset)
#         img0_tuple, label0 = self.dataset[index]
#         should_get_same_class = random.randint(0, 1)

#         if should_get_same_class:
#             # Find an image with the same class
#             while True:
#                 img1_tuple, label1 = random.choice(self.dataset)
#                 if label0 == label1:
#                     break
#         else:
#             # Find an image with a different class
#             while True:
#                 img1_tuple, label1 = random.choice(self.dataset)
#                 if label0 != label1:
#                     break

#         img0 = Image.open(img0_tuple).convert("RGB")
#         img1 = Image.open(img1_tuple).convert("RGB")

#         if self.transform is not None:
#             img0 = self.transform(img0)
#             img1 = self.transform(img1)

#         similarity_label = torch.tensor(int(label0 != label1), dtype=torch.float32)

#         return img0, img1, similarity_label

#     def generate_pairs(self, num_pairs):
#         pairs = []
#         for _ in range(num_pairs):
#             pairs.append(self.generate_pairs())
#         return pairs

#     def __len__(self):
#         return self.len

#     def __getitem__(self, index):
#         return self.generate_pair(index)

# if __name__ == "__main__":
#     dataset_path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
#     dataset_image_path = os.path.join(dataset_path, "train-image/image")
#     meta_data_path = os.path.join(dataset_path, "train-metadata.csv")

#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#     ])

#     isic_dataset = ISICDataset(dataset_path=dataset_image_path, metadata_path=meta_data_path, transform=transform)

#     siamese = SiameseDataset(dataset=isic_dataset, transform=transform)
import os
import pandas as pd
import kagglehub
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class ISICDataset(Dataset):
    def __init__(self, dataset_path, metadata_path, transform=None, augment_transform=None, num_augmentations=5):
        self.dataset_path = dataset_path
        self.transform = transform
        self.augment_transform = augment_transform
        self.num_augmentations = num_augmentations
        self.labels = self.load_labels(metadata_path)  # Load labels first
        self.data, self.malignant_data = self.load_data()  # Store image paths and malignant subset

    def load_labels(self, metadata_path):
        """Load the labels from the metadata file."""
        metadata = pd.read_csv(metadata_path)
        # Return a dictionary mapping the image filenames to their labels
        return {row['isic_id'] + '.jpg': row['target'] for _, row in metadata.iterrows()}

    def load_data(self):
        """Load image paths, and split into benign and malignant subsets."""
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
        """Total length includes both original data and augmented malignant images."""
        return len(self.data) + (self.num_augmentations * len(self.malignant_data))

    def __getitem__(self, index):
        """Load and return image along with its label."""
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

        # Load the image from disk
        img = Image.open(img_path).convert('RGB')

        # Apply augmentations only for malignant images
        if augmentation and self.augment_transform:
            img = self.augment_transform(img)

        # Apply the main transformation (resize, normalization, etc.)
        if self.transform:
            img = self.transform(img)

        # Get the label based on the filename
        label = self.labels[img_name]

        return img, label

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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])

    isic_dataset = ISICDataset(
        dataset_path=dataset_image_path,
        metadata_path=meta_data_path,
        transform=transform,
        augment_transform=augment_transform,
        num_augmentations=5  # Number of augmentations per malignant case
    )