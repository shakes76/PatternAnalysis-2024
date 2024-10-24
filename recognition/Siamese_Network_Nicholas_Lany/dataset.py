import os
import random
import pandas as pd
import kagglehub
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class ISICDataset(Dataset):
    def __init__(self, dataset_path, metadata_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = self.load_data()
        self.labels = self.load_labels(metadata_path)

    def load_data(self):
        image_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.jpeg'):
                    image_files.append(os.path.join(root, file))
        return image_files

    def load_labels(self, metadata_path):
        metadata = pd.read_csv(metadata_path)
        return {row['isic_id'] + '.jpg': row['target'] for _, row in metadata.iterrows()}

    def get_label_from_filename(self, filename):
        img_id = os.path.basename(filename)
        return self.labels.get(img_id, None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.get_label_from_filename(img)
        return img, label

class SiameseDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    # Yes, this function kind of returns a new pair of images and label each time you call the same index... But does it matter?
    def __getitem__(self, index):
        img0_tuple, label0 = self.dataset[index]

        # We need approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            # Look until the same class image is found
            while True:
                img1_tuple, label1 = self.dataset[random.randint(0, len(self.dataset) - 1)]
                if label0 == label1:
                    break
        else:
            # Look until a different class image is found
            while True:
                img1_tuple, label1 = self.dataset[random.randint(0, len(self.dataset) - 1)]
                if label0 != label1:
                    break

        img0 = Image.open(img0_tuple).convert("RGB")
        img1 = Image.open(img1_tuple).convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        similarity_label = torch.tensor(int(label0 != label1), dtype=torch.float32)

        return img0, img1, similarity_label

if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
    dataset_image_path = os.path.join(dataset_path, "train-image/image")
    meta_data_path = os.path.join(dataset_path, "train-metadata.csv")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    isic_dataset = ISICDataset(dataset_path=dataset_image_path, metadata_path=meta_data_path, transform=transform)

    siamese = SiameseDataset(dataset=isic_dataset, transform=transform)