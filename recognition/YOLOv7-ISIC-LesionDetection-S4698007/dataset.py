import os
import nibabel as nib
import gzip
import shutil
import torch
from torch.utils.data import Dataset, DataLoader

class NiftiDataset(Dataset):
    def __init__(self, image_directory, label_directory, normImage=False, max_images=50):
        self.image_directory = image_directory
        self.label_directory = label_directory
        self.normImage = normImage
        
        self.image_files = [f for f in os.listdir(image_directory) if f.endswith('.gz')]
        self.label_files = [f for f in os.listdir(label_directory) if f.endswith('.gz')]

        assert len(self.image_files) == len(self.label_files), "The number of images and labels must match."

        self.image_files = self.image_files[:max_images]
        self.label_files = self.label_files[:max_images]

    def decompress_gz(self, file_path):
        decompressed_file = file_path.replace('.gz', '')
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return decompressed_file

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = self.label_files[idx]

        compressed_image_path = os.path.join(self.image_directory, image_file)
        compressed_label_path = os.path.join(self.label_directory, label_file)

        decompressed_image_path = self.decompress_gz(compressed_image_path)
        decompressed_label_path = self.decompress_gz(compressed_label_path)

        img = nib.load(decompressed_image_path).get_fdata()
        label = nib.load(decompressed_label_path).get_fdata()

        if self.normImage:
            img = (img - img.mean()) / img.std()

        label = label.astype(int)
        label[label >= 6] = 0

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor

def create_dataloader(image_directory, label_directory, batch_size=4, max_images=50, normImage=True):
    dataset = NiftiDataset(image_directory, label_directory, normImage, max_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)