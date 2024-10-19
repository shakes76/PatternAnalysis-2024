from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class ADNI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.category_counts = {'NC': 0, 'AD': 0}  

        for label, sub_dir in enumerate(['NC', 'AD']):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            num_images = len(os.listdir(sub_dir_path))  
            self.category_counts[sub_dir] = num_images  

            for image_name in os.listdir(sub_dir_path):
                self.image_paths.append(os.path.join(sub_dir_path, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_category_counts(self):
        return self.category_counts