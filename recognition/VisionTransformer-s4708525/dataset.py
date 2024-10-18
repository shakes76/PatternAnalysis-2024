from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os

class ADNI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, sub_dir in enumerate(['NC', 'AD']):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            for image_name in os.listdir(sub_dir_path):
                self.image_paths.append(os.path.join(sub_dir_path, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        image = Image.open(self.image_paths[idx]).convert('RGB')  # 
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label