import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        ad_path = os.path.join(root_dir, 'AD')
        nc_path = os.path.join(root_dir, 'NC')

        if os.path.isdir(ad_path):
            for file_name in os.listdir(ad_path):
                file_path = os.path.join(ad_path, file_name)
                if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(file_path)
                    self.labels.append(0)

        if os.path.isdir(nc_path):
            for file_name in os.listdir(nc_path):
                file_path = os.path.join(nc_path, file_name)
                if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(file_path)
                    self.labels.append(1)

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
