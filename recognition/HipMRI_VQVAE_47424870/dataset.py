import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Custom dataset for MRI slices
class MRIDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Function to create data loaders
def get_data_loader(image_dir, batch_size=32, shuffle=True, transform=None):
    dataset = MRIDataset(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
