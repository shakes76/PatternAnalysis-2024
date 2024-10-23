import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform