import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BrainDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        for idx, class_name in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, class_name)
            
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(idx)  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("L") 

        if self.transform:
            image = self.transform(image)
        return image, label