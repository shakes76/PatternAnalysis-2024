import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from sklearn.model_selection import train_test_split
from torchvision import transforms

class ADNC_Dataset(Dataset):
    def __init__(self, meta_data_path, base_data_dir, transform=None, split='train', val_split=0.15):
        self.transform = transform
        self.split = split
        self.base_data_dir = base_data_dir
        
        with open(meta_data_path, 'r') as f:
            self.meta_data = json.load(f)
        
        self.image_paths = []
        self.labels = []
        
        all_image_paths, all_labels = self.load_data()

        if split == 'train':
            self.image_paths, _, self.labels, _ = train_test_split(all_image_paths, all_labels, test_size=val_split, stratify=all_labels)
        elif split == 'val':
            _, self.image_paths, _, self.labels = train_test_split(all_image_paths, all_labels, test_size=val_split, stratify=all_labels)
        else:
            self.image_paths = all_image_paths
            self.labels = all_labels

    def load_data(self):
        all_image_paths = []
        all_labels = []
        
        split_dir = os.path.join(self.base_data_dir, 'train')  

        for image_id, data in self.meta_data.items():
            if 'train' in data['masked']:
                img_path = os.path.join(split_dir, 'AD' if data['label'] == 0 else 'NC', data['masked'])
                all_image_paths.append(img_path)
                all_labels.append(data['label'])

        return all_image_paths, all_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
