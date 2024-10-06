import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for class_label, class_name in enumerate(['NC', 'AD']):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(class_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # Load as RGB
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Example usage
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/train', transform=data_transform)
test_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/test', transform=data_transform)