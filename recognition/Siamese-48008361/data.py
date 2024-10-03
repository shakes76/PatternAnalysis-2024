import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import random
from torchvision import transforms
import numpy as np

class SiameseDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, n_pairs=500000):
        self.df = pd.read_csv(csv_file, sep=',')
        self.df = self.df[['image_name', 'target']]
        self.image_dir = image_dir
        self.transform = transform
        
        self.benign_images = self.df[self.df['target'] == 0]['image_name'].tolist()
        self.malignant_images = self.df[self.df['target'] == 1]['image_name'].tolist()
        
        self.image_pairs, self.labels = self.generate_pairs(n_pairs)

    def generate_pairs(self, n_pairs):
        pairs = []
        labels = []
        
        for _ in range(n_pairs):
            if random.random() < 0.5:
                # Positive pair
                if random.random() < 0.5:
                    pair = random.sample(self.benign_images, 2)
                else:
                    pair = random.sample(self.malignant_images, 2)
                label = 1
            else:
                # Negative pair
                pair = (random.choice(self.benign_images), random.choice(self.malignant_images))
                label = 0
            
            pairs.append(pair)
            labels.append(label)
        
        return pairs, labels
    
    def show_sample_pairs(self, num_samples=5):
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        for i in range(num_samples):
            idx = random.randint(0, len(self) - 1)
            img1, img2, label = self[idx]
            
            # Convert tensor to numpy array and transpose to (H, W, C)
            img1 = img1.numpy().transpose(1, 2, 0)
            img2 = img2.numpy().transpose(1, 2, 0)
            
            # Denormalize if necessary
            if self.transform:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img1 = std * img1 + mean
                img2 = std * img2 + mean
            
            img1 = np.clip(img1, 0, 1)
            img2 = np.clip(img2, 0, 1)
            
            axes[i, 0].imshow(img1)
            axes[i, 0].axis('off')
            axes[i, 1].imshow(img2)
            axes[i, 1].axis('off')
            axes[i, 0].set_title(f"Pair {i+1}: {'Similar' if label == 1 else 'Different'}")
        
        plt.tight_layout()

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_name1, img_name2 = self.image_pairs[idx]
        label = self.labels[idx]

        # Load images
        img_path1 = os.path.join(self.image_dir, img_name1 + '.jpg')
        img_path2 = os.path.join(self.image_dir, img_name2 + '.jpg')
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')

        # Apply transformations
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    from torchvision import transforms

    # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create the dataset
    dataset = SiameseDataset('ISIC_2020_Training_GroundTruth_v2.csv', 
                             'data/ISIC_2020_Training_JPEG/train/', 
                             transform=transform,
                             n_pairs=200000)
    
    # Print some information about the dataset
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of benign images: {len(dataset.benign_images)}")
    print(f"Number of malignant images: {len(dataset.malignant_images)}")

    # Show some sample pairs
    dataset.show_sample_pairs(num_samples=5)