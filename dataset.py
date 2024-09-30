import os
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.utils as vutils
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

class BrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith('.png')]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            image = self.transform(image)
        return image
    
if __name__ == "__main__":
    dataset = BrainDataset(IMAGE_PATH)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                            shuffle=True)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig(f'./generated_images3/training_images.png')