import numpy as np
import torch
import tensorflow as tf
from torchvision import  datasets, transforms
import torchvision
import matplotlib.pyplot as plt

def data_set_creator():
    augmentation_transforms = transforms.Compose([
        #No vertical fliping was applied
        transforms.RandomHorizontalFlip(),
        #To account for skewing
        transforms.RandomRotation(30),
        #Coloring was altered to stop overfitting
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])


    data_dir = 'recognition/Stable Diffusion - 47219647/AD_NC/test'
    dataset = datasets.ImageFolder(root=data_dir, transform=augmentation_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)
    

    #Shows the first few comments for quality check
    #Uncomment if needed
    # def show_image(image_tensor):
    #     image = image_tensor.permute(1, 2, 0).numpy()
    #     plt.imshow(image)
    #     plt.axis('off')
    #     plt.show()

    # num_images_to_show = 5
    # for i in range(num_images_to_show):
    #     image, _ = dataset[i] 
    #     show_image(image)

    return data_loader


data_set_creator()


