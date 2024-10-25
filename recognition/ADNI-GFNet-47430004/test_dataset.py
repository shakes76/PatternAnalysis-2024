"""
This code tests whether the dataset loader is loading in the data correctly.
The verification is done by manual, visual inspection of the images.
"""
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dataset import get_dataloaders
import numpy as np

# Remember to disable random horizontal/vertical flip to make it easier to see
train_dataloader, _ = get_dataloaders(None)

images, labels = next(iter(train_dataloader))

images = images * 0.2385 + 0.1415 

grid_img = vutils.make_grid(images, nrow=4, padding=2)

np_grid_img = grid_img.numpy()

num_images_in_row = 4
image_height, image_width = 224, 224
figure_width = num_images_in_row * image_width / 100
figure_height = (32 // num_images_in_row + 1) * image_height / 100 
plt.figure(figsize=(figure_width, figure_height))


plt.imshow(np.transpose(np_grid_img, (1, 2, 0)), cmap='gray')
plt.axis('off')
plt.savefig("test/dataset/test_dataset", bbox_inches='tight', pad_inches=0)
plt.close()