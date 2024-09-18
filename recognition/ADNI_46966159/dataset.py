import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# specify directory of data
dataroot = "/home/lgmoak/Nextcloud/University/Courses/COMP3710/Assessment/PatternAnalysis-2024/recognition/ADNI_46966159/ADNI"
# dataroot = "/home/groups/comp3710/ADNI"

batch_size = 128
n_epochs = 20
lr = 0.001
image_size = 256

dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                           transform=transforms.Compose([
                                               transforms.Resize(image_size),
                                               transforms.CenterCrop(image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()