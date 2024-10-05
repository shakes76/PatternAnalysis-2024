import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms



# Root directory for dataset
dataroot = ""


# Spatial size of training images
image_size = 64

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Dataset and DataLoader
dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)