import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
data_dir = '/Users/gghollyd/comp3710/report/AD_NC/'
def dataloader(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  
            transforms.Resize((240, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Use single-channel normalization for grayscale
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  
            transforms.Resize((240, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Single-channel normalization for grayscale
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(root=data_dir + '/train', transform=data_transforms['train']),
        'test': datasets.ImageFolder(root=data_dir + '/test', transform=data_transforms['test']),
    }

    train_loader = DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, image_datasets['train'].classes

