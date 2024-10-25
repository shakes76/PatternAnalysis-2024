import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

data_dir = '/Users/gghollyd/comp3710/report/AD_NC/'

def dataloader(data_dir, val_split=0.2, batch_size=128):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  
            transforms.Resize((240, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  
            transforms.Resize((240, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  
        ]),
    }
    image_datasets = {
        'train': datasets.ImageFolder(root=data_dir + '/train', transform=data_transforms['train']),
        'test': datasets.ImageFolder(root=data_dir + '/test', transform=data_transforms['test']),
    }

    # split train dataset into train and validation sets
    num_train = len(image_datasets['train'])
    num_val = int(val_split * num_train)
    num_train = num_train - num_val

    train_subset, val_subset = random_split(image_datasets['train'], [num_train, num_val])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, image_datasets['train'].classes

