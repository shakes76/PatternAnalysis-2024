"""
Extra functions and classes that have been used to create this model.

This code does not been to be included to run the actual model. Much of the processing here
has been hardcoded into the model.

REF: 
"""
import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from dataset import *

image_size = 256
batch_size = 64
    # used for calculating mean and std
transform =  {
    'no_norm': transforms.Compose([
        transforms.Resize(image_size),
         transforms.Grayscale(),
        transforms.ToTensor(),
    ]),
}

# The path when running locally
data_directory = os.path.join('../../../AD_NC')
#the path to the directory on Rangpur
#data_directory = '/home/groups/comp3710/ADNI/AD_NC'

def get_mean_std(dataset):
    """Compute the mean and std value of the dataset"""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    mean = torch.zeros(1)
    std = torch.zeros(1)

    # Calculate mean and std for each
    for inputs, _ in dataloader:
        mean[0] += inputs.mean()
        std[0] += inputs.std()
    
    mean.div_(len(dataset))
    std.div_(len(dataset))
    
    return mean, std

if __name__ == "__main__":
    # Create train dataset without normalization
    transform_no_norm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # Only tensor conversion, no normalization
    ])

    new_dir = data_dir=os.path.join(data_directory, 'train')
    # Create train dataset
    train_dataset = ADNIDataset(new_dir, transform=transform, mode='no_norm')

    # Calculate mean and std
    mean, std = get_mean_std(train_dataset)
    print(f"Calculated mean: {mean}")
    print(f"Calculated std: {std}")
