"""
DOCSTRING ABOUT WHAT THIS FILE IS HERE 


Mean and standard deviation (for normalisation) should be calculated with the training dataset;

And then this mean/std should be used to normalise the train, val and test dataset! (Don't use test dataset to find mean/std, that causes data leakage).

"""
import torch
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mean_std(loader: DataLoader):
    mean = torch.zeros(3)
    squared_mean = torch.zeros(3)
    N = 0 # Number of batches
    for images, _ in tqdm(loader, desc="Computing mean and std"): 
        # Images are [32, 3, 224, 224]
        
        num_batches, num_channels, height, width = images.shape
        N += num_batches

        mean += images.sum(dim=(0,2,3)) # Mean is size [3], i.e it's the mean sum over each channel [R, G, B]...
        squared_mean += (images ** 2).sum(dim=(0,2,3)) # Accumulate squared mean
    mean /= N * height * width # Divide the summed mean by the number of pixels we've ever seen (i.e mean is on a per pixel basis)

    squared_mean /= N * height * width # Same with squared mean
    
    # Get std
    std = torch.sqrt((squared_mean - mean ** 2)) # Std per pixel
    
    return mean, std


def get_dataloaders(batch_size: int = 32, image_size: int = 224, train_fraction = 0.9, num_workers = 2, path: str = "./ADNI/AD_NC"):
     # train fraction: What percentage is used for training vs validation (0.9 is 90%).

    train_dir = f"{path}/train"
    test_dir = f"{path}/test"

    # Found from using get_mean_std in the past on training dataset.
    mean = torch.Tensor([0.1156, 0.1156, 0.1156]) 
    std =  torch.Tensor([0.2229, 0.2229, 0.2229])

    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(), # For augmentation and better generalisation
        transforms.RandomRotation(degrees=10),  # Random rotation between -10 and +10 degrees
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) # Mean and std from previous cell
    ])

    # Load the full train dataset
    train_data = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    # Load the test dataset (no splitting needed for test)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transforms)

    # Split the train_data into training and validation datasets
    train_size = int(train_fraction * len(train_data))  
    val_size = len(train_data) - train_size  
    train_dataset, val_dataset = random_split(train_data, [train_size, val_size])


    # Create DataLoaders for the training, validation, and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last= True) # Drops 8 images here out of 606 * 32 + 8 images...

    # Val/test dataloaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last= True)  # No need to shuffle validation set
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last= True)

    # Check dataset sizes
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_data)}")

    # Shuffle not needed for test because we aren't training on it. Shuffle is mainly used because, imagine if you had all your 0 classifications in the beginning and the dataloader ate this. You're feeding the NN hundreds of entire "0s" that it's going to backprop in this direction, instead of an 'unbiased sample'. This can affect how well it learns. Shuffling helps better convergence, and generalisation.

    return train_loader, val_loader, test_loader
