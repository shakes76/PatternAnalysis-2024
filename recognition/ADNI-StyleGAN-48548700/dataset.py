# Import necessary libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a function to create a data loader
def get_loader(dataset, log_resolution, batch_size):
    # Define a series of data transformations to be applied to the images
    transform = transforms.Compose(
        [
            # Resize images to a specified resolution (2^log_resolution x 2^log_resolution)
            transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
            
            # Convert the images to PyTorch tensors
            transforms.ToTensor(),
            
            # Apply random horizontal flips to augment the data (50% probability)
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Normalize the pixel values of the images to have a mean and standard deviation of 0.5
            transforms.Normalize(
                [0.5, 0.5, 0.5],  # Mean for each channel
                [0.5, 0.5, 0.5],  # Standard deviation for each channel
            ),
        ]
    )
    
    # Create an ImageFolder dataset object that loads images from the specified directory
    dataset = datasets.ImageFolder(root=dataset, transform=transform)
    
    # Create a data loader that batches and shuffles the data
    loader = DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=True,          # Shuffle the data for randomness
    )
    
    return loader
