import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# Define the paths
train_dir = 'dataset/AD_NC/train'
test_dir = 'dataset/AD_NC/test'
batch_size = 32

IMAGE_DIM = 240 
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_DIM//PATCH_SIZE) ** 2
D_MODEL = (PATCH_SIZE**2) * 3

transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.ToTensor(), 
    # Using imagenet normalization https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to create data loaders for train and test sets
def create_data_loader(root_dir, batch_size=32, train=True):
    """
    Creates a DataLoader for the given dataset directory.
    
    Args:
        root_dir (str): Directory containing the dataset (train or test).
        batch_size (int): Number of samples per batch.
        train (bool): Whether this is a training loader (shuffles data if True).
    
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    return loader


# Print class-to-index mapping for reference
if __name__ == "__main__":
    # Create DataLoaders for training and testing
    train_loader = create_data_loader(train_dir, batch_size=batch_size, train=True)
    test_loader = create_data_loader(test_dir, batch_size=batch_size, train=False)

    print(f"Class-to-index mapping: {train_loader.dataset.class_to_idx}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of testing samples: {len(test_loader.dataset)}")