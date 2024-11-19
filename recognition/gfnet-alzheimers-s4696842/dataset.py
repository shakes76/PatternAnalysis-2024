from os import path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# (Height, Width)
ADNI_IMAGE_DIMENSIONS = (256, 240)


def get_dataloader(root_dir, batch_size, shuffle, transform):
    """
    Creates a DataLoader for an image dataset stored in a specified directory with the given transformations.

    Args:
        root_dir (str): Path to the directory containing the dataset.
        batch_size (int): Number of samples per batch to load.
        shuffle (bool): Whether to shuffle the data at every epoch.
        transform (torchvision.transforms.Compose): A composition of image transformations to apply.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the specified dataset with the applied transformations.
    """
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_train_test_dataloaders(root_dir, train_batch_size, test_batch_size):
    """
    Creates ADNI image DataLoaders for training and testing with respective transformations.

    The training DataLoader applies data augmentation such as random affine transformations, resizing, and color jittering,
    while the test DataLoader applies only basic grayscale and resizing transformations.

    Args:
        root_dir (str): Path to the root directory containing 'train' and 'test' subdirectories for the datasets.
        train_batch_size (int): Number of samples per batch for the training DataLoader.
        test_batch_size (int): Number of samples per batch for the test DataLoader.

    Returns:
        tuple:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    """
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomAffine(
                degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1)
            ),
            transforms.RandomResizedCrop(size=(256, 240), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.15, 2.0)),
            transforms.Resize(ADNI_IMAGE_DIMENSIONS),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(ADNI_IMAGE_DIMENSIONS),
            transforms.ToTensor(),
        ]
    )

    train_loader = get_dataloader(
        path.join(root_dir, "train"),
        train_batch_size,
        shuffle=True,
        transform=train_transform,
    )
    test_loader = get_dataloader(
        path.join(root_dir, "test"),
        test_batch_size,
        shuffle=False,
        transform=test_transform,
    )
    return train_loader, test_loader
