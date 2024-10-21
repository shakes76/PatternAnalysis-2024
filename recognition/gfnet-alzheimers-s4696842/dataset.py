from os import path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# (Height, Width)
ADNI_IMAGE_DIMENSIONS = (256, 240)


def get_dataloader(root_dir, batch_size, shuffle, transform):
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_train_test_dataloaders(root_dir, train_batch_size, test_batch_size):
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1)),
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
