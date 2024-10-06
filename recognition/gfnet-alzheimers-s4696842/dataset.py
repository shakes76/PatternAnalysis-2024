from os import path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

ADNI_IMAGE_DIMENSIONS = (256, 240)


def get_dataloader(root_dir, batch_size, shuffle):
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(ADNI_IMAGE_DIMENSIONS),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_train_test_dataloaders(root_dir, train_batch_size, test_batch_size):
    train_loader = get_dataloader(
        path.join(root_dir, "train"), train_batch_size, shuffle=True
    )
    test_loader = get_dataloader(
        path.join(root_dir, "test"), test_batch_size, shuffle=False
    )
    return train_loader, test_loader
