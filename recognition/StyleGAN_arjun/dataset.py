import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

RANGPUR_PATH = "/home/groups/comp3710/" # Dataset path on Rangpur Cluster
PATH = '/Volumes/Acasis WD_Black/Documents/Deep Learning/Datasets/' # Dataset path on personal machine

# -- Dataset Attributes --
ADNI_TRAIN_PATH = "ADNI/AD_NC/train"
ADNI_TEST_PATH = "ADNI/AD_NC/test"
CIFAR_PATH = "cifar10/"

ADNI_IMG_SIZE = 240
CIFAR_IMG_SIZE = 32

def process_adni(batch_size, rangpur=False):
    """
    Returns dataset and dataloader for ADNI (For Image Generation).
    """
    transform = transforms.Compose([
        transforms.Resize((ADNI_IMG_SIZE, ADNI_IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Determine the correct path based on the rangpur flag
    base_path = RANGPUR_PATH if rangpur else PATH
    train_set = datasets.ImageFolder(base_path + ADNI_TRAIN_PATH, transform=transform)
    test_set = datasets.ImageFolder(base_path + ADNI_TEST_PATH, transform=transform)

    dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataset, dataloader

def process_cifar(batch_size, rangpur=False):
    """
    Returns dataset and dataloader for CIFAR10 (For Image Generation).
    """
    transform = transforms.Compose([
        transforms.Resize((CIFAR_IMG_SIZE, CIFAR_IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Determine the correct path based on the rangpur flag
    base_path = RANGPUR_PATH if rangpur else PATH
    train_set = datasets.CIFAR10(base_path + CIFAR_PATH, transform=transform, train=True, download=True)
    test_set = datasets.CIFAR10(base_path + CIFAR_PATH, transform=transform, train=True, download=True)

    dataset = torch.utils.data.ConcatDataset([train_set, test_set])

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataset, dataloader
