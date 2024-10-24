import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CropBrainRegion:
    """
    Cut off the black area of brain image use cv2.
    """
    def __call__(self, img):
        img = np.array(img)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_img = img[y:y + h, x:x + w]
        return Image.fromarray(cropped_img)


def get_data_loaders(train_dir, val_dir, test_dir, batch_size=32):
    """
    Do data augmentation and put the data into Dataloader.
    """
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(25),
        CropBrainRegion(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1174], std=[0.2163])
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(25),
        CropBrainRegion(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1174], std=[0.2163])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
