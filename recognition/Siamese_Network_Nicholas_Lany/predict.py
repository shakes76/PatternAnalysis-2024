import torch
import os
import pandas as pd
import kagglehub
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import CNN, SiameseNetwork
from dataset import ISICDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prediction_threshold = 0.5

def load_model(model_path):
    siamese_model = SiameseNetwork().to(device)
    siamese_model.load_state_dict(torch.load(model_path, map_location=device))
    return siamese_model.cnn 


def classify_images(model, dataset, transform=None, num_samples=100):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in dataloader:
            if total_images >= num_samples:
                break

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct_predictions += (predicted == labels).sum().item()
            total_images += labels.size(0)

    accuracy = correct_predictions / total_images
    print(f'Accuracy of the model on the test dataset: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
    dataset_image_path = os.path.join(dataset_path, "train-image/image")
    meta_data_path = os.path.join(dataset_path, "train-metadata.csv")
    model_save_path = os.path.join(os.getcwd(), 'model.pth')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])

    isic_dataset = ISICDataset(
        dataset_path=dataset_image_path,
        metadata_path=meta_data_path,
        transform=transform,
        augment_transform=augment_transform,
        num_augmentations=5
    )

    model = load_model(model_save_path)
    model.eval()
    classify_images(model, isic_dataset, transform=transform)
