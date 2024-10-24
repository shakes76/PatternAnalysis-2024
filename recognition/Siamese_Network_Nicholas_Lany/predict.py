import torch
import os
import pandas as pd
import kagglehub
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import CNN
from dataset import ISICDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prediction_threshold = 0.5

def load_model(model_path):
    model = CNN(shape=(256, 256), num_classes=2).to(device)  # Load CNN model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

def classify_images(model, dataset, transform=None):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    correct_predictions = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in dataloader:
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

    isic_dataset = ISICDataset(dataset_path=dataset_image_path, metadata_path=meta_data_path, transform=transform)

    model = load_model(model_save_path)
    classify_images(model, isic_dataset, transform=transform)
