import torch
import os
import pandas as pd
import kagglehub
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import SiameseNetwork, ContrastiveLoss
from dataset import ISICDataset, SiameseDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prediction_threshold = 0.5

def evaluate_model(dataset, model_path, transform=None):
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    dataloader = DataLoader(SiameseDataset(dataset, transform=transform, num_pairs=1000), batch_size=16, shuffle=False)
    
    correct_predictions = 0
    total_pairs = 0

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            similarity_scores, _ = model(img1, img2)

            predicted_labels = (similarity_scores > prediction_threshold).float() 
            predicted_classes = predicted_labels.argmax(dim=1)

            correct_predictions += (predicted_classes == labels).sum().item()
            total_pairs += labels.size(0)

    accuracy = correct_predictions / total_pairs
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

    isic_dataset = ISICDataset(dataset_path=dataset_path, metadata_path=meta_data_path)

    evaluate_model(dataset=isic_dataset, model_path=model_save_path, transform=transform)
