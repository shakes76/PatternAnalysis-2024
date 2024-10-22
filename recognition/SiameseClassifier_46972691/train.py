# train.py
# Trains the Siamese Network and then the ImageClassifier using the learned embeddings.
# Author: Harrison Martin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from modules import SiameseNetwork, ImageClassifier
from dataset import SiameseDataset, ImageDataset, split_dataset
import pandas as pd

def main():
    # Load metadata
    metadata_df = pd.read_csv('recognition/SiameseClassifier_46972691/test_dataset_2020_Kaggle/train-metadata.csv')

    # Split the dataset
    train_df, val_df, test_df = split_dataset(metadata_df)

    # Define transformations
    basic_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure all images are the same size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Image path
    image_path = 'recognition/SiameseClassifier_46972691/test_dataset_2020_Kaggle/train-image/image'

    # Create datasets
    # For Siamese Network
    train_dataset_siam = SiameseDataset(image_folder=image_path, df=train_df, transform=basic_transforms)
    val_dataset_siam = SiameseDataset(image_folder=image_path, df=val_df, transform=basic_transforms)

    # For Image Classification
    train_dataset_cls = ImageDataset(image_folder=image_path, df=train_df, transform=basic_transforms)
    val_dataset_cls = ImageDataset(image_folder=image_path, df=val_df, transform=basic_transforms)
    test_dataset_cls = ImageDataset(image_folder=image_path, df=test_df, transform=basic_transforms)

    # Create DataLoaders
    train_loader_siam = DataLoader(train_dataset_siam, batch_size=32, shuffle=True, num_workers=5)
    val_loader_siam = DataLoader(val_dataset_siam, batch_size=32, shuffle=False, num_workers=5)

    train_loader_cls = DataLoader(train_dataset_cls, batch_size=32, shuffle=True, num_workers=5)
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=32, shuffle=False, num_workers=5)
    test_loader_cls = DataLoader(test_dataset_cls, batch_size=32, shuffle=False, num_workers=5)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train Siamese Network
    siamese_model = SiameseNetwork().to(device)
    criterion_siam = nn.BCEWithLogitsLoss()
    optimizer_siam = optim.Adam(siamese_model.parameters(), lr=0.001)
    num_epochs = 1

    print("Training Siamese Network...")
    train_siamese(siamese_model, train_loader_siam, val_loader_siam, criterion_siam, optimizer_siam, device, num_epochs)

    # Save the embedding network's state_dict
    torch.save(siamese_model.embedding_net.state_dict(), 'embedding_net.pth')

    # Load the trained embeddings into the ImageClassifier
    classifier_model = ImageClassifier().to(device)
    classifier_model.embedding_net.load_state_dict(torch.load('embedding_net.pth'))

    # Optionally freeze embedding layers
    # for param in classifier_model.embedding_net.parameters():
    #     param.requires_grad = False

    # Define criterion and optimizer for classifier
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer_cls = optim.Adam(classifier_model.parameters(), lr=0.001)

    print("Training Image Classifier...")
    train_classifier(classifier_model, train_loader_cls, val_loader_cls, criterion_cls, optimizer_cls, device, num_epochs)

    # Evaluate on test set
    print("Evaluating on Test Set...")
    evaluate_classifier(classifier_model, test_loader_cls, criterion_cls, device)

def train_siamese(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        i = 0
        for img1, img2, labels in train_loader:
            print(i)
            i+=1
            if i == 10:
                break
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img1.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Siamese Loss: {epoch_loss:.4f}")

        # Validate the model
        validate_siamese(model, val_loader, criterion, device)

def validate_siamese(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        i = 0
        for img1, img2, labels in val_loader:
            print(i)
            i+=1
            if i == 10:
                break
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * img1.size(0)

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / total
    print(f"Siamese Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

def train_classifier(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        i = 0
        for images, labels in train_loader:
            print(i)
            i+=1
            if i == 10:
                break
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Classifier Loss: {epoch_loss:.4f}")

        # Validate the model
        validate_classifier(model, val_loader, criterion, device)

def validate_classifier(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        i = 0
        for images, labels in val_loader:
            print(i)
            i+=1
            if i == 10:
                break
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / total
    print(f"Classifier Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

def evaluate_classifier(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        i = 0
        for images, labels in test_loader:
            print(i)
            i+=1
            if i == 10:
                break
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    

if __name__ == '__main__':
    main()
