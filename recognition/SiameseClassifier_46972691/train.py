# train.py
# Trains the Siamese Network and then the ImageClassifier using the learned embeddings.
# Author: Harrison Martin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from modules import SiameseNetwork, ImageClassifier
from dataset import SiameseDataset, ImageDataset, split_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from datetime import datetime

def main():
    # Create 'results' directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Create a timestamped subdirectory within 'results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Load metadata
    metadata_df = pd.read_csv('recognition/SiameseClassifier_46972691/test_dataset_2020_Kaggle/train-metadata.csv')

    # Split the dataset
    train_df, val_df, test_df = split_dataset(metadata_df)

    # Oversample the minority class in the training set
    train_df_balanced = oversample_minority_class(train_df)

    # Calculate class weights based on the balanced training data
    class_weights = calculate_class_weights(train_df_balanced)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define data augmentation transforms for training
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    # Validation and test transforms (no augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Image path
    image_path = 'recognition/SiameseClassifier_46972691/test_dataset_2020_Kaggle/train-image/image'

    # Create datasets
    # For Siamese Network
    train_dataset_siam = SiameseDataset(image_folder=image_path, df=train_df_balanced, transform=train_transforms)
    val_dataset_siam = SiameseDataset(image_folder=image_path, df=val_df, transform=val_test_transforms)

    # For Image Classification
    train_dataset_cls = ImageDataset(image_folder=image_path, df=train_df_balanced, transform=train_transforms)
    val_dataset_cls = ImageDataset(image_folder=image_path, df=val_df, transform=val_test_transforms)
    test_dataset_cls = ImageDataset(image_folder=image_path, df=test_df, transform=val_test_transforms)

    # Create DataLoaders
    train_loader_siam = DataLoader(train_dataset_siam, batch_size=64, shuffle=True, num_workers=4)
    val_loader_siam = DataLoader(val_dataset_siam, batch_size=64, shuffle=False, num_workers=4)

    train_loader_cls = DataLoader(train_dataset_cls, batch_size=64, shuffle=True, num_workers=4)
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=64, shuffle=False, num_workers=4)
    test_loader_cls = DataLoader(test_dataset_cls, batch_size=64, shuffle=False, num_workers=4)

    # Initialise models
    siamese_model = SiameseNetwork().to(device)
    classifier_model = ImageClassifier().to(device)

    # Define loss functions with class weights for imbalance handling
    criterion_siam = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
    optimizer_siam = optim.Adam(siamese_model.parameters(), lr=0.0001)  # Adjusted learning rate if necessary

    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].to(device))
    optimizer_cls = optim.Adam(classifier_model.parameters(), lr=0.0001)  # Adjusted learning rate if necessary

    num_epochs = 100  # Adjust number of epochs if needed

    # Train Siamese Network
    print("Training Siamese Network...")
    train_siamese(siamese_model, train_loader_siam, val_loader_siam, criterion_siam, optimizer_siam, device, num_epochs)

    # Save the Siamese model
    torch.save(siamese_model.state_dict(), os.path.join(results_dir, 'siamese_network.pth'))

    # Train Image Classifier
    print("Training Image Classifier...")
    train_classifier(classifier_model, train_loader_cls, val_loader_cls, criterion_cls, optimizer_cls, device, num_epochs)

    # Save the trained classifier model
    torch.save(classifier_model.state_dict(), os.path.join(results_dir, 'image_classifier.pth'))

    # Evaluate on test set
    print("Evaluating on Test Set...")
    evaluate_classifier(classifier_model, test_loader_cls, criterion_cls, device, results_dir)  # Pass results_dir to the function

def oversample_minority_class(df):
    # Separate majority and minority classes
    df_majority = df[df['target'] == 0]
    df_minority = df[df['target'] == 1]

    # Oversample the minority class
    df_minority_oversampled = resample(df_minority,
                                       replace=True,  # Sample with replacement
                                       n_samples=len(df_majority),  # Match number of majority class samples
                                       random_state=42)

    # Combine majority class with oversampled minority class
    df_balanced = pd.concat([df_majority, df_minority_oversampled])

    # Shuffle the dataframe
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

def calculate_class_weights(df):
    class_counts = df['target'].value_counts().sort_index()
    total_samples = len(df)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights.values, dtype=torch.float32)

def train_siamese(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for img1, img2, labels in train_loader:
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
        for img1, img2, labels in val_loader:
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

        for images, labels in train_loader:
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
        for images, labels in val_loader:
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

def evaluate_classifier(model, test_loader, criterion, device, results_dir):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Collect all labels and outputs for computing metrics
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)

    # Apply sigmoid to outputs
    all_outputs_sigmoid = torch.sigmoid(torch.from_numpy(all_outputs)).numpy()

    # Compute ROC AUC score
    roc_auc = roc_auc_score(all_labels, all_outputs_sigmoid)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs_sigmoid)

    # Plot and save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    # Threshold outputs to get binary predictions
    preds = (all_outputs_sigmoid >= 0.5).astype(int)

    # Compute accuracy
    correct = (preds == all_labels).sum()
    total = len(all_labels)
    test_acc = correct / total

    # Classification report
    class_report = classification_report(all_labels, preds, target_names=['Benign', 'Malignant'])
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(class_report)

    # Save classification report to a text file
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, preds)

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == '__main__':
    main()