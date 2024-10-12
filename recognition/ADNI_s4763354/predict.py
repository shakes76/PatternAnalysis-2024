import torch
from torchvision import transforms
from modules_BEST import GFNet, GFNetPyramid
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import random
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader

def predict_and_visualize(model, image_path, class_names=['NC', 'AD']):
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image_tensor = transform(image)
    
    # Add normalization after ensuring 3 channels
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_tensor = normalize(image_tensor)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
    
    predicted_class = class_names[predicted.item()]
    confidence = probabilities[0][predicted.item()].item()

    # Visualize the image and prediction
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Input Image\nPredicted: {predicted_class} ({confidence:.2%})')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    bars = plt.bar(class_names, probabilities[0].cpu().numpy())
    plt.title('Class Probabilities')
    plt.ylabel('Probability')
    plt.ylim(0, 1)

    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.2%}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('predict/predict_image.png')

def plot_confusion_matrix(all_labels, all_preds, class_names=['NC', 'AD']):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('predict/confusion_matrix.png')

def main(model_path, test_dir, num_samples=5):
    # Load the model
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GFNetPyramid(
        img_size=224, 
        patch_size=4, 
        num_classes=2,  
        embed_dim=[96, 192, 384, 768],
        depth=[2, 2, 10, 2],
        mlp_ratio=[4, 4, 4, 4],
        drop_path_rate=0.3,
    )
    
    num_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Get test data loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    all_preds = []
    all_labels = []
    all_image_paths = []

    # Run predictions on all test images
    for images, labels in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    # Correctly handle the image paths
    all_image_paths = [os.path.join(test_dir, sample[0].split('/')[-2], sample[0].split('/')[-1]) for sample in test_loader.dataset.samples]

    # Visualize a few random samples
    sample_indices = random.sample(range(len(all_image_paths)), num_samples)
    for idx in sample_indices:
        predict_and_visualize(model, all_image_paths[idx])

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds)

    # Print overall accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    print(f"Overall Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main(model_path='best_newmod_pretrain/new_mod_hb_adamw_cosine.pth', test_dir='/home/groups/comp3710/ADNI/AD_NC/test')
