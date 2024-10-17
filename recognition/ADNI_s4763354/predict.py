'''
Author: Lok Yee Joey Cheung 
This file is created as a test script to produce test output of the trained model, with visualizations. 
'''
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import random
from modules import GFNetPyramid, GFNet
import torch.nn as nn

def show_example_images(model, test_loader, num_examples=5, class_names=['NC', 'AD']):
    model.eval()
    all_examples = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            for img, pred, prob, label in zip(images, preds, probabilities, labels):
                all_examples.append((img, pred, prob, label))
    
    # Randomly select num_examples from all_examples
    selected_examples = random.sample(all_examples, min(num_examples, len(all_examples)))
    
    # Plot examples
    fig, axes = plt.subplots(1, len(selected_examples), figsize=(20, 4))
    fig.suptitle('Example Classifications', fontsize=16)
    
    for i, (img, pred, prob, label) in enumerate(selected_examples):
        # Denormalize the image
        mean = torch.tensor([0.485]).view(1, 1, 1).to(device)
        std = torch.tensor([0.229]).view(1, 1, 1).to(device)
        img = img * std + mean
        
        img = img[0].cpu().numpy()
        
        axes[i].imshow(img, cmap='gray')
        
        predicted_class = class_names[pred.item()]
        true_class = class_names[label.item()]
        confidence = prob[pred.item()].item()
        
        title = f'Pred: {predicted_class}\nTrue: {true_class}\nConf: {confidence:.2f}'
        axes[i].set_title(title, color='green' if pred == label else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predict/example_classifications.png')
    plt.close()

def plot_confusion_matrix(all_labels, all_preds, class_names=['NC', 'AD']):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('predict/confusion_matrix.png')
    plt.close()

def main(model_path, test_dir, num_samples=5):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the model
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

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Ensure 3 channels
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Create the dataset and data loader
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    all_preds = []
    all_labels = []
    model.eval()

    # Run predictions on all test images
    for images, labels in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Show example images
    show_example_images(model, test_loader, num_samples)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds)

    # Print overall accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    print(f"Overall Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main(model_path='best_newmod_pretrain/new_mod_hb_adamw_cosine.pth', test_dir='/home/groups/comp3710/ADNI/AD_NC/test')
