import torch
from torchvision import transforms
from modules import GFNet, GFNetPyramid
import os
import random
import torch.nn as nn
from torchvision import datasets
from PIL import Image
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
        
    # Print overall accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    print(f"Overall Accuracy: {accuracy:.2%}")

    # Visualize a few random samples
    sample_indices = random.sample(range(len(all_image_paths)), num_samples)
    for idx in sample_indices:
        predict_and_visualize(model, all_image_paths[idx])

if __name__ == "__main__":
    main(model_path='best_newmod_pretrain/new_mod_hb_adamw_cosine.pth', test_dir='/home/groups/comp3710/ADNI/AD_NC/test')

