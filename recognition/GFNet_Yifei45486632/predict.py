import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomImageDataset  # 假设自定义数据集文件名为 custom_dataset.py
import os
from modules import GFNet  
from functools import partial
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch = 4
test_data_dir = "./test"  
checkpoint_path = f"./models/model_epoch_{epoch}.pth" 

# Data transformation (note that the test set does not require data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the test dataset
test_dataset = CustomImageDataset(directory=test_data_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the model
model = GFNet(
    img_size=224, 
    patch_size=16, in_chans=3, num_classes=2,
    embed_dim=256, depth=10, mlp_ratio=4, drop_path_rate=0.15,
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
)

model = nn.parallel.DataParallel(model)
model.to(device)

model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
model.eval()

# Test model
correct_predictions = 0
total_samples = 0
# Show progress bar
with torch.no_grad():
    progress_bar = tqdm(test_loader, desc=f"Testing Epoch {epoch}", unit="batch")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward propagation
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate accuracy
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Update the accuracy in the progress bar description
        accuracy = correct_predictions / total_samples
        progress_bar.set_postfix(accuracy=f"{accuracy:.4f}")

    # Output the final test accuracy
    final_accuracy = correct_predictions / total_samples
    print(f"Final Test Accuracy at Epoch {epoch}: {final_accuracy:.4f}")