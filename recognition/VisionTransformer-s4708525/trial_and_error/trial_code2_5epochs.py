from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch
import torch.nn as nn
from path import *
import time
import torch.optim as optim

# Set device to CUDA if available, else fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ADNI_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, sub_dir in enumerate(['NC', 'AD']):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            for image_name in os.listdir(sub_dir_path):
                self.image_paths.append(os.path.join(sub_dir_path, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # image = Image.open(self.image_paths[idx]).convert('L')  # Convert to grayscale
        image = Image.open(self.image_paths[idx]).convert('RGB')  # 
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Define Convolutional Patch Embedding
class ConvEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, kernel_size=7, stride=4, padding=2):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-Attention
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output  # Residual connection
        x = self.norm1(x)
        
        # Feedforward MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output  # Residual connection
        x = self.norm2(x)
        return x

# CvT Model
class CvT(nn.Module):
    def __init__(self, img_size=224, in_channels=3, num_classes=2, embed_dim=64, num_heads=8, mlp_dim=128, num_transformer_blocks=4, dropout=0.1):
        super(CvT, self).__init__()
        
        # Convolutional embedding layer
        self.conv_embedding = ConvEmbedding(in_channels, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        # Convolutional embedding
        x = self.conv_embedding(x)  # Shape: (batch_size, embed_dim, H/stride, W/stride)
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Classification head (use the average of all patch embeddings for final prediction)
        x = x.mean(dim=1)  # Global average pooling over patches
        return self.mlp_head(x)
    
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total}%')

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}%')



if __name__ == "__main__":

        # Define the transform for images (resizing and normalization)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),  # Data augmentation: random flip
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard values for RGB # Modify mean and std if necessary
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard values for RGB
    ])


    train_dataset = ADNI_Dataset(train_path_server, transform=transform)
    test_dataset = ADNI_Dataset(test_path_server, transform=transform_test)


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # # Implement a learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Instantiate the CvT model
    model = CvT(img_size=224, in_channels=3, num_classes=2, embed_dim=64, num_heads=8, mlp_dim=128, num_transformer_blocks=4, dropout=0.1)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print('Start training')
    start_time = time.time()
    train_model(model, train_loader, criterion, optimizer)
    end_time = time.time()

    total_training_time = end_time - start_time

    minutes, seconds = divmod(total_training_time, 60)
    print('Training time: %d minutes, %d seconds' % (minutes, seconds))
    print()
    print('End training')
    print('Eval start!')
    evaluate_model(model, test_loader)
    print('Eval end!')
