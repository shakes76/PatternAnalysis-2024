import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from scipy.ndimage import zoom

# Load Nifti files
def load_nifti_file(filepath):
    img = nib.load(filepath)
    img_data = img.get_fdata()
    return img_data

# Dice Similarity Coefficient
def dice_coefficient(pred, target, smooth=1e-5):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Custom dataset for loading Nifti files
class ProstateDataset(Dataset):
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = load_nifti_file(self.image_files[idx])
        mask = load_nifti_file(self.mask_files[idx])
        
        # Resize to 256x256 if necessary
        if image.shape != (256, 256):
            image = zoom(image, (256 / image.shape[0], 256 / image.shape[1]), order=1)
            mask = zoom(mask, (256 / mask.shape[0], 256 / mask.shape[1]), order=0)
        
        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32)
        return image, mask

# UNet model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        mid = self.middle(enc)
        dec = self.decoder(mid)
        return dec

# Train the model
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.unsqueeze(1).float()
            masks = masks.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

# Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.unsqueeze(1).float()
            masks = masks.unsqueeze(1).float()

            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()

            dice = dice_coefficient(outputs, masks)
            dice_scores.append(dice.item())

    avg_dice = np.mean(dice_scores)
    print(f"Average Dice Score: {avg_dice}")
    return avg_dice

# Data Preparation
image_files = ['path_to_image1.nii', 'path_to_image2.nii']  # Replace with actual file paths
mask_files = ['path_to_mask1.nii', 'path_to_mask2.nii']  # Replace with actual file paths

# Split into training and testing sets
train_images, test_images, train_masks, test_masks = train_test_split(image_files, mask_files, test_size=0.2)

train_dataset = ProstateDataset(train_images, train_masks, transform=ToTensor())
test_dataset = ProstateDataset(test_images, test_masks, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model, loss function, optimizer
model = UNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, num_epochs=25)
evaluate_model(model, test_loader)
