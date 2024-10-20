from modules import UNet
from dataset import MedicalImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dice coefficient function
def dice_coefficient(pred, target, epsilon=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)
    return dice

# Training loop
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=50):
    model.train()
    print("> Training")

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_dice = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Sigmoid for BCELoss
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            dice_score = dice_coefficient(outputs, labels)
            running_dice += dice_score.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.5f}, Dice: {dice_score:.5f}")

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.5f}, Average Dice: {avg_dice:.5f}')


# Initialize model, criterion, optimizer, and scheduler
model = UNet().to(device)

# Dataset
image_dir = r'HipMRI_study_keras_slices_data/keras_slices_seg_train'
train_dataset = MedicalImageDataset(image_dir=image_dir, normImage=True, load_type='2D')
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4)

# Loss, optimizer, and learning rate scheduler
criterion = nn.BCELoss()  # Or CrossEntropyLoss for multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=50)

# Save the trained model
save_path = './model.pth'
torch.save(model.state_dict(), save_path)
