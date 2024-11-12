import os
import torch
import numpy as np
import random
import argparse
from modules import UNet3D
from dataset import CustomDataset  # 确保这里是正确的类名
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import torchio as tio

# Set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001)
parser.add_argument('--epoch', default=20)
parser.add_argument('--device', default='cuda')
parser.add_argument('--loss', default='dice')
parser.add_argument('--dataset_root', type=str,
                    default=r'C:\Users\111\Desktop\3710\新建文件夹\数据集\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1',
                    help='Root directory of the dataset')
args = parser.parse_args()

# Define the model
model = UNet3D(in_channels=1, out_channels=6).to(args.device)

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape, f"Shapes don't match {inputs.shape} != {targets.shape}"

        # Skip background class
        inputs = inputs[:, 1:]
        targets = targets[:, 1:]

        # Sum over elements per sample and per class
        axes = tuple(range(2, len(inputs.shape)))  # 这里的范围从2开始，适应5D张量
        intersection = torch.sum(inputs * targets, axes)
        addition = torch.sum(torch.square(inputs) + torch.square(targets), axes)

        # 计算Dice损失
        dice_score = (2 * intersection + self.smooth) / (addition + self.smooth)
        return 1 - torch.mean(dice_score)

criterion = DiceLoss().to(args.device)

# Define the data augmentation class
class Augment:
    def __init__(self):
        self.shrink = tio.CropOrPad((16, 32, 32))
        self.flip = tio.transforms.RandomFlip(0, flip_probability=0.5)

    def __call__(self, image, mask):
        image = self.shrink(image)
        mask = self.shrink(mask)
        image = self.flip(image)
        mask = self.flip(mask)
        return image, mask

# Define the train and test dataloaders
train_dataset = CustomDataset(mode='train', dataset_path=args.dataset_root)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_dataset = CustomDataset(mode='test', dataset_path=args.dataset_root)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

# Initialize data augmentation
augment = Augment()

# The training and validation process
start_time = time.time()
for epoch in range(args.epoch):
    model.train()
    train_epoch_loss = []

    for idx, (data_x, data_y) in enumerate(train_dataloader):
        data_x = data_x.to(torch.float32).to(args.device)
        data_y = data_y.to(torch.float32).to(args.device)

        # Ensure data_x is 5D
        if data_x.dim() == 4:  # If it's a 4D tensor
            data_x = data_x.unsqueeze(1)  # Add a channel dimension

        data_x, data_y = augment(data_x, data_y)  # Apply augmentation

        # Ensure data_y is 5D
        if data_y.dim() == 4:  # If it's a 4D tensor
            data_y = data_y.unsqueeze(1)  # Add a channel dimension

        labely = torch.nn.functional.one_hot(data_y.squeeze(1).long(), num_classes=6).permute(0, 4, 1, 2, 3).float().to(args.device)
        outputs = model(data_x)
        optimizer.zero_grad()
        loss = criterion(outputs, labely)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())

    train_epochs_loss.append(np.average(train_epoch_loss))
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch}: Train Loss: {train_epochs_loss[-1]:.4f}')

    if epoch % 1 == 0:
        model.eval()
        valid_epoch_loss = []
        with torch.no_grad():
            for idx, (data_x, data_y) in enumerate(test_dataloader):
                data_x = data_x.to(torch.float32).to(args.device)
                data_y = data_y.to(torch.float32).to(args.device)

                # Ensure data_x is 5D
                if data_x.dim() == 4:  # If it's a 4D tensor
                    data_x = data_x.unsqueeze(1)  # Add a channel dimension

                # Ensure data_y is 5D
                if data_y.dim() == 4:  # If it's a 4D tensor
                    data_y = data_y.unsqueeze(1)  # Add a channel dimension

                labely = torch.nn.functional.one_hot(data_y.squeeze(1).long(), num_classes=6).permute(0, 4, 1, 2, 3).float().to(args.device)
                outputs = model(data_x)
                loss = criterion(outputs, labely)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())

        valid_epochs_loss.append(np.average(valid_epoch_loss))
        # Save the trained model
        torch.save(model.state_dict(), f'epoch_{epoch}_loss{args.loss}.pth')

# Plotting the training and validation loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Training loss
ax1.plot(train_loss, label='Train Loss (Dice)', color='blue', linewidth=2)
ax1.set_title("Train Loss (Dice)", fontsize=16)  
ax1.set_xlabel("Iterations", fontsize=14)
ax1.set_ylabel("Loss", fontsize=14)
ax1.legend()
ax1.grid(True)

# Right plot: Training and validation loss comparison
ax2.plot(np.arange(0, len(train_epochs_loss)), train_epochs_loss, '-o', label='Epoch Train Loss', color='orange', markersize=4)
ax2.plot(np.arange(0, len(valid_epochs_loss)), valid_epochs_loss, '-o', label='Epoch Valid Loss', color='green', markersize=4)
ax2.set_title("Train and Validation Loss", fontsize=16)
ax2.set_xlabel("Epochs", fontsize=14)
ax2.set_ylabel("Loss", fontsize=14)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(f"train_loss_and_valid_loss.png")
plt.show()