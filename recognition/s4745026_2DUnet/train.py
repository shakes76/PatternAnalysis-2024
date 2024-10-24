import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import DiceLoss, UNet2D
import time
import os
# from dataset import load_data_3D
# Training code for the 2D U-Net
num_epochs = 3
learning_rate = 0.001

# Instantiate the model
model = UNet2D(in_channels=1, out_channels=2).cuda()

# Loss function and optimizer
loss_func = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

# Learning rate scheduler
sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.005, max_lr=learning_rate, step_size_up=15, step_size_down=15, mode='triangular', verbose=False
)
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=0.001, verbose=False
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30]
)

train_loader = ""
validate_loader = ""
test_loader = ""


def dice_score(y_pred, y_true, smooth=1.0):
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)

    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice


print("> Training")
start = time.time()
for epoch in range(num_epochs):
    model.train()
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.cuda(), masks.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, masks)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.5f}")

    # Update learning rate
    scheduler.step()

    model.eval()
    with torch.no_grad():
        dice_scores = []
        for val_images, val_masks in validate_loader:
            val_images, val_masks = val_images.cuda(), val_masks.cuda()
            val_outputs = model(val_images)
            dice = dice_score(val_outputs, val_masks)
            dice_scores.append(dice)
        avg_dice_score = sum(dice_scores) / len(dice_scores)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Validation Dice Score: {avg_dice_score:.4f}")

end = time.time()
elapsed = end - start
print(f"Training took {elapsed:.2f} secs or {elapsed / 60:.2f} mins in total.")

# Testing and visualization
model.eval()
with torch.no_grad():
    for test_images, test_masks in test_loader:
        test_images = test_images.cuda()
        test_outputs = model(test_images)
        predicted_masks = torch.argmax(test_outputs, dim=1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(test_images[0].cpu().squeeze(), cmap='gray')
        plt.title("Input Image")

        plt.subplot(1, 3, 2)
        plt.imshow(test_masks[0].cpu().squeeze(), cmap='gray')
        plt.title("Segmentation Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_masks[0].cpu().squeeze(), cmap='gray')
        plt.title("Predicted Mask")

        plt.show()
        break
