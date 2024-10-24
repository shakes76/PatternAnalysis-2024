import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, init_features=32):
        super(UNet2D, self).__init__()

        # Number of features in the first layer
        features = init_features

        # Encoder
        self.encoder1 = UNet2D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet2D._block(
            features * 8, features * 16, name="bottleneck")

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2D._block(
            (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2D._block(
            (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2D._block(
            (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet2D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features,
                              out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.softmax(self.conv(dec1), dim=1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True)
        )


def dice_score(y_pred, y_true, smooth=1.0):
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)

    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(
        ), f"Shape mismatch: {y_pred.size()} != {y_true.size()}"

        batch_size = y_pred.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.contiguous().view(batch_size, num_classes, -1)
        y_true = y_true.contiguous().view(batch_size, num_classes, -1)

        intersection = (y_pred * y_true).sum(2)
        union = y_pred.sum(2) + y_true.sum(2)
        dsc = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1. - dsc.mean()


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
