# modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=6):
        super(UNet, self).__init__()

        # Down path
        self.encoder_conv1 = self.conv_block(1, 64)
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv2 = self.conv_block(64, 128)
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv3 = self.conv_block(128, 256)
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv4 = self.conv_block(256, 512)
        self.encoder_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = self.conv_block(512, 1024)

        # Up path
        self.upconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv6 = self.conv_block(1024, 512)

        self.upconv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv7 = self.conv_block(512, 256)

        self.upconv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv8 = self.conv_block(256, 128)

        self.upconv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv9 = self.conv_block(128, 64)

        # Output
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Double convolution block with ReLU activation.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Down Path
        c1 = self.encoder_conv1(x)
        p1 = self.encoder_pool1(c1)

        c2 = self.encoder_conv2(p1)
        p2 = self.encoder_pool2(c2)

        c3 = self.encoder_conv3(p2)
        p3 = self.encoder_pool3(c3)

        c4 = self.encoder_conv4(p3)
        p4 = self.encoder_pool4(c4)

        # Bottleneck
        c5 = self.bottleneck_conv(p4)

        # Up Path
        u6 = self.upconv6(c5)
        concat6 = torch.cat([u6, c4], dim=1)
        c6 = self.decoder_conv6(concat6)

        u7 = self.upconv7(c6)
        concat7 = torch.cat([u7, c3], dim=1)
        c7 = self.decoder_conv7(concat7)

        u8 = self.upconv8(c7)
        concat8 = torch.cat([u8, c2], dim=1)
        c8 = self.decoder_conv8(concat8)

        u9 = self.upconv9(c8)
        concat9 = torch.cat([u9, c1], dim=1)
        c9 = self.decoder_conv9(concat9)

        # Output Layer
        output = self.output_conv(c9)

        return output

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=6, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: (B, C, H, W) - Raw logits from the network.
        targets: (B, H, W) - Ground truth labels with class indices (0 to num_classes - 1).
        """
        # Ensure targets have shape (B, H, W)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # Remove channel dimension
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Apply softmax to get class probabilities
        inputs_softmax = F.softmax(inputs, dim=1)
        
        # Compute Dice loss for each class
        intersection = (inputs_softmax * targets_one_hot).sum(dim=(0, 2, 3))
        cardinality = (inputs_softmax + targets_one_hot).sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Compute mean Dice loss over classes
        dice_loss = 1.0 - dice_per_class.mean()
        return dice_loss