import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Encoder (Reduced depth)
        self.conv1 = conv_block(1, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)

        # Bottleneck (Remove)
        self.bottleneck = conv_block(256, 512)

        # Decoder (Reduced depth)
        self.upconv3 = up_block(512, 256)
        self.conv3_1 = conv_block(512, 256)

        self.upconv2 = up_block(256, 128)
        self.conv2_1 = conv_block(256, 128)

        self.upconv1 = up_block(128, 64)
        self.conv1_1 = conv_block(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, 6, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(nn.MaxPool2d(2)(conv1))
        conv3 = self.conv3(nn.MaxPool2d(2)(conv2))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(conv3))

        # Decoder
        upconv3 = self.upconv3(bottleneck)
        conv3_1 = self.conv3_1(torch.cat([upconv3, conv3], dim=1))

        upconv2 = self.upconv2(conv3_1)
        conv2_1 = self.conv2_1(torch.cat([upconv2, conv2], dim=1))

        upconv1 = self.upconv1(conv2_1)
        conv1_1 = self.conv1_1(torch.cat([upconv1, conv1], dim=1))

        output = self.out_conv(conv1_1)
        return output
    
def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    if pred_flat.sum() == 0 and target_flat.sum() == 0:
        return 1.0  # If both are empty, Dice score is 1
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)