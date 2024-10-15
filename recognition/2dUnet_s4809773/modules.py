import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        
        # Encoder part (downsampling)
        self.conv1 = nn.Conv2d(1, 64, 3, padding="same")  # input: 1 channel (e.g. grayscale image), 64 filters
        self.conv2 = nn.Conv2d(64, 64, 3, padding="same")
        self.pool1 = nn.MaxPool2d(2)  # Downsample by a factor of 2

        self.conv3 = nn.Conv2d(64, 128, 3, padding="same")
        self.conv4 = nn.Conv2d(128, 128, 3, padding="same")
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding="same")
        self.conv6 = nn.Conv2d(256, 256, 3, padding="same")
        self.pool3 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(256, 512, 3, padding="same")
        self.conv8 = nn.Conv2d(512, 512, 3, padding="same")
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck part
        self.conv9 = nn.Conv2d(512, 1024, 3, padding="same")
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding="same")
        
        # Decoder part (upsampling)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # Upsample
        self.conv11 = nn.Conv2d(1024, 512, 3, padding="same")
        self.conv12 = nn.Conv2d(512, 512, 3, padding="same")

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(512, 256, 3, padding="same")
        self.conv14 = nn.Conv2d(256, 256, 3, padding="same")

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(256, 128, 3, padding="same")
        self.conv16 = nn.Conv2d(128, 128, 3, padding="same")

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv17 = nn.Conv2d(128, 64, 3, padding="same")
        self.conv18 = nn.Conv2d(64, 64, 3, padding="same")

        # Final output layer
        self.output_conv = nn.Conv2d(64, num_classes, 1)  # 1x1 convolution to reduce to the number of classes
        
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        p1 = self.pool1(x1)

        x2 = F.relu(self.conv3(p1))
        x2 = F.relu(self.conv4(x2))
        p2 = self.pool2(x2)

        x3 = F.relu(self.conv5(p2))
        x3 = F.relu(self.conv6(x3))
        p3 = self.pool3(x3)

        x4 = F.relu(self.conv7(p3))
        x4 = F.relu(self.conv8(x4))
        p4 = self.pool4(x4)

        # Bottleneck
        x5 = F.relu(self.conv9(p4))
        x5 = F.relu(self.conv10(x5))

        # Decoder with skip connections
        up1 = self.upconv1(x5)  # Upsample
        up1 = torch.cat((up1, x4), dim=1)  # Skip connection
        up1 = F.relu(self.conv11(up1))
        up1 = F.relu(self.conv12(up1))

        up2 = self.upconv2(up1)
        up2 = torch.cat((up2, x3), dim=1)
        up2 = F.relu(self.conv13(up2))
        up2 = F.relu(self.conv14(up2))

        up3 = self.upconv3(up2)
        up3 = torch.cat((up3, x2), dim=1)
        up3 = F.relu(self.conv15(up3))
        up3 = F.relu(self.conv16(up3))

        up4 = self.upconv4(up3)
        up4 = torch.cat((up4, x1), dim=1)
        up4 = F.relu(self.conv17(up4))
        up4 = F.relu(self.conv18(up4))

        # Final output layer
        output = self.output_conv(up4)  # Final 1x1 convolution for classification
        return output