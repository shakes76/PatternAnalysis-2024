import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        # Encoder part (downsampling)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck part
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding="same"),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding="same"),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # Decoder part (upsampling)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv11 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv15 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv17 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Final output layer
        self.output_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        p1 = self.pool1(x1)

        x2 = self.conv3(p1)
        x2 = self.conv4(x2)
        p2 = self.pool2(x2)

        x3 = self.conv5(p2)
        x3 = self.conv6(x3)
        p3 = self.pool3(x3)

        x4 = self.conv7(p3)
        x4 = self.conv8(x4)
        p4 = self.pool4(x4)

        # Bottleneck
        x5 = self.conv9(p4)
        x5 = self.conv10(x5)

        # Decoder with skip connections
        up1 = self.upconv1(x5)
        up1 = torch.cat((up1, x4), dim=1)
        up1 = self.conv11(up1)
        up1 = self.conv12(up1)

        up2 = self.upconv2(up1)
        up2 = torch.cat((up2, x3), dim=1)
        up2 = self.conv13(up2)
        up2 = self.conv14(up2)

        up3 = self.upconv3(up2)
        up3 = torch.cat((up3, x2), dim=1)
        up3 = self.conv15(up3)
        up3 = self.conv16(up3)

        up4 = self.upconv4(up3)
        up4 = torch.cat((up4, x1), dim=1)
        up4 = self.conv17(up4)
        up4 = self.conv18(up4)

        # Final output layer
        output = self.output_conv(up4)
        return output