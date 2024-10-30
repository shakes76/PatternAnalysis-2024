'''containing the source code of the components of your model. Each component must be
implementated as a class or a function
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(UNet, self).__init__()

        # down path with max pooling
        self.encoder_conv1 = self.conv_block(in_channels, 64)
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv2 = self.conv_block(64, 128)
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv3 = self.conv_block(128, 256)
        self.encoder_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv4 = self.conv_block(256, 512)
        self.encoder_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = self.conv_block(512, 1024)

        # up path
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
        Double convolution block with Relu activation function
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
        # At eacj stage ensure that size matchs corresponding down path level
        # If not resize accordingly
        if u6.size() != c4.size():
            u6 = F.interpolate(u6, size=c4.size()[2:], mode='bilinear', align_corners=True)
        concat6 = torch.cat([u6, c4], dim=1)
        c6 = self.decoder_conv6(concat6)

        u7 = self.upconv7(c6)
        if u7.size() != c3.size():
            u7 = F.interpolate(u7, size=c3.size()[2:], mode='bilinear', align_corners=True)
        concat7 = torch.cat([u7, c3], dim=1)
        c7 = self.decoder_conv7(concat7)

        u8 = self.upconv8(c7)
        if u8.size() != c2.size():
            u8 = F.interpolate(u8, size=c2.size()[2:], mode='bilinear', align_corners=True)
        concat8 = torch.cat([u8, c2], dim=1)
        c8 = self.decoder_conv8(concat8)

        u9 = self.upconv9(c8)
        if u9.size() != c1.size():
            u9 = F.interpolate(u9, size=c1.size()[2:], mode='bilinear', align_corners=True)
        concat9 = torch.cat([u9, c1], dim=1)
        c9 = self.decoder_conv9(concat9)

        # Output Layer
        output = self.output_conv(c9)

        return output


