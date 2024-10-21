import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, init_features=32):
        super(UNet3D, self).__init__()

        features = init_features
        # Encoder
        self.encoder1 = UNet3D._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder5 = UNet3D._block(features * 8, features * 16)  # New layer
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = UNet3D._block(features * 16, features * 32)

        # Decoder
        self.upconv5 = nn.ConvTranspose3d(features * 32, features * 16, kernel_size=2, stride=2)
        self.decoder5 = UNet3D._block(features * 16 * 2, features * 16)  # New layer
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block(features * 8 * 2, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block(features * 4 * 2, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block(features * 2 * 2, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features)

        # Final Conv Layer
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))  # New layer

        # Bottleneck
        bottleneck = self.bottleneck(self.pool5(enc5))

        # Decoder path
        dec5 = self.upconv5(bottleneck)  # New layer
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)

        dec4 = self.upconv4(dec5)
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

        return self.softmax(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features):
        # Simple convolutional block with two Conv3d + ReLU layers
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=features),
            nn.ReLU(inplace=True)
        )
