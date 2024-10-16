import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Contracting path
        self.enc1 = self.contracting_block(in_channels, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)

        # Bottleneck
        self.bottleneck = self.contracting_block(512, 1024)

        # Expanding path (adjust in_channels for concatenated skip connections)
        self.dec4 = self.expanding_block(1024 + 512, 512)  # 1024 from bottleneck + 512 from enc4
        self.dec3 = self.expanding_block(512 + 256, 256)   # 512 from dec4 + 256 from enc3
        self.dec2 = self.expanding_block(256 + 128, 128)   # 256 from dec3 + 128 from enc2
        self.dec1 = self.expanding_block(128 + 64, 64)     # 128 from dec2 + 64 from enc1

        # Final output layer
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Downsampling
        )

    def expanding_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),  # Upsampling
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        print("enc1 shape:", enc1.shape)  # Debug print
        enc2 = self.enc2(enc1)
        print("enc2 shape:", enc2.shape)  # Debug print
        enc3 = self.enc3(enc2)
        print("enc3 shape:", enc3.shape)  # Debug print
        enc4 = self.enc4(enc3)
        print("enc4 shape:", enc4.shape)  # Debug print

        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        print("bottleneck shape:", bottleneck.shape)  # Debug print

        # Decoding path
        dec4 = self.dec4(self.center_crop(bottleneck, enc4))  # Crop bottleneck to match enc4
        print("dec4 shape before concat:", dec4.shape)  # Debug print
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection with enc4
        print("dec4 shape after concat:", dec4.shape)  # Debug print
        dec3 = self.dec3(self.center_crop(dec4, enc3))  # Crop dec4 to match enc3
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection with enc3
        dec2 = self.dec2(self.center_crop(dec3, enc2))  # Crop dec3 to match enc2
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection with enc2
        dec1 = self.dec1(self.center_crop(dec2, enc1))  # Crop dec2 to match enc1
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection with enc1

        return self.final_conv(dec1)

    def center_crop(self, tensor, target_tensor):
        """
        Crops the input tensor to match the spatial size of the target tensor.
        """
        _, _, d, h, w = target_tensor.size()  # Get spatial dimensions of target tensor
        tensor_cropped = tensor[:, :, :d, :h, :w]  # Crop tensor to match target dimensions
        return tensor_cropped

