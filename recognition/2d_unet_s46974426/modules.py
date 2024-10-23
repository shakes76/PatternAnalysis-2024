import torch
import torch.nn as nn

'''
PyTorch UNet initialiser to be called in other python scripts
Initialisation done for common medical image segmentation
'''
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        # reduce spatial dimensions, increases channels
        # aims to learn more complex features at higher resolutions
        self.enc1 = self.contract_block(in_channels, 64)
        self.enc2 = self.contract_block(64, 128)
        self.enc3 = self.contract_block(128, 256)
        self.enc4 = self.contract_block(256, 512)
        
        # deepest abstract level of UNet (bottleneck)
        self.bottleneck = self.contract_block(512, 1024)
        
        # decoder to mirror contractin path above
        self.upconv4 = self.expand_block(1024, 512)
        self.upconv3 = self.expand_block(512, 256)
        self.upconv2 = self.expand_block(256, 128)
        self.upconv1 = self.expand_block(128, 64)
        
        # final layer kernal size of 1 because binary segmentation
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def __call__(self, x):
        # contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # bottleneck
        bottleneck = self.bottleneck(enc4)
        
        # expanding path
        upconv4 = self.upconv4(bottleneck, enc4)
        upconv3 = self.upconv3(upconv4, enc3)
        upconv2 = self.upconv2(upconv3, enc2)
        upconv1 = self.upconv1(upconv2, enc1)
        
        return torch.sigmoid(self.final(upconv1))
    
    '''
    UNet encoder - reduces the spatial dimensions while increasing the number of feature channels
    '''
    def contract_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    '''
    UNet decoder - upsamples the feature maps back to the original image dimensions while reducing the number of channels
    '''
    def expand_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.ReLU()
        )

