""" 
File: modules.py
Author: Øystein Kvandal
Description: Contains the 2D UNET model for the medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()

        # Downstream
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv2d(1, 32)
        self.down_conv_2 = double_conv2d(32, 64)
        self.down_conv_3 = double_conv2d(64, 128)
        self.down_conv_4 = double_conv2d(128, 256)

        # Upstream
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=128,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_2 = double_conv2d(256, 128)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=128,
                                            out_channels=64,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_3 = double_conv2d(128, 64)
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=64,
                                            out_channels=32,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_4 = double_conv2d(64, 32)
        self.out = nn.Conv2d(in_channels=32,
                            out_channels=6,
                            kernel_size=1)
    

    def forward(self, image):
        # Encoding
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        # x8 = self.max_pool_2x2(x7)
        # x9 = self.down_conv_5(x8)

        ### Decoding

        # x = self.up_trans_1(x9)
        # print(x.size())
        # x = self.up_conv_1(torch.cat([x, x7], 1))
        
        x = self.up_trans_2(x7)

        x = self.up_conv_2(torch.cat([x, x5], 1))

        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))

        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))

        x = self.out(x)
        return x

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-14

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)        
        targets = F.one_hot(targets, num_classes=6)
        targets = targets.squeeze(1)
        targets = targets.permute(0, 3, 1, 2).float()

        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(inputs.size(0), inputs.size(1), -1)

        intersect = torch.abs(inputs * targets).sum()
        dice = (2 * intersect + self.smooth) / (torch.abs(inputs).sum(-1).sum() + torch.abs(targets).sum(-1).sum() + self.smooth)
        return abs(1 - dice.mean())

def double_conv2d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


# def crop_image(tensor, target_tensor):
#     border = (tensor.size()[2] - target_tensor.size()[2]) // 2
#     return tensor[:, :, border:(tensor.size()[2] - border), border:(tensor.size()[3] - border)]