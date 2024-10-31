import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Down double conv layers
        self.down1 = DoubleConv(in_channels=1, out_channels=32)
        self.down2 = DoubleConv(in_channels=32, out_channels=64)
        self.down3 = DoubleConv(in_channels=64, out_channels=128)

        # Bottle neck
        self.down4 = DoubleConv(in_channels=128, out_channels=256)

        #max pool layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Up transpose layers + Double Conv
        self.up_trans1 = nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
        self.up1 = DoubleConv(256, 128)


        self.up_trans2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2,stride=2)
        self.up2 = DoubleConv(128, 64) 

        self.up_trans3 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2)
        self.up3 = DoubleConv(64, 32)


        self.final = nn.Conv2d(in_channels=32,out_channels=6,kernel_size=1)


    def forward(self, initial):


        # down 
        c1 = self.down1(initial)
        p1 = self.max_pool(c1)

        c2 = self.down2(p1)
        p2 = self.max_pool(c2)

        c3 = self.down3(p2)
        p3 = self.max_pool(c3)



        c4 = self.down4(p3)

        # upsample
        t1 = self.up_trans1(c4)
        d1 = self.up1(torch.cat([t1, c3], 1))

        t2 = self.up_trans2(d1)
        d2 = self.up2(torch.cat([t2, c2], 1))

        t3 = self.up_trans3(d2)
        d3 = self.up3(torch.cat([t3, c1], 1))


        # output
        out = self.final(d3)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride= 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x)
