import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class doubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),  
		)
    def forward(self, x):
        return self.conv(x)    
        
class uNet(nn.Module):
    def __init__(
            self, in_channels = 3, out_channels = 1, features=[64,128,256,512],
		):
        super(uNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
		#uNet down part
        for feature in features:
            self.downs.append(doubleConv(in_channels, feature))
            in_channels = feature
            
		#uNet up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size = 2, stride=2
				)
			)
            self.ups.append(doubleConv(feature*2, feature))
            
        self.bottleneck = doubleConv(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skipConnections = []
        
        for down in self.downs:
            x = down(x)  
            skipConnections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skipConnections = skipConnections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skipConnection = skipConnections[idx//2]
            if x.shape != skipConnection.shape:
                x = TF.resize(x, size = skipConnection.shape[2:])
            concatSkip = torch.cat((skipConnection, x), dim=1)
            x = self.ups[idx+1](concatSkip)
            
        return self.final_conv(x)

