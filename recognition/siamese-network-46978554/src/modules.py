"""Model components"""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=False):
        super(SiameseNetwork, self).__init__()

        # Use a ResNet backbone without the last FC layer (we'll define our own)
        if pretrained:
            resnet_base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet_base = resnet50()
        self.net = nn.Sequential(
            *list(resnet_base.children())[:-1],
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64, bias=True)
        )

        if pretrained:
            for param in self.net[:-3].parameters():
                param.requires_grad = False

    def forward(self, x1, x2):
        out1 = self.net(x1)
        out2 = self.net(x2)

        return out1, out2
