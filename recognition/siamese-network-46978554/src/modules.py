"""Model components"""

import torch.nn as nn
from torchvision.models import resnet50


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.resnet = resnet50()

    def forward(self, x1, x2):
        out1 = self.resnet(x1)
        out2 = self.resnet(x2)

        return out1, out2
