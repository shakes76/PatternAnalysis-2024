import torch
import torch.nn as nn
from GFNet.gfnet import GFNet  

class GFNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GFNetBinaryClassifier, self).__init__()
        self.model = GFNet(num_classes=num_classes,drop_rate=0.3, dropcls=0.2)
    
    def forward(self, x):
        return self.model(x)

