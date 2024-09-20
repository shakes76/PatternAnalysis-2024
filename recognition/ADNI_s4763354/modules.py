import torch
import torch.nn as nn
from gfnet import gfnet_tiny  

class GFNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GFNetBinaryClassifier, self).__init__()
        self.model = gfnet_tiny(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

