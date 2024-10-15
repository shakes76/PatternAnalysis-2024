import torch
import torch.nn as nn
from timm import create_model

class GFnetClassifier(nn.Module):
    def __init__(self, num_classes = 2):
        super(GFnetClassifier, self).__init__()
        self.backbone = create_model('gfnet_base', pretrained = False)
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

    def forward(self,x):
        return self.backbone(x)
    
    