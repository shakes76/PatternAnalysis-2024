"""
Model architecture for a GFNet image transformer 

Benjamin Thatcher 
s4784738    
"""

import torch
import torch.nn as nn
import timm  # For GFNet and pretrained models

class GFNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(GFNetClassifier, self).__init__()
        
        # Load GFNet without pretrained weights from timm
        self.gfnet = timm.create_model('gfnet_base', pretrained=pretrained)
        
        # Replace the classification head to match the number of output classes
        in_features = self.gfnet.head.fc.in_features
        self.gfnet.head.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.gfnet(x)

# Function to get the model instance
def get_model(num_classes=2, pretrained=False):
    model = GFNetClassifier(num_classes=num_classes, pretrained=pretrained)
    return model
        