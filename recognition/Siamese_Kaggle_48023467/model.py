# Model definitions of Siamese Network

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class Siamese(nn.Module):
    def __init__(self, dropout=0.3, margin=1.0):
        super(Siamese, self).__init__()
        
        # Load pre-trained ResNet model
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Strip ther final FC layer as it is implemented seperately in Siamese
        self.resnet = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Evaluate number of output features for FC
        resnet_out_features = self.resnet.fc.in_features * 2
        
        # Freeze parameters of feature extractor
        for param in self.resnet.parameters():
            param.requires_grad = True
            
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(resnet_out_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, 64)
        )
        
        # Seperately definite the classifier for predictions
        self.classifier = nn.Sequential(
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        
        self.tripletLoss = nn.TripletMarginLoss(margin=1.0)
    
        return
    
    # Forward pass once
    def forward_once(self, data):
        out = self.resnet(data)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
    
    # Classify siamese results into classes
    def classify(self, data):
        return self.classifier(data)
    
    # Main forward pass to be fed into loss functio
    def forward(self, data, pos, neg):

        out_data = self.forward_once(data)
        out_pos = self.forward_once(pos)
        out_neg = self.forward_once(neg)
        return out_data, out_pos, out_neg