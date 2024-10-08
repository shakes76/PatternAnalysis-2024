import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=64, dropout_rate=0.3):
        super(SiameseNetwork, self).__init__()

        # Load pre-trained ResNet50 model
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # ResNet50 output features
        resnet_output_features = 2048
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(resnet_output_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, embedding_dim)
        )
        
        # Classifier
        self.classifier = nn.Linear(embedding_dim, 2)
    
    def forward_once(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out

    def forward(self, x1, x2, x3):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out3 = self.forward_once(x3)
        return out1, out2, out3
    
    def get_embedding(self, x):
        return self.forward_once(x)
    
    def classify(self, x):
        embedding = self.get_embedding(x)
        return self.classifier(embedding)

# Triplet Loss Function remains the same
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        loss = F.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()

# Basic helpers to later be imported in train.py
def get_model(embedding_dim=128):
    return SiameseNetwork(embedding_dim=embedding_dim)

def get_loss(margin=1.0):
    return TripletLoss(margin)